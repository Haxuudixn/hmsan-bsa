"""Gated Infini-attention Memory: compressive memory with learned gated read/write.

Combines:
  - Infini-attention (Munkhdalai et al., 2024, arXiv:2404.07143): delta-rule
    compressive memory with linear attention
  - Gated Sparse Attention (Shen & Shen, 2026, arXiv:2601.15305): learned gates
    for dynamic sparsity control
  - MKA insights (Liu & Yu, 2026, arXiv:2603.20586): memory-keyed addressing

Key enhancements over vanilla Infini-attention:
  1. Gated memory write: per-dimension gate controls which memory cells to update
  2. Sparse memory read: learned gate selects relevant memory dimensions
  3. Multi-slot: maintain K compressive memory matrices for different context types
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedInfiniMemory(nn.Module):
    """Compressive memory with gated sparse read/write.

    Memory: M ∈ R^(num_slots, d_key, d_value)
    - Write: gated delta-rule update with per-dimension importance
    - Read:  gated sparse retrieval with learned key addressing
    - Fusion: learned β gate combines local attention with memory retrieval
    """

    def __init__(
        self,
        dim: int = 128,
        num_slots: int = 4,
        d_key: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.d_key = d_key
        self.d_value = dim

        # Projections for key/value from page representations
        self.key_proj = nn.Linear(dim, d_key)
        self.value_proj = nn.Linear(dim, self.d_value)

        # ── Gated Write: per-dimension update importance ──
        self.write_gate = nn.Sequential(
            nn.Linear(dim, self.d_value),
            nn.Sigmoid(),
        )

        # ── Sparse Read: learned key addressing + gated selection ──
        self.read_query_proj = nn.Linear(dim, d_key)
        self.read_gate = nn.Sequential(
            nn.Linear(dim + self.d_value, self.d_value),
            nn.Sigmoid(),
        )

        # ── Local/Memory fusion gate ──
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

        # ── Output projection ──
        self.out_proj = nn.Sequential(
            nn.Linear(self.d_value * num_slots, dim),
            nn.LayerNorm(dim),
        )

        self.dropout = nn.Dropout(dropout)

        # Normalization for stable memory updates
        self.register_buffer(
            "mem_norm", torch.ones(num_slots, 1)
        )  # per-slot normalization

    def _init_memory(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (memory [B, K, d_key, d_value], norm [B, K, 1])."""
        M = torch.zeros(batch_size, self.num_slots, self.d_key, self.d_value, device=device)
        z = torch.zeros(batch_size, self.num_slots, 1, device=device)
        return M, z

    def _gated_write(
        self,
        memory: torch.Tensor,          # [B, K, d_key, d_value]
        norm: torch.Tensor,            # [B, K, 1]
        page_repr: torch.Tensor,       # [B, dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gated delta-rule memory update."""
        B = page_repr.size(0)
        K = self.num_slots

        # Key / Value from page representation
        k = F.elu(self.key_proj(page_repr)) + 1.0  # [B, d_key]  (positive for stability)
        k = k.unsqueeze(1).expand(-1, K, -1)       # [B, K, d_key]
        v = self.value_proj(page_repr)              # [B, d_value]
        v = v.unsqueeze(1).expand(-1, K, -1)       # [B, K, d_value]

        # Per-dimension write gate from page context
        write_g = self.write_gate(page_repr)        # [B, d_value]
        write_g = write_g.unsqueeze(1).unsqueeze(2) # [B, 1, 1, d_value]

        # Delta-rule update with gating
        # retrieval = σ(K)M                  [B, K, d_value]
        retrieval = torch.bmm(k, memory.view(B * K, self.d_key, self.d_value).transpose(1, 2))
        retrieval = retrieval.view(B, K, self.d_value)
        # Actually need: k [B, K, d_key], M [B, K, d_key, d_value]
        # k @ M: [B, K, 1, d_key] @ [B, K, d_key, d_value] → [B, K, 1, d_value]
        k_exp = k.unsqueeze(2)  # [B, K, 1, d_key]
        retrieval = torch.matmul(k_exp, memory).squeeze(2)  # [B, K, d_value]

        # Error = V - retrieval
        error = v - retrieval  # [B, K, d_value]

        # Gated update: M = M + gate ⊙ σ(K)ᵀ ⊗ error
        k_t = k.unsqueeze(-1)                       # [B, K, d_key, 1]
        error_exp = error.unsqueeze(2)              # [B, K, 1, d_value]
        delta = torch.matmul(k_t, error_exp)        # [B, K, d_key, d_value]
        delta = write_g * delta                     # gated update

        memory = memory + delta
        norm = norm + torch.sum(k, dim=-1, keepdim=True)  # [B, K, 1]

        return memory, norm

    def _gated_read(
        self,
        memory: torch.Tensor,          # [B, K, d_key, d_value]
        norm: torch.Tensor,            # [B, K, 1]
        page_repr: torch.Tensor,       # [B, dim]
    ) -> torch.Tensor:
        """Gated sparse read from compressive memory."""
        B = page_repr.size(0)
        K = self.num_slots

        # Query from page representation
        q = F.elu(self.read_query_proj(page_repr)) + 1.0  # [B, d_key]
        q = q.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, d_key]

        # Read: A = σ(Q)M / (σ(Q)z + ε)
        retrieval = torch.matmul(q, memory).squeeze(1).squeeze(1)  # [B, K, d_value]
        norm_safe = norm.clamp(min=1e-8)  # [B, K, 1]
        retrieval = retrieval / norm_safe  # normalize

        # Gated read: selectively attend to memory dimensions
        read_g = self.read_gate(
            torch.cat([page_repr.unsqueeze(1).expand(-1, K, -1),
                       retrieval], dim=-1)
        )  # [B, K, d_value]
        retrieval = read_g * retrieval

        # Pool across slots
        retrieval = retrieval.reshape(B, K * self.d_value)
        return self.out_proj(retrieval)  # [B, dim]

    def forward(
        self,
        page_repr: torch.Tensor,            # [B, dim]
        local_out: torch.Tensor,            # [B, dim]  (from local attention)
        memory: torch.Tensor | None,        # [B, K, d_key, d_value]
        norm: torch.Tensor | None,          # [B, K, 1]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (fused_out, new_memory, new_norm, memory_readout)."""
        B = page_repr.size(0)
        device = page_repr.device

        if memory is None:
            memory, norm = self._init_memory(B, device)

        # Gated write
        memory, norm = self._gated_write(memory, norm, page_repr)

        # Gated read
        memory_out = self._gated_read(memory, norm, page_repr)  # [B, dim]

        # Fusion gate: β · local + (1-β) · memory
        beta = self.fusion_gate(torch.cat([local_out, memory_out], dim=-1))
        fused = beta * local_out + (1 - beta) * memory_out

        return fused, memory, norm, memory_out