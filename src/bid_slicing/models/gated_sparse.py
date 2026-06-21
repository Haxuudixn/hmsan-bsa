"""Gated Sparse Attention: learnable gates to dynamically select sparse connections.

Based on "Gated Sparse Attention: Combining Computational Efficiency with
Training Stability for Long-Context" (Shen & Shen, 2026, arXiv:2601.15305).

Core insight: replace fixed sparsity patterns with learned gating that predicts
which token pairs should interact. Uses straight-through estimator for gradients
through discrete top-k / threshold sparsification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedSparsifier(nn.Module):
    """Learnable gate that produces sparse attention mask from Q-K pairs.

    Two modes:
      - topk: keep top-k connections per query (deterministic sparsity)
      - threshold: keep connections where gate > τ (adaptive sparsity)
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        mode: str = "topk",
        k: int = 16,
        threshold: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.mode = mode
        self.k = k
        self.threshold = threshold
        self.temperature = temperature

        # Gate network: [Q; K] → scalar
        self.gate_proj = nn.Sequential(
            nn.Linear(head_dim * 2, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, 1),
        )

    def forward(
        self,
        q: torch.Tensor,           # [B, H, S, hd]   or  [B, H, S_q, hd]
        k: torch.Tensor,           # [B, H, S_k, hd]
        candidate_mask: torch.Tensor | None = None,  # [S_q, S_k] optional pre-filter
    ) -> torch.Tensor:
        """Returns gated additive mask [1, 1, S_q, S_k] (0 = keep, -inf = mask)."""
        B, H, Sq, hd = q.shape
        Sk = k.shape[2]

        # Compute gate scores: g_ij = σ(W·[q_i; k_j])
        q_exp = q.unsqueeze(3).expand(-1, -1, -1, Sk, -1)  # [B, H, Sq, Sk, hd]
        k_exp = k.unsqueeze(2).expand(-1, -1, Sq, -1, -1)  # [B, H, Sq, Sk, hd]
        pair = torch.cat([q_exp, k_exp], dim=-1)            # [B, H, Sq, Sk, 2hd]
        gates = torch.sigmoid(self.gate_proj(pair) / self.temperature).squeeze(-1)  # [B, H, Sq, Sk]

        # Apply candidate pre-filter
        if candidate_mask is not None:
            gates = gates * candidate_mask.unsqueeze(0).unsqueeze(0)

        # Sparsify
        if self.mode == "topk":
            actual_k = min(self.k, Sk)
            _, topk_idx = torch.topk(gates, k=actual_k, dim=-1)
            sparse_mask = torch.zeros_like(gates)
            sparse_mask.scatter_(-1, topk_idx, 1.0)
        elif self.mode == "threshold":
            sparse_mask = (gates > self.threshold).float()
            # Ensure at least 1 connection per query
            sparse_mask.scatter_(-1, gates.argmax(dim=-1, keepdim=True), 1.0)
        else:
            sparse_mask = (gates > 0).float()

        # Straight-through: use sparse_mask for forward, gates for backward
        # This allows gradient flow through the discrete mask
        if self.training:
            sparse_mask = sparse_mask + gates - gates.detach()

        # Convert to additive mask: 0 → keep, -inf → mask
        attn_mask = (1.0 - sparse_mask) * -1e9
        return attn_mask  # [B, H, Sq, Sk]


class GatedMultiHeadAttention(nn.Module):
    """Multi-head attention with gated sparse connection selection."""

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        sparsity_mode: str = "topk",
        sparsity_k: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.gate = GatedSparsifier(
            dim=dim,
            head_dim=self.head_dim,
            mode=sparsity_mode,
            k=sparsity_k,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [B, S, dim] → [B, S, dim]"""
        B, S, D = x.shape
        H = self.num_heads
        hd = self.head_dim

        q = self.q_proj(x).view(B, S, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, hd).transpose(1, 2)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Gated sparse mask
        gate_mask = self.gate(q, k, candidate_mask)

        # Apply mask
        attn = attn + gate_mask
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class GatedFusionGate(nn.Module):
    """Learned gate for modality fusion: predicts per-sample modality importance.

    Used in BlockMultiModalEncoder to selectively blend text / layout / image
    features based on block type and layout cues.
    """

    def __init__(self, dim: int, num_modalities: int = 3):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_modalities),
        )

    def forward(
        self,
        layout_feat: torch.Tensor,     # [B, layout_dim]
        modality_feats: list[torch.Tensor],  # list of [B, dim]
    ) -> torch.Tensor:
        """Returns gated fusion [B, dim]."""
        gates = F.softmax(self.gate_net(layout_feat), dim=-1)  # [B, M]
        stacked = torch.stack(modality_feats, dim=-1)          # [B, dim, M]
        fused = torch.bmm(stacked, gates.unsqueeze(-1)).squeeze(-1)  # [B, dim]
        return fused