"""Dilated Gated Attention: LongNet-style dilation + Gated Sparse token selection.

Combines:
  - LongNet (Ding et al., 2023, arXiv:2307.02486): dilated attention windows
  - Gated Sparse Attention (Shen & Shen, 2026, arXiv:2601.15305): learned sparse gates

For each layer l with dilation d_l and window w:
  1. Candidates = tokens at positions i ± k·d_l for k ∈ [1, w]
  2. Gate predicts which candidates to actually attend to
  3. Attention computed only over gated-in candidates
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bid_slicing.models.gated_sparse import GatedSparsifier


def _build_dilated_candidate_mask(
    seq_len: int,
    window_size: int,
    dilation: int,
    num_global: int,
    device: torch.device,
) -> torch.Tensor:
    """Build candidate (pre-filter) mask for dilated attention.

    Returns [S, S] where 1 = candidate, 0 = excluded.
    S = num_global + seq_len blocks.
    """
    N = seq_len
    G = num_global
    S = G + N
    mask = torch.zeros(S, S, device=device)

    # Global tokens: candidates with everything
    mask[:G, :] = 1.0

    # Block tokens: dilated window candidates
    for i in range(N):
        gi = G + i
        mask[gi, :G] = 1.0  # all global tokens
        mask[gi, gi] = 1.0  # self

        for k in range(1, window_size + 1):
            left = i - k * dilation
            right = i + k * dilation
            if left >= 0:
                mask[gi, G + left] = 1.0
            if right < N:
                mask[gi, G + right] = 1.0

    return mask


class DilatedGatedAttentionLayer(nn.Module):
    """Single layer with dilated window candidates + gated sparse selection."""

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        window_size: int = 5,
        dilation: int = 1,
        num_global: int = 4,
        sparsity_k: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.dilation = dilation
        self.num_global = num_global
        self.scale = self.head_dim ** 0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.gate = GatedSparsifier(
            dim=dim,
            head_dim=self.head_dim,
            mode="topk",
            k=sparsity_k,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, G+N, dim] → [B, G+N, dim]"""
        B, S, D = x.shape
        H = self.num_heads
        hd = self.head_dim
        N = S - self.num_global

        q = self.q_proj(x).view(B, S, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, hd).transpose(1, 2)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Build dilated candidate mask
        cand_mask = _build_dilated_candidate_mask(
            N, self.window_size, self.dilation, self.num_global, x.device
        )

        # Gated sparse selection within dilated candidates
        gate_mask = self.gate(q, k, cand_mask)

        attn = attn + gate_mask
        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class DilatedGatedPageEncoder(nn.Module):
    """LongNet + Gated Sparse Page Encoder.

    Layers: dilation = [1, 2, 4, 8, ...] (exponential growth)
    Each layer uses gated sparse selection within its dilated window.
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        base_window_size: int = 5,
        num_global: int = 4,
        sparsity_k: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_global = num_global
        self.num_layers = num_layers

        # Learnable global tokens
        self.global_tokens = nn.Parameter(torch.randn(num_global, dim) * 0.02)

        # Layers with exponentially growing dilation
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            dilation = 2 ** l
            self.layers.append(
                nn.ModuleDict({
                    "attn": DilatedGatedAttentionLayer(
                        dim=dim,
                        num_heads=num_heads,
                        window_size=base_window_size,
                        dilation=dilation,
                        num_global=num_global,
                        sparsity_k=sparsity_k,
                        dropout=dropout,
                    ),
                    "norm1": nn.LayerNorm(dim),
                    "ffn": nn.Sequential(
                        nn.Linear(dim, dim * 4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(dim * 4, dim),
                        nn.Dropout(dropout),
                    ),
                    "norm2": nn.LayerNorm(dim),
                })
            )

        self.pool_proj = nn.Sequential(
            nn.Linear(dim * num_global, dim),
            nn.LayerNorm(dim),
        )

    def forward(
        self, block_vectors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (page_repr [B, dim], encoded_blocks [B, N, dim])."""
        B, N, D = block_vectors.shape
        G = self.num_global

        global_t = self.global_tokens.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([global_t, block_vectors], dim=1)

        for layer in self.layers:
            x = x + layer["attn"](layer["norm1"](x))
            x = x + layer["ffn"](layer["norm2"](x))

        global_out = x[:, :G, :]
        page_repr = self.pool_proj(global_out.reshape(B, G * D))
        encoded_blocks = x[:, G:, :]

        return page_repr, encoded_blocks