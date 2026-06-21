from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ═══════════════════════════════════════════════════════════════════════
# GSA: Gated Sparse Attention (Shen & Shen, 2026)
# Adapted for page-level block encoding with global tokens
# ═══════════════════════════════════════════════════════════════════════

class GatedSparseLocalAttention(nn.Module):
    """Gated Sparse Attention (GSA) adapted for page-level block encoding.

    Core components from Shen & Shen (2026):
      1. Gated Lightning Indexer — sigmoid-based block importance scoring
      2. Adaptive Sparsity — variance-based dynamic k selection
      3. Dual Gating — G1 (output gate) + G2 (value gate)

    Sequence layout: [global_tokens (G)] + [block_tokens (N)]
    - Global tokens attend to everything (full attention)
    - Block tokens use GSA: indexer → top-k → sparse SDPA
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        k_base: int = 16,
        k_min: int = 4,
        k_max: int = 32,
        indexer_heads: int = 4,
        indexer_dim: int = 32,
        num_global: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.num_global = num_global

        # ── SDPA projections ──
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # ── G1: Output Gate (element-wise sigmoid) ──
        self.g1_proj = nn.Linear(dim, dim)

        # ── G2: Value Gate (modulate values before aggregation) ──
        self.g2_proj = nn.Linear(dim, dim)

        # ── Gated Lightning Indexer ──
        # Low-dimensional projections for cheap all-pair scoring
        # I(s) = Σ_j σ(w_j) · σ(q_j · k_s + b_j)
        self.indexer_dim = indexer_dim
        self.indexer_heads = indexer_heads
        self.W_iq = nn.Parameter(torch.randn(indexer_heads, dim, indexer_dim) * 0.02)
        self.W_ik = nn.Parameter(torch.randn(indexer_heads, dim, indexer_dim) * 0.02)
        self.W_iw = nn.Parameter(torch.randn(indexer_heads, dim) * 0.02)
        self.b_i = nn.Parameter(torch.zeros(indexer_heads))

        # ── Adaptive Sparsity ──
        self.k_base = k_base
        self.k_min = k_min
        self.k_max = k_max
        self.register_buffer("ema_var", torch.tensor(1.0))
        self.ema_decay = 0.99

        self.dropout = nn.Dropout(dropout)

    # ── Gated Lightning Indexer ──

    def _indexer_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Compute all-pair importance scores via gated lightning indexer.

        I(t,s) = Σ_j σ(h_t · W_iw_j) · σ(q_tj · k_sj + b_j)

        Args:
            x: [B, S, dim]  (S = G + N)
        Returns:
            scores: [B, S, S]  (only block→block matters; global ignored)
        """
        B, S, D = x.shape
        HI = self.indexer_heads
        dI = self.indexer_dim

        # Low-dim projections: [B, S, HI, dI]
        qI = torch.einsum("bsd,hde->bshe", x, self.W_iq)
        kI = torch.einsum("bsd,hde->bshe", x, self.W_ik)

        # Head weights: [B, S, HI]
        wI = torch.einsum("bsd,hd->bsh", x, self.W_iw)

        # Cross scores: qI_t · kI_s  → [B, S, S, HI]
        qk = torch.einsum("bthe,bshe->btsh", qI, kI)  # pairwise dot

        # Sigmoid gating on both terms
        scores_per_head = torch.sigmoid(wI.unsqueeze(1)) * torch.sigmoid(
            qk + self.b_i.view(1, 1, 1, HI)
        )  # [B, S, S, HI]

        # Sum over indexer heads → [B, S, S]
        return scores_per_head.sum(dim=-1)  # ∈ (0, HI)

    # ── Adaptive Sparsity ──

    def _adaptive_k(self, scores: torch.Tensor) -> int:
        """Determine selection budget from score variance.

        k = clamp(round(k_base * Var(I) / EMA_Var), k_min, k_max)

        High variance → confident → prune aggressively (smaller k).
        Low variance → ambiguous → retain more context (larger k).
        """
        if not self.training:
            return self.k_base

        with torch.no_grad():
            # Variance over the scored dimension per item
            var = scores.var(dim=-1).mean().item()
            self.ema_var = self.ema_decay * self.ema_var + (1 - self.ema_decay) * var
            ratio = var / (self.ema_var + 1e-8)
            k = int(round(self.k_base * ratio))
            return max(self.k_min, min(self.k_max, k))

    # ── Forward ──

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, G+N, dim] → [B, G+N, dim]"""
        B, S, D = x.shape
        G = self.num_global
        N = S - G
        H = self.num_heads

        q = self.q_proj(x).view(B, S, H, self.head_dim).transpose(1, 2)  # [B,H,S,hd]
        k = self.k_proj(x).view(B, S, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, self.head_dim).transpose(1, 2)

        # ── G2: Value Gate (paper Eq. 9) ──
        g2 = torch.sigmoid(self.g2_proj(x))  # [B, S, D]
        g2_v = g2.view(B, S, H, self.head_dim).transpose(1, 2)
        v = v * g2_v

        # ── Compute all-pair attention ──
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, S, S]

        if N > 0:
            # ── Gated Lightning Indexer for block tokens ──
            indexer_scores = self._indexer_scores(x)  # [B, S, S]

            # Adaptive k
            # Only look at block→block portion for variance
            block_scores = indexer_scores[:, G:, G:]  # [B, N, N]
            k_val = self._adaptive_k(block_scores)
            k_val = min(k_val, N)  # can't select more than N

            # Build sparse mask
            # Global tokens: full attention (no masking)
            mask = torch.ones(B, H, S, S, device=x.device, dtype=torch.bool)

            if k_val < N:
                for b in range(B):
                    # Block → all: select top-k from indexer scores
                    scores_b = indexer_scores[b]  # [S, S]
                    for i in range(G, S):
                        # For position i (block), select top-k based on indexer
                        # Exclude global tokens from top-k selection (they're always attended)
                        block_only = scores_b[i, G:]  # [N]
                        # Select additional top-k beyond globals
                        _, top_k_idx = torch.topk(block_only, k_val)
                        # Map back to full sequence indices
                        full_idx = top_k_idx + G
                        # Set mask: attend to globals + selected blocks
                        row_mask = torch.zeros(S, device=x.device, dtype=torch.bool)
                        row_mask[:G] = True        # always attend globals
                        row_mask[full_idx] = True   # attend top-k blocks
                        mask[b, :, i, :] = mask[b, :, i, :] & row_mask.unsqueeze(0)

            attn = attn.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, S, D)
        attn_out = self.out_proj(context)

        # ── G1: Output Gate (our existing gate, paper's G1) ──
        g1 = torch.sigmoid(self.g1_proj(x))
        return g1 * attn_out + (1.0 - g1) * x


# ═══════════════════════════════════════════════════════════════════════
# Gated FFN (GLU variant) — retained from before
# ═══════════════════════════════════════════════════════════════════════

class GatedFFN(nn.Module):
    """Gated feed-forward with GELU gating.

    h = GELU(W_gate * x) * (W_value * x)
    out = W_out * h
    """

    def __init__(self, dim: int = 128, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expansion
        self.w_gate = nn.Linear(dim, hidden)
        self.w_value = nn.Linear(dim, hidden)
        self.w_out = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.gelu(self.w_gate(x)) * self.w_value(x)
        return self.dropout(self.w_out(gated))


# ═══════════════════════════════════════════════════════════════════════
# Page Transformer Layer
# ═══════════════════════════════════════════════════════════════════════

class PageTransformerLayer(nn.Module):
    """Transformer layer with GSA (Gated Sparse Attention) + GatedFFN."""

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        k_base: int = 16,
        num_global: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = GatedSparseLocalAttention(
            dim=dim,
            num_heads=num_heads,
            k_base=k_base,
            num_global=num_global,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GatedFFN(dim=dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════
# Page Encoder
# ═══════════════════════════════════════════════════════════════════════

class PageEncoder(nn.Module):
    """GSA-based page encoder with gated sparse attention.

    Input:  block_vectors [B, N, dim]
    Output: page_repr      [B, dim]
            encoded_blocks [B, N, dim]
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        k_base: int = 16,
        num_global: int = 4,
        dropout: float = 0.1,
        **kwargs,  # absorb legacy args (window_size, etc.)
    ):
        super().__init__()
        self.dim = dim
        self.num_global = num_global

        self.global_tokens = nn.Parameter(torch.randn(num_global, dim) * 0.02)

        self.layers = nn.ModuleList([
            PageTransformerLayer(
                dim=dim,
                num_heads=num_heads,
                k_base=k_base,
                num_global=num_global,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.pool_proj = nn.Sequential(
            nn.Linear(dim * num_global, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, block_vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = block_vectors.shape
        G = self.num_global
        global_t = self.global_tokens.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([global_t, block_vectors], dim=1)

        for layer in self.layers:
            x = layer(x)

        page_repr = self.pool_proj(x[:, :G, :].reshape(B, G * D))
        encoded_blocks = x[:, G:, :]

        return page_repr, encoded_blocks


# ═══════════════════════════════════════════════════════════════════════
# Gated Inter-Page Attention — retained
# ═══════════════════════════════════════════════════════════════════════

class GatedInterPageAttention(nn.Module):
    """Cross-page attention with learnable gating."""

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        window: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window = window
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        page_repr: torch.Tensor,
        page_history: torch.Tensor,
    ) -> torch.Tensor:
        if page_history.size(1) == 0:
            return page_repr

        query = page_repr.unsqueeze(1)
        attn_out, _ = self.mha(query, page_history, page_history)
        attn_out = attn_out.squeeze(1)

        g = self.gate(torch.cat([page_repr, attn_out], dim=-1))
        return self.norm(g * attn_out + (1.0 - g) * page_repr)


# ── Backward-compatible aliases ──
SparseLocalAttention = GatedSparseLocalAttention
InterPageAttention = GatedInterPageAttention