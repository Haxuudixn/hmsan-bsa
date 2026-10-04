from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfiniMemory(nn.Module):
    """Infini-attention compressive memory (arXiv:2404.07143).

    Compresses token/page representations into a fixed-size matrix M ∈ R^(d×d)
    via incremental outer-product updates. Retrieval uses linear attention
    (ELU+1 activation), achieving O(d²) memory and O(d²) per-step retrieval.

    Mathematical formulation:
      Activation:  σ(x) = ELU(x) + 1
      Retrieval:   A = σ(Q)·M_{t-1} / (σ(Q)·z_{t-1} + ε)
      Update:      M_t = M_{t-1} + σ(K)^T·V
                   z_t = z_{t-1} + Σσ(K)

    References:
      Munkhdalai et al., "Leave No Context Behind", 2024.
    """

    def __init__(
        self,
        dim: int = 128,
        key_dim: int | None = None,
        value_dim: int | None = None,
        dropout: float = 0.1,
        use_delta_rule: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.key_dim = key_dim or dim
        self.value_dim = value_dim or dim
        self.use_delta_rule = use_delta_rule
        self.eps = 1e-8

        # Projections for key, value, query
        self.W_k = nn.Linear(dim, self.key_dim, bias=False)
        self.W_v = nn.Linear(dim, self.value_dim, bias=False)
        self.W_q = nn.Linear(dim, self.key_dim, bias=False)

        # Output projection (maps value_dim back to dim)
        self.out_proj = nn.Sequential(
            nn.Linear(self.value_dim, dim),
            nn.Dropout(dropout),
        )

        # Learnable gate for combining current input with memory readout
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

        # Delta-rule: learnable beta for removing retrieved info from memory
        if use_delta_rule:
            self.W_beta = nn.Linear(dim, self.key_dim, bias=False)

    @staticmethod
    def _activation(x: torch.Tensor) -> torch.Tensor:
        """σ(x) = ELU(x) + 1  (non-negative, non-linear)."""
        return F.elu(x) + 1.0

    def _init_state(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize memory matrix and normalization vector."""
        M = torch.zeros(batch_size, self.key_dim, self.value_dim, device=device)
        z = torch.zeros(batch_size, self.key_dim, device=device)
        return M, z

    def _retrieve(
        self,
        query: torch.Tensor,      # [B, dim]
        M: torch.Tensor,          # [B, kd, vd]
        z: torch.Tensor,          # [B, kd]
    ) -> torch.Tensor:
        """Retrieve compressed context via linear attention.

        A = σ(Q)·M / (σ(Q)·z + ε)
        """
        Q = self.W_q(query)                         # [B, kd]
        Q_act = self._activation(Q)                  # [B, kd]

        # Numerator: σ(Q)·M  → [B, vd]
        num = torch.bmm(Q_act.unsqueeze(1), M).squeeze(1)

        # Denominator: σ(Q)·z + ε  → [B]
        den = (Q_act * z).sum(dim=-1, keepdim=True) + self.eps

        return num / den  # [B, vd]

    def _update(
        self,
        key: torch.Tensor,        # [B, dim]
        value: torch.Tensor,      # [B, dim]
        M: torch.Tensor,          # [B, kd, vd]
        z: torch.Tensor,          # [B, kd]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Incremental memory update.

        M_new = M_old + σ(K)^T·V
        z_new = z_old + Σσ(K)
        """
        K = self.W_k(key)                            # [B, kd]
        V = self.W_v(value)                          # [B, vd]
        K_act = self._activation(K)                  # [B, kd]

        # Outer product update: σ(K)^T·V  → [B, kd, vd]
        M_new = M + torch.bmm(
            K_act.unsqueeze(-1), V.unsqueeze(1)
        )
        z_new = z + K_act                            # [B, kd]

        return M_new, z_new

    def forward(
        self,
        x: torch.Tensor,                 # [B, dim]
        M: torch.Tensor,                 # [B, kd, vd]
        z: torch.Tensor,                 # [B, kd]
        update_mask: torch.Tensor | None = None,  # [B] float ∈ [0,1], 0=skip update
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process one step.

        Returns:
            output:   [B, dim]   — gated combination of input + memory readout
            M_new:    [B, kd, vd]
            z_new:    [B, kd]
        """
        # ── Retrieve ──
        mem_readout = self._retrieve(x, M, z)         # [B, vd]
        mem_out = self.out_proj(mem_readout)           # [B, dim]

        # ── Gate: combine current input with memory readout ──
        gate_input = torch.cat([x, mem_out], dim=-1)
        g = self.gate(gate_input)                      # [B, dim]
        output = g * x + (1 - g) * mem_out             # [B, dim]

        # ── Update (with optional mask) ──
        if update_mask is not None:
            # Scale key activation by mask to reduce update contribution
            key = x
            value = x
            K = self.W_k(key)                          # [B, kd]
            V = self.W_v(value)                        # [B, vd]
            K_act = self._activation(K) * update_mask.unsqueeze(-1)
            M_new = M + torch.bmm(K_act.unsqueeze(-1), V.unsqueeze(1))
            z_new = z + K_act
        else:
            M_new, z_new = self._update(x, x, M, z)

        return output, M_new, z_new


class InfiniPageMemory(InfiniMemory):
    """Infini-attention page-level compressive memory.

    Compresses all past page representations into M ∈ R^(d×d).
    No boundary gating — every page contributes equally to the memory.
    """

    def __init__(self, dim: int = 128, dropout: float = 0.1):
        super().__init__(dim=dim, dropout=dropout)

    def init_state(self, batch_size: int, device: torch.device):
        return self._init_state(batch_size, device)

    def step(
        self,
        page_repr: torch.Tensor,       # [B, dim]
        M: torch.Tensor,               # [B, d, d]
        z: torch.Tensor,               # [B, d]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (enhanced_page [B,dim], M_new, z_new)."""
        return self.forward(page_repr, M, z)


class InfiniSectionMemory(InfiniMemory):
    """Infini-attention section-level compressive memory with boundary gating.

    Extends InfiniMemory with boundary-aware update masking:
      - B-SECTION → update_mask ≈ 0    (new section, old memory not updated)
      - I/E-SECTION → update_mask ≈ 1  (continuation, full update)
    """

    def __init__(
        self,
        dim: int = 128,
        num_boundaries: int = 4,
        dropout: float = 0.1,
        boundary_gate: bool = True,
    ):
        super().__init__(dim=dim, dropout=dropout)
        self.use_boundary_gate = boundary_gate

        # Boundary-sensitive mask predictor.  A7 removes it entirely, which
        # degrades the section memory to a plain every-page Infini update.
        if boundary_gate:
            self.boundary_gate = nn.Sequential(
                nn.Linear(dim + num_boundaries, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
                nn.Sigmoid(),
            )

    def init_state(self, batch_size: int, device: torch.device):
        return self._init_state(batch_size, device)

    def step(
        self,
        page_repr: torch.Tensor,          # [B, dim]
        boundary_logits: torch.Tensor,    # [B, 4]
        M: torch.Tensor,                  # [B, d, d]
        z: torch.Tensor,                  # [B, d]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (enhanced_page [B,dim], M_new, z_new)."""
        if not self.use_boundary_gate:
            # A7: no boundary-aware update masking.
            return self.forward(page_repr, M, z)

        b_soft = F.softmax(boundary_logits, dim=-1)
        gate_input = torch.cat([page_repr, b_soft], dim=-1)
        update_mask = self.boundary_gate(gate_input).squeeze(-1)  # [B]

        return self.forward(page_repr, M, z, update_mask=update_mask)