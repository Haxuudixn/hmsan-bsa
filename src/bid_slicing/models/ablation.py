"""Ablation switches for HMSAN-BSA.

The flags below control the eight gating positions plus the attention mode.
Defaults reproduce the architecture that produced ``outputs/hmsan_bsa_final``
exactly, so existing checkpoints keep loading with ``strict=True``.

Design note: when a switch is off, the corresponding parameters are **not
allocated**, so every preset is a genuine reduced model rather than a masked
copy.  Parameter counts therefore differ slightly between rows and must be
reported per row (see ``scripts/paper/summarize_comparison.py``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

ATTENTION_MODES = ("gsa", "local_window", "dense")


@dataclass(frozen=True)
class AblationFlags:
    """Switch set for one experimental configuration."""

    # ── GSA components ──
    g1_output_gate: bool = True
    g2_value_gate: bool = True
    adaptive_sparsity: bool = True
    attention_mode: str = "gsa"      # gsa | local_window | dense
    local_window: int = 10

    # ── Block encoder fusion ──
    gated_fusion: bool = True

    # ── Page transformer ──
    gated_ffn: bool = True
    gated_interpage: bool = True

    # ── Compressive memory ──
    page_memory: bool = True
    section_memory: bool = True
    boundary_gate: bool = True

    def __post_init__(self) -> None:
        if self.attention_mode not in ATTENTION_MODES:
            raise ValueError(
                f"attention_mode must be one of {ATTENTION_MODES}, "
                f"got {self.attention_mode!r}"
            )
        if self.local_window < 1:
            raise ValueError("local_window must be >= 1")

    # ── (de)serialisation ──

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "AblationFlags":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def replace(self, **kwargs) -> "AblationFlags":
        return replace(self, **kwargs)

    def signature(self) -> str:
        """Short stable tag for output directories and logs."""
        parts = [
            "gsa" if self.attention_mode == "gsa" else self.attention_mode,
            f"g1{int(self.g1_output_gate)}",
            f"g2{int(self.g2_value_gate)}",
            f"adk{int(self.adaptive_sparsity)}",
            f"gf{int(self.gated_fusion)}",
            f"ffn{int(self.gated_ffn)}",
            f"ip{int(self.gated_interpage)}",
            f"pm{int(self.page_memory)}",
            f"sm{int(self.section_memory)}",
            f"bg{int(self.boundary_gate)}",
        ]
        return "-".join(parts)

    def is_default(self) -> bool:
        return self == AblationFlags()


# ── Named presets, matching the paper's ablation table ──
#  full            : delivered architecture
#  A1..A11         : see docs/experiment_plan_comparison_2026-09-21.md 4.3
PRESETS: dict[str, tuple[str, AblationFlags]] = {
    "full": ("Full HMSAN-BSA", AblationFlags()),
    "A1_no_g1": ("A1: -G1 output gate", AblationFlags(g1_output_gate=False)),
    "A2_no_g2": ("A2: -G2 value gate", AblationFlags(g2_value_gate=False)),
    "A3_fixed_k": (
        "A3: fixed sparsity (k=k_base)",
        AblationFlags(adaptive_sparsity=False),
    ),
    "A4_no_gated_fusion": (
        "A4: -gated fusion (concat+Linear)",
        AblationFlags(gated_fusion=False),
    ),
    "A5_standard_ffn": (
        "A5: standard FFN (no GLU)",
        AblationFlags(gated_ffn=False),
    ),
    "A6_no_page_memory": (
        "A6: -Infini page memory",
        AblationFlags(page_memory=False),
    ),
    "A7_no_boundary_gate": (
        "A7: -boundary gate",
        AblationFlags(boundary_gate=False),
    ),
    "A8_no_interpage_gate": (
        "A8: -inter-page gate",
        AblationFlags(gated_interpage=False),
    ),
    "A9_local_window": (
        "A9: local window attention (w=10)",
        AblationFlags(attention_mode="local_window", local_window=10),
    ),
    "A10_dense": (
        "A10: dense attention",
        AblationFlags(attention_mode="dense"),
    ),
    "A11_no_gates": (
        "A11: all gates off",
        AblationFlags(
            g1_output_gate=False,
            g2_value_gate=False,
            adaptive_sparsity=False,
            gated_fusion=False,
            gated_ffn=False,
            gated_interpage=False,
            boundary_gate=False,
        ),
    ),
    "A6A7_no_memory": (
        "A6+A7: no compressive memory at all",
        AblationFlags(page_memory=False, section_memory=False),
    ),
}


def resolve_preset(name: str) -> tuple[str, AblationFlags]:
    """Look up a preset by name (raises with the valid names on failure)."""
    if name not in PRESETS:
        raise KeyError(f"unknown ablation preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name]