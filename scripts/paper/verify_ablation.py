"""Verify E0.3: strict checkpoint load + per-preset structure/size/forward."""
import gc, json, sys
from pathlib import Path

REPO = Path(r"C:\Users\Administrator\Documents\Codex\2026-08-20\https\outputs\hmsan-bsa")
sys.path.insert(0, str(REPO / "src"))

import torch
from bid_slicing.models.ablation import AblationFlags, PRESETS
from bid_slicing.models.hmsan_bsa import HMSAN_BSA
from bid_slicing.data.label_schema import LabelSchema

LABELS = LabelSchema.from_yaml(str(REPO / "configs" / "labels.yaml"))


def build(flags, text_freeze=True, image_freeze=True):
    return HMSAN_BSA(
        text_encoder_name="hfl/chinese-roberta-wwm-ext",
        text_max_length=510,
        text_freeze=text_freeze,
        image_encoder_name="google/vit-base-patch16-224",
        image_freeze=image_freeze,
        hidden_size=128,
        num_classes=LABELS.num_classes,
        num_boundaries=LABELS.num_boundaries,
        page_encoder_layers=2,
        page_encoder_heads=4,
        gsa_k_base=16,
        gsa_k_min=4,
        gsa_k_max=32,
        gsa_indexer_heads=4,
        gsa_indexer_dim=32,
        num_global_tokens=4,
        inter_page_window=3,
        dropout=0.1,
        flags=flags,
    )


def tiny_forward(model):
    """Text-only, 6 blocks over 2 pages - no image branch, so no ViT work."""
    N = 6
    out = model(
        texts=["\u6d4b\u8bd5\u6587\u672c"] * N,
        ocr_texts=[""] * N,
        image_blocks=[None] * N,
        block_type_ids=torch.zeros(N, dtype=torch.long),
        bbox_norm=torch.randn(N, 8),
        font_size=torch.rand(N, 1),
        is_bold=torch.zeros(N, 1),
        is_italic=torch.zeros(N, 1),
        font_color=torch.rand(N, 3),
        page_splits=[3, 6],
    )
    assert len(out["block_logits"]) == 2
    assert out["block_logits"][0].shape == (1, 3, LABELS.num_classes)
    assert out["section_logits"][0].shape == (1, LABELS.num_classes)
    assert out["boundary_logits"][0].shape == (1, 4)
    return out


print("=" * 78)
print("1) default flags must reproduce the delivered checkpoint exactly")
print("=" * 78)
model = build(AblationFlags())
total = sum(p.numel() for p in model.parameters())
print(f"AblationFlags() signature : {AblationFlags().signature()}")
print(f"is_default               : {AblationFlags().is_default()}")
print(f"total params             : {total:,}")
assert total == 191_169_859, f"default arch drifted: {total:,}"

ckpt_path = REPO / "outputs" / "hmsan_bsa_final" / "last.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=True)
print(f"strict load of last.pt   : OK (missing={len(missing)}, unexpected={len(unexpected)})")
print(f"checkpoint epoch         : {ckpt.get('epoch')}")
model.eval()
with torch.no_grad():
    tiny_forward(model)
print("default forward          : OK")
del model, ckpt
gc.collect()

print()
print("=" * 78)
print("2) every preset: parameter count + forward + module presence")
print("=" * 78)
rows = []
for name, (label, flags) in PRESETS.items():
    m = build(flags)
    t = sum(p.numel() for p in m.parameters())
    r = sum(p.numel() for p in m.parameters() if p.requires_grad)
    m.eval()
    with torch.no_grad():
        tiny_forward(m)
    present = {
        "fusion_gate": hasattr(m.block_encoder, "fusion_gate"),
        "page_memory": m.page_memory is not None,
        "section_memory": m.section_memory is not None,
        "boundary_gate": (m.section_memory is not None
                          and hasattr(m.section_memory, "boundary_gate")),
        "interpage_gate": hasattr(m.inter_page_attn, "gate"),
    }
    attn = m.page_encoder.layers[0].attn
    g1 = hasattr(attn, "g1_proj")
    g2 = hasattr(attn, "g2_proj")
    idx = hasattr(attn, "W_iq")
    ffn_gated = hasattr(m.page_encoder.layers[0].ffn, "w_value")
    rows.append((name, label, flags.signature(), t, r, g1, g2, idx, ffn_gated, present))
    print(f"{name:<20} total={t:>12,}  trainable={r:>12,}  "
          f"g1={int(g1)} g2={int(g2)} idx={int(idx)} ffn={int(ffn_gated)} "
          f"fuse={int(present['fusion_gate'])} pm={int(present['page_memory'])} "
          f"sm={int(present['section_memory'])} bg={int(present['boundary_gate'])} "
          f"ip={int(present['interpage_gate'])}")
    del m
    gc.collect()

sigs = {r[2] for r in rows}
assert len(sigs) == len(rows), "duplicate signatures -> two presets identical"
totals = [r[3] for r in rows]
# A3 only changes a runtime selection rule (no parameters), and A4's plain
# concat layer has the same shape as the gate it replaces, so ties are expected
# for exactly those two pairs.  Anything else would mean a switch is inert.
EXPECTED_TIES = {
    # A3 only switches a runtime selection rule (no parameters at all) and
    # A4's plain concat layer has exactly the shape of the gate it replaces.
    frozenset({"full", "A3_fixed_k", "A4_no_gated_fusion"}),
    # Both replace the indexer with a fixed mask, so both drop it.
    frozenset({"A9_local_window", "A10_dense"}),
}
by_total = {}
for r in rows:
    by_total.setdefault(r[3], []).append(r[0])
print()
for total, names in sorted(by_total.items()):
    if len(names) > 1:
        print(f"param-count tie at {total:,}: {', '.join(names)}")
        # Legitimate when the ablated part has no parameters (A3) or when the
        # replacement has exactly the same shape (A4's concat vs the gate,
        # A1/A2's two identically shaped gates, A9/A10 both drop the indexer).
        assert set(names) in (
            {"full", "A3_fixed_k", "A4_no_gated_fusion"},
            {"A1_no_g1", "A2_no_g2"},
            {"A9_local_window", "A10_dense"},
        ), f"unexpected tie: {names}"
print(f"\nunique signatures: {len(sigs)}/{len(rows)}   unique param counts: {len(set(totals))}/{len(rows)}")

out = REPO / "docs" / "paper_materials" / "ablation_params.json"
out.write_text(json.dumps({
    "protocol": "text_freeze=True, image_freeze=True; total params include frozen encoders",
    "rows": [
        {
            "preset": n, "label": lab, "signature": sig, "parameters_total": t,
            "parameters_trainable": r,
            "modules": {
                "g1_output_gate": g1, "g2_value_gate": g2, "lightning_indexer": idx,
                "gated_ffn": ffn_gated, "gated_fusion": p["fusion_gate"],
                "page_memory": p["page_memory"], "section_memory": p["section_memory"],
                "boundary_gate": p["boundary_gate"], "interpage_gate": p["interpage_gate"],
            },
        }
        for (n, lab, sig, t, r, g1, g2, idx, ffn_gated, p) in rows
    ],
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out}")