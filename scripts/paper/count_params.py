"""Count model parameters under each freeze setting, for the paper's efficiency table.

Writes ``docs/paper_materials/params.json`` (machine readable) and prints a
Markdown table that can be pasted into the paper.  No GPU / no training needed.

Usage:
    python scripts/paper/count_params.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from bid_slicing.models.hmsan_bsa import HMSAN_BSA  # noqa: E402
from bid_slicing.data.label_schema import LabelSchema  # noqa: E402

# Modules whose parameter count is reported separately in the paper table.
MODULES = [
    "block_encoder.text_encoder",
    "block_encoder.image_encoder",
    "block_encoder.layout_encoder",
    "block_encoder.fusion_gate",
    "block_encoder.fusion_norm",
    "block_encoder.text_proj",
    "block_encoder.image_proj",
    "block_encoder.layout_proj",
    "page_encoder",
    "page_memory",
    "section_memory",
    "inter_page_attn",
    "boundary_head",
    "classification_head",
    "section_head",
]

SETTINGS = [
    ("text_frozen_image_frozen", True, True),
    ("text_trained_image_frozen", False, True),
    ("text_frozen_image_trained", True, False),
    ("both_trained", False, False),
]


def build(label_schema, text_freeze, image_freeze):
    return HMSAN_BSA(
        text_encoder_name="hfl/chinese-roberta-wwm-ext",
        text_max_length=510,
        text_freeze=text_freeze,
        image_encoder_name="google/vit-base-patch16-224",
        image_freeze=image_freeze,
        hidden_size=128,
        num_classes=label_schema.num_classes,
        num_boundaries=label_schema.num_boundaries,
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
    )


def counts(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def resolve(model, dotted):
    module = model
    for part in dotted.split("."):
        module = getattr(module, part, None)
        if module is None:
            return None
    return module


def main():
    label_schema = LabelSchema.from_yaml(str(REPO_ROOT / "configs" / "labels.yaml"))

    report = {"settings": {}, "modules_text_frozen_image_frozen": {}}
    for name, text_freeze, image_freeze in SETTINGS:
        model = build(label_schema, text_freeze, image_freeze)
        total, trainable = counts(model)
        report["settings"][name] = {
            "text_freeze": text_freeze,
            "image_freeze": image_freeze,
            "parameters_total": total,
            "parameters_trainable": trainable,
            "parameters_frozen": total - trainable,
        }
        if name == "text_frozen_image_frozen":
            for dotted in MODULES:
                module = resolve(model, dotted)
                if module is None:
                    continue
                t, r = counts(module)
                report["modules_text_frozen_image_frozen"][dotted] = {
                    "parameters_total": t,
                    "parameters_trainable": r,
                }
        del model

    out_dir = REPO_ROOT / "docs" / "paper_materials"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "params.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n| setting | text_freeze | image_freeze | total | trainable | frozen |")
    print("|---|---|---|---|---|---|")
    for name, entry in report["settings"].items():
        print(
            f"| {name} | {entry['text_freeze']} | {entry['image_freeze']} | "
            f"{entry['parameters_total']:,} | {entry['parameters_trainable']:,} | "
            f"{entry['parameters_frozen']:,} |"
        )
    print("\n| module (both frozen) | total | trainable |")
    print("|---|---|---|")
    for name, entry in report["modules_text_frozen_image_frozen"].items():
        print(
            f"| {name} | {entry['parameters_total']:,} | "
            f"{entry['parameters_trainable']:,} |"
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()