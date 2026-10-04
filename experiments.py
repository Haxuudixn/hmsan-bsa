"""Experiment runner for HMSAN-BSA ablation studies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from torch.utils.data import DataLoader
import yaml

from bid_slicing.data.dataset import (
    BidDocumentDataset,
    load_documents_from_zips,
    collate_document,
)
from bid_slicing.models.hmsan_bsa import HMSAN_BSA
from bid_slicing.models.ablation import AblationFlags, PRESETS, resolve_preset
from bid_slicing.data.label_schema import LabelSchema
from train import train_epoch, validate, set_seed


# Experiment name -> ablation preset.  The first ten names are the legacy
# aliases from the original nine-experiment table; the rest have no equivalent.
_EXPERIMENT_PRESETS = {
    "full": "full",
    "no_g1": "A1_no_g1",
    "no_g2": "A2_no_g2",
    "fixed_sparse": "A3_fixed_k",
    "no_gated_fusion": "A4_no_gated_fusion",
    "no_gated_ffn": "A5_standard_ffn",
    "no_memory": "A6A7_no_memory",
    "no_boundary_gate": "A7_no_boundary_gate",
    "no_interpage_gate": "A8_no_interpage_gate",
    "local_window": "A9_local_window",
    "no_page_memory": "A6_no_page_memory",
    "dense_attention": "A10_dense",
    "no_gates": "A11_no_gates",
}

# Each entry carries the resolved AblationFlags that are actually handed to
# HMSAN_BSA.  Before this, the switches in this file were never passed to the
# model, so all nine "experiments" silently trained the identical architecture.
EXPERIMENTS = {}
for _exp_name, _preset_name in _EXPERIMENT_PRESETS.items():
    _label, _preset_flags = resolve_preset(_preset_name)
    EXPERIMENTS[_exp_name] = {
        "name": _label,
        "desc": _label,
        "preset": _preset_name,
        "flags": _preset_flags,
        "k_base": 16,
    }
del _exp_name, _preset_name, _label, _preset_flags




def run_experiment(
    exp_name: str,
    exp_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    label_schema: LabelSchema,
    device: torch.device,
    output_dir: Path,
    epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 42,
):
    """Run a single experiment."""
    set_seed(seed)
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_config['name']}")
    print(f"Description: {exp_config['desc']}")
    print(f"{'='*60}")

    flags = exp_config.get("flags") or resolve_preset(exp_config["preset"])[1]
    print(f"Ablation flags: {flags.signature()}")
    model = HMSAN_BSA(
        text_encoder_name="hfl/chinese-roberta-wwm-ext",
        text_max_length=510,
        text_freeze=True,
        image_encoder_name="google/vit-base-patch16-224",
        image_freeze=True,
        hidden_size=128,
        num_classes=label_schema.num_classes,
        num_boundaries=label_schema.num_boundaries,
        page_encoder_layers=2,
        page_encoder_heads=4,
        gsa_k_base=exp_config["k_base"],
        gsa_k_min=4,
        gsa_k_max=32,
        gsa_indexer_heads=4,
        gsa_indexer_dim=32,
        num_global_tokens=4,
        inter_page_window=3,
        dropout=0.1,
        flags=flags,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {trainable:,} trainable / {total:,} total")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)

        print(f"Epoch {epoch:2d}: "
              f"train_loss={train_metrics['loss']:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f}")

        history.append({
            "epoch": epoch,
            "train": {k: float(v) if isinstance(v, (int, float)) else v
                      for k, v in train_metrics.items()},
            "val": {k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in val_metrics.items()},
        })

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": best_val_acc,
                "config": {k: v for k, v in exp_config.items() if k != "flags"},
                "ablation_flags": flags.to_dict(),
                "ablation_signature": flags.signature(),
            }, exp_dir / "best_model.pt")

    with open(exp_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return {
        "experiment": exp_name,
        "name": exp_config["name"],
        "ablation_signature": flags.signature(),
        "best_val_acc": best_val_acc,
        "final_train_loss": history[-1]["train"]["loss"],
        "final_val_loss": history[-1]["val"]["val_loss"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=r"D:\投标文件切片\新建文件夹")
    parser.add_argument("--output_dir", default="outputs/experiments")
    parser.add_argument("--label_config", default="configs/labels.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiments", nargs="*",
                       default=None,
                       help="Specific experiments to run (default: all)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    label_schema = LabelSchema.from_yaml(args.label_config)

    # Load data
    data_dir = Path(args.data_dir)
    zip_paths = sorted(data_dir.glob("training_data_*.zip"))
    documents = load_documents_from_zips([str(z) for z in zip_paths])
    dataset = BidDocumentDataset(documents)
    train_ds, val_ds, _ = dataset.split_by_documents(0.7, 0.15, args.seed)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              collate_fn=collate_document, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=collate_document, num_workers=0)

    # Determine which experiments to run
    if args.experiments:
        exps_to_run = {k: EXPERIMENTS[k] for k in args.experiments if k in EXPERIMENTS}
    else:
        exps_to_run = EXPERIMENTS

    print(f"Running {len(exps_to_run)} experiments")

    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for exp_name, exp_config in exps_to_run.items():
        result = run_experiment(
            exp_name, exp_config,
            train_loader, val_loader,
            label_schema, device, output_dir,
            epochs=args.epochs, seed=args.seed,
        )
        results.append(result)

    # Save results table
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Experiment':<25} {'Best Val Acc':>12}")
    print("-" * 40)
    for r in sorted(results, key=lambda x: x["best_val_acc"], reverse=True):
        print(f"{r['name']:<25} {r['best_val_acc']:>12.4f}")
    print(f"{'='*60}")

    # Save readable summary
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("HMSAN-BSA Ablation Study Results\n")
        f.write("=" * 60 + "\n\n")
        for r in sorted(results, key=lambda x: x["best_val_acc"], reverse=True):
            f.write(f"{r['name']:<40} Val Acc: {r['best_val_acc']:.4f}\n")


if __name__ == "__main__":
    main()