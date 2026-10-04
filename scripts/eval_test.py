"""Evaluate trained checkpoints on the held-out test split.

The test split is rebuilt exactly as in ``train.py`` (same data dir, same
page-range caps, same seed/ratios), so the 66 test segments are the ones the
model never saw.

Reports accuracy, macro-F1, per-class precision/recall/F1/support and the
confusion matrix for every requested checkpoint, plus the model's overall
parameter count.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from bid_slicing.data.dataset import (  # noqa: E402
    BidDocumentDataset,
    IGNORE_INDEX,
    load_documents_from_zips,
    collate_document,
)
from bid_slicing.data.label_schema import LabelSchema  # noqa: E402
from bid_slicing.models.hmsan_bsa import HMSAN_BSA  # noqa: E402

import train as train_mod  # noqa: E402


def build_model(label_schema, dropout, device):
    model = HMSAN_BSA(
        text_encoder_name="hfl/chinese-roberta-wwm-ext",
        text_max_length=510,
        text_freeze=False,
        image_encoder_name="google/vit-base-patch16-224",
        image_freeze=True,
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
        dropout=dropout,
    )
    return model.to(device)


@torch.no_grad()
def predict(model, dataloader, device, label_schema, use_amp):
    """Return (predictions, targets, per-document stats) for the loader."""
    model.eval()
    preds, targets = [], []
    documents = []
    for batch in tqdm(dataloader, desc="Test", leave=False):
        block_type_ids = batch["block_type_ids"].to(device)
        page_splits = batch["page_splits"]
        with train_mod._autocast(device, use_amp):
            output = model(
                texts=batch["texts"],
                ocr_texts=batch["ocr_texts"],
                image_blocks=batch["image_blocks"],
                block_type_ids=block_type_ids,
                bbox_norm=batch["bbox_norm"].to(device),
                font_size=batch["font_size"].to(device),
                is_bold=batch["is_bold"].to(device),
                is_italic=batch["is_italic"].to(device),
                font_color=batch["font_color"].to(device),
                page_splits=page_splits,
            )
        labels = batch["labels"]
        doc_preds, doc_targets = [], []
        for i, block_logits in enumerate(output["block_logits"]):
            page_start = 0 if i == 0 else page_splits[i - 1]
            page_end = page_splits[i]
            doc_preds.append(block_logits.argmax(dim=-1).view(-1).cpu())
            doc_targets.append(labels[page_start:page_end])
        doc_pred = torch.cat(doc_preds).view(-1)
        doc_target = torch.cat(doc_targets).view(-1)
        mask = doc_target != IGNORE_INDEX
        documents.append({
            "pdf_name": batch.get("pdf_name", [""])[0] if isinstance(batch.get("pdf_name"), list) else str(batch.get("pdf_name", "")),
            "blocks": int(doc_target.numel()),
            "labeled": int(mask.sum().item()),
            "correct": int((doc_pred[mask] == doc_target[mask]).sum().item()),
        })
        preds.append(doc_pred)
        targets.append(doc_target)
    return torch.cat(preds), torch.cat(targets), documents


def confusion_matrix(preds, targets, num_classes):
    mask = targets != IGNORE_INDEX
    valid_preds = preds[mask].tolist()
    valid_targets = targets[mask].tolist()
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for pred, target in zip(valid_preds, valid_targets):
        matrix[target][pred] += 1
    return matrix


def main():
    parser = argparse.ArgumentParser(description="evaluate checkpoints on the test split")
    parser.add_argument(
        "--data_dir",
        default=r"C:\Users\Administrator\Desktop\训练数据\final_train",
    )
    parser.add_argument("--output_dir", default="outputs/hmsan_bsa_final")
    parser.add_argument("--label_config", default="configs/labels.yaml")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["best_model.pt", "last.pt", "checkpoint_epoch5.pt"],
    )
    parser.add_argument("--max_blocks_per_sample", type=int, default=3000)
    parser.add_argument("--max_pages_per_sample", type=int, default=150)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--out_json", default="docs/paper_materials/eval_test.json")
    parser.add_argument(
        "--selection",
        choices=["last", "val_acc", "val_macro_f1"],
        default="last",
        help="Validation-only criterion that picks the checkpoint. `last` is the "
             "final epoch and matches the frozen reporting protocol; the two val "
             "options are the appendix robustness view. The test split is never "
             "used to choose.",
    )
    parser.add_argument(
        "--also_test_all",
        action="store_true",
        help="Appendix only: also score EVERY checkpoint on test. That is the "
             "selection-time test use the paper claims never happens, so do not "
             "report these numbers as the headline result.",
    )
    args = parser.parse_args()

    repo = REPO_ROOT
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    out_json = Path(args.out_json)
    if not out_json.is_absolute():
        out_json = repo / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"device={device} amp={use_amp}")

    label_schema = LabelSchema.from_yaml(str(repo / args.label_config))
    zip_paths = sorted(Path(args.data_dir).glob("training_data_*.zip"))
    print(f"packages: {len(zip_paths)}")
    documents = load_documents_from_zips(
        [str(z) for z in zip_paths],
        args.max_blocks_per_sample,
        args.max_pages_per_sample,
    )
    dataset = BidDocumentDataset(documents)
    train_ds, val_ds, test_ds = dataset.split_by_documents(
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f"split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=collate_document, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=collate_document, num_workers=0,
    )

    report = {
        "data_dir": str(args.data_dir),
        "split": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "train_segments": len(train_ds),
            "val_segments": len(val_ds),
            "test_segments": len(test_ds),
        },
        "class_labels": [label_schema.decode_class(i) for i in range(label_schema.num_classes)],
        "protocol": (
            "Checkpoint selection uses the validation split only. The test split is "
            "evaluated exactly once, for the single selected checkpoint. Choosing a "
            "checkpoint by its test score is the practice the paper states never "
            "happens; --also_test_all exists only for an appendix sensitivity view "
            "and its numbers must not be reported as the headline result."
        ),
        "val_models": [],
        "also_test_all": bool(args.also_test_all),
        "models": [],
    }

    def per_class_of(metrics):
        return [
            {
                "id": c,
                "label": label_schema.decode_class(c),
                "precision": metrics.get(f"precision_{c}", 0.0),
                "recall": metrics.get(f"recall_{c}", 0.0),
                "f1": metrics.get(f"f1_{c}", 0.0),
                "support": metrics.get(f"support_{c}", 0),
            }
            for c in range(label_schema.num_classes)
        ]

    def build_entry(name, checkpoint, metrics, n_params, n_trainable):
        return {
            "checkpoint": name,
            "source_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_val_accuracy": float(checkpoint.get("val_accuracy", float("nan"))),
            "parameters_total": n_params,
            "parameters_trainable": n_trainable,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics.get("macro_f1", 0.0),
            "valid_blocks": metrics["valid_count"],
            "total_blocks": metrics["total_count"],
            "per_class": per_class_of(metrics),
        }

    def load_checkpoint_model(name):
        checkpoint = torch.load(output_dir / name, map_location=device, weights_only=False)
        model = build_model(label_schema, args.dropout, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return checkpoint, model, n_params, n_trainable

    # ---- pass 1: validation only; this is the only input to selection ----
    for name in args.checkpoints:
        path = output_dir / name
        if not path.exists():
            print(f"skip {name}: not found at {path}")
            continue
        checkpoint, model, n_params, n_trainable = load_checkpoint_model(name)

        val_preds, val_targets, _ = predict(
            model, val_loader, device, label_schema, use_amp
        )
        val_metrics = train_mod.compute_metrics(
            val_preds, val_targets, num_classes=label_schema.num_classes
        )
        report["val_models"].append(
            build_entry(name, checkpoint, val_metrics, n_params, n_trainable)
        )
        print(f"[val] {name}: epoch={int(checkpoint.get('epoch', -1))} "
              f"acc={val_metrics['accuracy']:.4f} "
              f"macro_f1={val_metrics.get('macro_f1', 0.0):.4f}")

        if args.also_test_all:
            t_preds, t_targets, t_docs = predict(
                model, test_loader, device, label_schema, use_amp
            )
            t_metrics = train_mod.compute_metrics(
                t_preds, t_targets, num_classes=label_schema.num_classes
            )
            t_entry = build_entry(name, checkpoint, t_metrics, n_params, n_trainable)
            t_entry["confusion_matrix"] = confusion_matrix(
                t_preds, t_targets, label_schema.num_classes
            )
            t_entry["documents"] = t_docs
            report["models"].append(t_entry)
            del t_preds, t_targets

        del val_preds, val_targets, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not report["val_models"]:
        raise SystemExit("no checkpoints were evaluated")

    # ---- selection: validation only ----
    if args.selection == "last":
        chosen = max(report["val_models"], key=lambda item: item["source_epoch"])
    elif args.selection == "val_acc":
        chosen = max(report["val_models"], key=lambda item: item["accuracy"])
    else:
        chosen = max(report["val_models"], key=lambda item: item["macro_f1"])

    report["selection"] = {
        "criterion": args.selection,
        "selected_checkpoint": chosen["checkpoint"],
        "val_accuracy": chosen["accuracy"],
        "val_macro_f1": chosen["macro_f1"],
        "test_evaluated_once": True,
    }
    print(f"\nselected by {args.selection}: {chosen['checkpoint']} "
          f"(val acc={chosen['accuracy']:.4f}, "
          f"val macro_f1={chosen['macro_f1']:.4f})")

    # ---- pass 2: the test split, once, for the selected checkpoint ----
    checkpoint, model, n_params, n_trainable = load_checkpoint_model(
        chosen["checkpoint"]
    )
    preds, targets, documents_stats = predict(
        model, test_loader, device, label_schema, use_amp
    )
    metrics = train_mod.compute_metrics(
        preds, targets, num_classes=label_schema.num_classes
    )
    test_entry = build_entry(
        chosen["checkpoint"], checkpoint, metrics, n_params, n_trainable
    )
    test_entry["confusion_matrix"] = confusion_matrix(
        preds, targets, label_schema.num_classes
    )
    test_entry["documents"] = documents_stats
    report["selected_test"] = test_entry
    print(f"\n[test] {chosen['checkpoint']}: acc={metrics['accuracy']:.4f} "
          f"macro_f1={metrics.get('macro_f1', 0.0):.4f} "
          f"({metrics['valid_count']}/{metrics['total_count']} valid blocks)")

    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\nwrote {out_json}")

    # Convenience CSV next to the JSON: the SELECTED checkpoint, on test.
    with open(out_json.with_name("per_class_metrics.csv"), "w",
              encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "label", "precision", "recall", "f1", "support"])
        for item in test_entry["per_class"]:
            writer.writerow([item["id"], item["label"],
                             f"{item['precision']:.4f}", f"{item['recall']:.4f}",
                             f"{item['f1']:.4f}", item["support"]])
    print(f"wrote {out_json.with_name('per_class_metrics.csv')}")


if __name__ == "__main__":
    main()
