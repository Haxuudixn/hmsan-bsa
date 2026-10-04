"""Score the clean path-3 checkpoints on the held-out test split -- once.

Why this file exists
--------------------
Every table in ``docs/path3_thesis_draft_2026-09-30.md`` is a *validation*
reading, and the published test numbers (0.9167 / 0.8854) belong to a
warm-started checkpoint (see ``outputs/path3_clean/delivered/result.json``), so
they are dead.  The clean matrix therefore needs a test pass of its own.

``outputs/path3_diag/corpus_index.json`` cannot supply that pass: its
``dirty_val`` / ``dirty_test`` eval sets describe the *dirty* partition
(train=303 / val=65 / test=66), not the clean split (306 / 66 / 62) the matrix
was trained on, so the two disagree about which documents are held out.

What it does
------------
Rebuilds the split exactly as ``train.py`` does (same ZIPs, same caps, same
ratios, same seed) and, for each requested checkpoint, replays the training
path -- same ablation preset, same frozen-ViT feature cache, batch size 1, bf16
autocast -- over *both* the validation and the test split.

Test is only trusted when the replayed validation column reproduces the
checkpoint's own ``history.json`` last-epoch reading.  That check is what makes
the test column comparable with every number already reported.  Selection is by
validation only: nothing here trains, and nothing ranks checkpoints by their
test score.

    python scripts/paper/eval_clean_test.py --dry-run \
        --checkpoint delivered/last
    python scripts/paper/eval_clean_test.py \
        --checkpoint delivered/last --checkpoint A2_no_g2/last
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from bid_slicing.data.dataset import (  # noqa: E402
    BidDocumentDataset,
    IGNORE_INDEX,
    collate_document,
    load_documents_from_zips,
)
from bid_slicing.data.label_schema import LabelSchema  # noqa: E402
from bid_slicing.data.vit_cache import ViTFeatureCache  # noqa: E402
from bid_slicing.models.ablation import resolve_preset  # noqa: E402
from bid_slicing.models.hmsan_bsa import HMSAN_BSA  # noqa: E402

import train as train_mod  # noqa: E402

DEFAULT_DATA_DIR = r"C:\Users\Administrator\Desktop\训练数据\final_train"
DEFAULT_MATRIX_ROOT = REPO / "outputs" / "path3_clean"
DEFAULT_MANIFEST = REPO / "experiments" / "path3_clean_manifest.json"
DEFAULT_OUT = REPO / "outputs" / "path3_diag" / "test_eval_clean.json"
DEFAULT_CACHE = REPO / "cache" / "google__vit-base-patch16-224-bf16"
SPLIT_MANIFEST = REPO / "outputs" / "path3_diag" / "split_manifest.json"

TIERS = [("<100", 0, 99), ("100-999", 100, 999),
         ("1k-5k", 1000, 4999), ("5k+", 5000, 10 ** 9)]


# ---------------------------------------------------------------------------
# checkpoint discovery -- same rule as scripts/paper/leak_diag_eval.py
# ---------------------------------------------------------------------------

def _row_preset(run_dir: Path, manifest_presets: dict) -> str:
    for name in ("run_meta.json", "result.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            preset = json.loads(path.read_text(encoding="utf-8")).get("ablation_preset")
        except (OSError, ValueError):
            preset = None
        if preset:
            return preset
    return manifest_presets.get(run_dir.name, "full")


def matrix_checkpoints(matrix_root: Path, manifest_path: Path) -> list:
    presets: dict = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        presets = {run["id"]: (run.get("ablation_preset") or "full")
                   for run in manifest.get("runs", [])}
    found: list = []
    if not matrix_root.is_dir():
        return found
    for run_dir in sorted(p for p in matrix_root.iterdir() if p.is_dir()):
        preset = _row_preset(run_dir, presets)
        for suffix, filename in (("best", "best_model.pt"), ("last", "last.pt")):
            path = run_dir / filename
            if path.is_file():
                found.append({"id": f"{run_dir.name}/{suffix}",
                              "preset": preset, "path": path,
                              "run_dir": run_dir})
    return found


# ---------------------------------------------------------------------------
# inference -- verbatim from scripts/paper/leak_diag_eval.py
# ---------------------------------------------------------------------------

def build_model(label_schema, flags, cache, device):
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
        dropout=0.1,
        flags=flags,
        vit_cache=cache,
    )
    return model.to(device)


@torch.no_grad()
def predict_document(model, document, device, use_amp):
    """Per-block predictions/targets for one document, as int16 numpy arrays."""
    batch = collate_document([document])
    with train_mod._autocast(device, use_amp):
        output = model(
            texts=batch["texts"],
            ocr_texts=batch["ocr_texts"],
            image_blocks=batch["image_blocks"],
            block_type_ids=batch["block_type_ids"].to(device),
            bbox_norm=batch["bbox_norm"].to(device),
            font_size=batch["font_size"].to(device),
            is_bold=batch["is_bold"].to(device),
            is_italic=batch["is_italic"].to(device),
            font_color=batch["font_color"].to(device),
            page_splits=batch["page_splits"],
        )
    labels = batch["labels"]
    page_splits = batch["page_splits"]
    pieces_pred, pieces_target = [], []
    for index, block_logits in enumerate(output["block_logits"]):
        start = 0 if index == 0 else page_splits[index - 1]
        end = page_splits[index]
        pieces_pred.append(block_logits.argmax(dim=-1).view(-1).cpu())
        pieces_target.append(labels[start:end])
    return (
        torch.cat(pieces_pred).view(-1).numpy().astype(np.int16),
        torch.cat(pieces_target).view(-1).numpy().astype(np.int16),
    )


def _macro_min_support(per_class, threshold: int) -> float:
    pool = [item["f1"] for item in per_class if item["support"] >= threshold]
    return float(np.mean(pool)) if pool else 0.0


def metrics(pred: np.ndarray, target: np.ndarray, num_classes: int,
            labels: list) -> dict:
    """train.py's metric definition, plus the imbalance-aware variants."""
    valid = target != IGNORE_INDEX
    pred, target = pred[valid], target[valid]
    support = np.bincount(target, minlength=num_classes)
    correct = pred == target
    accuracy = float(correct.mean()) if correct.size else 0.0

    per_class = []
    f1s, weighted = [], 0.0
    for cid in range(num_classes):
        tp = int(((pred == cid) & (target == cid)).sum())
        fp = int(((pred == cid) & (target != cid)).sum())
        fn = int(((pred != cid) & (target == cid)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({
            "id": cid,
            "label": labels[cid],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support[cid]),
        })
        f1s.append(f1)
        weighted += support[cid] * f1

    total = int(support.sum())
    present = [item for item in per_class if item["support"] > 0]

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for cid in range(num_classes):
        rows = target == cid
        if rows.any():
            matrix[cid] = np.bincount(pred[rows], minlength=num_classes)

    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)),
        "macro_f1_present_classes": (float(np.mean([item["f1"] for item in present]))
                                     if present else 0.0),
        "macro_f1_support_ge_100": _macro_min_support(per_class, 100),
        "macro_f1_support_ge_500": _macro_min_support(per_class, 500),
        "weighted_f1": weighted / total if total else 0.0,
        "valid_count": total,
        "present_classes": len(present),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def tier_macro_f1(result: dict) -> dict:
    """Same four tiers as scripts/paper/support_stratified.py."""
    per_class = result["per_class"]
    tiers, classes, blocks = {}, {}, {}
    for name, lo, hi in TIERS:
        idx = [i for i, item in enumerate(per_class) if lo <= item["support"] <= hi]
        tiers[name] = float(np.mean([per_class[i]["f1"] for i in idx])) if idx else None
        classes[name] = len(idx)
        blocks[name] = int(sum(per_class[i]["support"] for i in idx))
    return {"macro_f1": tiers, "classes": classes, "blocks": blocks}


def history_last_epoch(run_dir: Path):
    path = run_dir / "history.json"
    if not path.is_file():
        return None
    records = json.loads(path.read_text(encoding="utf-8"))
    epoch_records = [r for r in records if r.get("phase") != "mid_epoch"]
    if not epoch_records:
        return None
    last = max(epoch_records, key=lambda r: int(r["epoch"]))
    val = last.get("val", {})
    return {
        "epoch": int(last["epoch"]),
        "accuracy": float(val["accuracy"]),
        "macro_f1": float(val["macro_f1"]),
        "valid_count": int(val.get("valid_count", 0)),
    }


def verify_split(splits) -> bool:
    if not SPLIT_MANIFEST.is_file():
        print(f"  (no {SPLIT_MANIFEST.name}; split not cross-checked)")
        return True
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    ok = True
    for name, dataset in zip(("train", "val", "test"), splits):
        expected = manifest["splits"][name]
        got_pdfs = {d.pdf_name for d in dataset.documents}
        same = (len(dataset.documents) == expected["segments"]
                and got_pdfs == set(expected["pdfs"]))
        print(f"  split {name:<5} segments={len(dataset.documents):<4}"
              f" (manifest {expected['segments']:<4}) "
              f"pdfs_match={got_pdfs == set(expected['pdfs'])}  "
              f"{'OK' if same else 'MISMATCH'}")
        ok = ok and same
    if not ok:
        print("  !! the rebuilt split disagrees with split_manifest.json, so the "
              "checkpoints were trained on a different partition; refusing to "
              "report test numbers.")
    return ok


def write_report(report: dict, out_path: Path) -> None:
    """Persist after every checkpoint, so a kill is not a total loss."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                        encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--matrix_root", default=str(DEFAULT_MATRIX_ROOT))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--checkpoint", action="append", default=[],
                   help="checkpoint id such as delivered/last (repeatable)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--label_config", default="configs/labels.yaml")
    p.add_argument("--vit_cache_dir", default=str(DEFAULT_CACHE))
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--max_blocks_per_sample", type=int, default=3000)
    p.add_argument("--max_pages_per_sample", type=int, default=150)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tolerance", type=float, default=1e-4,
                   help="max |diff| allowed between the replayed val metrics and "
                        "the checkpoint's own history.json last epoch (bf16 "
                        "autocast is not bit-reproducible, so this is not 0)")
    p.add_argument("--limit_docs", type=int, default=0,
                   help="debug: score only the first N documents of each split")
    p.add_argument("--dry-run", action="store_true",
                   help="rebuild the split and print the plan; no inference")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    zip_paths = sorted(str(z) for z in data_dir.glob("training_data_*.zip"))
    if not zip_paths:
        raise SystemExit(f"no training_data_*.zip under {data_dir}")
    print(f"loading {len(zip_paths)} ZIP(s) from {data_dir}")
    documents = load_documents_from_zips(
        zip_paths, args.max_blocks_per_sample, args.max_pages_per_sample)
    total_blocks = sum(len(p.blocks) for d in documents for p in d.pages)
    print(f"corpus: {len(documents)} segments, {total_blocks} blocks, "
          f"{len({d.pdf_name for d in documents})} source PDFs")

    splits = BidDocumentDataset(documents).split_by_documents(
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    print(f"split: train={len(splits[0])} val={len(splits[1])} "
          f"test={len(splits[2])}")
    if not verify_split(splits):
        return 2
    del documents
    gc.collect()

    catalogue = matrix_checkpoints(Path(args.matrix_root), Path(args.manifest))
    by_id = {item["id"]: item for item in catalogue}
    if not args.checkpoint:
        raise SystemExit("pass at least one --checkpoint (e.g. delivered/last); "
                         f"available: {sorted(by_id)}")
    missing = [cid for cid in args.checkpoint if cid not in by_id]
    if missing:
        raise SystemExit(f"unknown checkpoint id(s) {missing}; "
                         f"available: {sorted(by_id)}")
    checkpoints = [by_id[cid] for cid in args.checkpoint]
    print(f"checkpoints: {', '.join(args.checkpoint)}")

    if args.dry_run:
        for item in checkpoints:
            print(f"  {item['id']:<32} preset={item['preset']:<18} "
                  f"history_last_epoch={history_last_epoch(item['run_dir'])}")
        print("dry-run: nothing scored")
        return 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"device={device} amp={use_amp}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    label_schema = LabelSchema.from_yaml(args.label_config)
    cache = ViTFeatureCache.open(args.vit_cache_dir)
    print(f"vit cache: {cache.describe()}")

    report = {
        "schema": 1,
        "generated_by": "scripts/paper/eval_clean_test.py",
        "data_dir": str(data_dir),
        "matrix_root": str(args.matrix_root),
        "test_policy": ("the test split is scored once, after all training has "
                        "stopped, over frozen checkpoints; every checkpoint "
                        "reported here is selected by validation only"),
        "val_reference": ("history.json last-epoch record (phase != 'mid_epoch'); "
                          "the replayed val column must reproduce it"),
        "split": {"train": len(splits[0]), "val": len(splits[1]),
                  "test": len(splits[2]), "train_ratio": args.train_ratio,
                  "val_ratio": args.val_ratio, "seed": args.seed},
        "protocol": {"max_blocks_per_sample": args.max_blocks_per_sample,
                     "max_pages_per_sample": args.max_pages_per_sample,
                     "amp": use_amp, "device": str(device),
                     "vit_cache": cache.describe()},
        "models": [],
    }
    if out_path.is_file():
        try:
            previous = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        fresh = set(args.checkpoint)
        kept = [e for e in previous.get("models", []) if e.get("id") not in fresh]
        report["models"] = kept
        if kept:
            print(f"kept {len(kept)} model(s) from {out_path.name}")

    val_docs = list(splits[1].documents)
    test_docs = list(splits[2].documents)
    if args.limit_docs:
        val_docs = val_docs[: args.limit_docs]
        test_docs = test_docs[: args.limit_docs]

    all_ok = True
    for item in checkpoints:
        path = Path(item["path"])
        if not path.is_file():
            print(f"skip {item['id']}: missing {path}")
            all_ok = False
            continue

        try:
            checkpoint = torch.load(path, map_location="cpu", mmap=True,
                                    weights_only=False)
        except (RuntimeError, TypeError, ValueError):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        _, flags = resolve_preset(item["preset"])
        signature = checkpoint.get("ablation_signature")
        if signature and signature != flags.signature():
            raise SystemExit(f"{item['id']}: checkpoint signature {signature} != "
                             f"preset {item['preset']} signature "
                             f"{flags.signature()}")
        model = build_model(label_schema, flags, cache, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.block_encoder.vit_cache_misses = 0

        entry = {
            "id": item["id"],
            "checkpoint": str(path),
            "preset": item["preset"],
            "ablation_signature": flags.signature(),
            "source_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_val_accuracy": float(
                checkpoint.get("val_accuracy", float("nan"))),
            "parameters_total": int(sum(p.numel() for p in model.parameters())),
            "sets": {},
        }
        del checkpoint
        gc.collect()

        print()
        print(f"=== {item['id']}  ({path.name}, epoch {entry['source_epoch']}) ===")

        for set_name, docs in (("val", val_docs), ("test", test_docs)):
            started = time.time()
            pieces_pred, pieces_target, skipped = [], [], []
            for position, document in enumerate(docs, 1):
                try:
                    pred, target = predict_document(model, document, device, use_amp)
                except torch.cuda.OutOfMemoryError:
                    gc.collect()
                    torch.cuda.empty_cache()
                    try:
                        pred, target = predict_document(model, document, device, use_amp)
                    except torch.cuda.OutOfMemoryError:
                        skipped.append(document.pdf_name)
                        print(f"  [oom] {set_name}/{document.pdf_name} skipped")
                        continue
                pieces_pred.append(pred)
                pieces_target.append(target)
                if position % 20 == 0 or position == len(docs):
                    elapsed = time.time() - started
                    print(f"  {set_name} {position}/{len(docs)} docs {elapsed:.0f}s "
                          f"({elapsed / position:.1f}s/doc)", flush=True)
            if not pieces_pred:
                continue
            result = metrics(np.concatenate(pieces_pred),
                             np.concatenate(pieces_target),
                             label_schema.num_classes,
                             list(label_schema.class_labels))
            result["segments_scored"] = len(pieces_pred)
            result["skipped_segments"] = skipped
            result["inference_seconds"] = round(time.time() - started, 1)
            result["tiers"] = tier_macro_f1(result)
            entry["sets"][set_name] = result
            print(f"  {set_name:<5} acc={result['accuracy']:.4f} "
                  f"macroF1={result['macro_f1']:.4f} "
                  f"({result['valid_count']} blocks)")

        entry["vit_cache_misses"] = int(model.block_encoder.vit_cache_misses)

        reference = history_last_epoch(item["run_dir"])
        if reference and "val" in entry["sets"]:
            got = entry["sets"]["val"]
            deltas = {
                "accuracy": abs(got["accuracy"] - reference["accuracy"]),
                "macro_f1": abs(got["macro_f1"] - reference["macro_f1"]),
                "valid_count": abs(got["valid_count"] - reference["valid_count"]),
            }
            ok = (deltas["accuracy"] <= args.tolerance
                  and deltas["macro_f1"] <= args.tolerance
                  and deltas["valid_count"] == 0)
            entry["val_fidelity"] = {"reference": reference,
                                     "abs_delta": deltas, "ok": bool(ok),
                                     "tolerance": args.tolerance}
            all_ok = all_ok and ok
            print(f"  fidelity vs history.json ep{reference['epoch']}: "
                  f"d_acc={deltas['accuracy']:.2e} "
                  f"d_macroF1={deltas['macro_f1']:.2e} "
                  f"d_blocks={deltas['valid_count']}  {'OK' if ok else 'FAIL'}")
        else:
            entry["val_fidelity"] = {"reference": None, "ok": False,
                                     "tolerance": args.tolerance}
            all_ok = False
            print("  fidelity: no history.json last-epoch record -- the test "
                  "number would be unverified")

        report["models"].append(entry)
        write_report(report, out_path)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_report(report, out_path)
    print()
    print(f"wrote {out_path}")

    print()
    header = (f"{'model':<30}{'ep':>4}{'val acc':>10}{'val F1':>9}"
              f"{'test acc':>10}{'test F1':>9}{'fidel':>8}")
    print(header)
    print("-" * len(header))
    for entry in report["models"]:
        val = entry["sets"].get("val", {})
        test = entry["sets"].get("test", {})
        print(f"{entry['id']:<30}{entry['source_epoch']:>4}"
              f"{val.get('accuracy', float('nan')):>10.4f}"
              f"{val.get('macro_f1', float('nan')):>9.4f}"
              f"{test.get('accuracy', float('nan')):>10.4f}"
              f"{test.get('macro_f1', float('nan')):>9.4f}"
              f"{'OK' if entry.get('val_fidelity', {}).get('ok') else 'FAIL':>8}")

    print()
    print("test support tiers (macro F1):")
    for entry in report["models"]:
        test = entry["sets"].get("test")
        if not test:
            continue
        tiers = test["tiers"]["macro_f1"]
        cells = "  ".join(
            f"{name}={tiers[name]:.4f}" if tiers[name] is not None else f"{name}=n/a"
            for name, _, _ in TIERS)
        print(f"  {entry['id']:<30} overall={test['macro_f1']:.4f}  {cells}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())