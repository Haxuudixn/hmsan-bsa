"""Step 0 leak audit, phase 2: re-score the delivered checkpoints on a clean val.

Phase 1 (``leak_diag_scan.py``) rebuilds the corpus partition and marks the
documents that sit in exports which predate the ft3 warm-start run.  This phase
loads the very checkpoints the matrix produced, replays them over

  * ``dirty_val``      - the delivered val split (65 segments), the one the
                         checkpoints were selected on;
  * ``val_no_ft3``     - val minus documents that occur in a pre-ft3 export;
  * ``val_no_straddle``- val minus documents that the split put on two sides;
  * ``val_clean``      - val with both leaks removed;
  * ``test_clean``     - the same condition applied to the test split;
  * ``clean_val_test`` - val_clean + test_clean.

Nothing is trained and no protocol file is touched.  Inference reproduces the
training path exactly: same model construction (same ablation preset, same
frozen-ViT feature cache, bf16 autocast), same collate, batch size 1.  The
dirty-val numbers double as a fidelity check - they must match history.json,
otherwise the clean numbers would not be comparable.

    python scripts/paper/leak_diag_eval.py
    python scripts/paper/leak_diag_eval.py --checkpoint full_3ep
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from bid_slicing.data.dataset import (  # noqa: E402
    IGNORE_INDEX,
    collate_document,
    load_documents_from_zips,
)
from bid_slicing.data.label_schema import LabelSchema  # noqa: E402
from bid_slicing.data.vit_cache import ViTFeatureCache  # noqa: E402
from bid_slicing.models.ablation import resolve_preset  # noqa: E402
from bid_slicing.models.hmsan_bsa import HMSAN_BSA  # noqa: E402

import train as train_mod  # noqa: E402

DEFAULT_INDEX = REPO / "outputs" / "path3_diag" / "corpus_index.json"
DEFAULT_OUT = REPO / "outputs" / "path3_diag" / "leak_diag_eval.json"
DEFAULT_CACHE = REPO / "cache" / "google__vit-base-patch16-224-bf16"

CHECKPOINTS = [
    {
        "id": "full_3ep",
        "preset": "full",
        "path": REPO / "outputs" / "path3_matrix" / "full_3ep" / "best_model.pt",
    },
    {
        "id": "A6_no_page_memory",
        "preset": "A6_no_page_memory",
        "path": REPO / "outputs" / "path3_matrix" / "A6_no_page_memory" / "best_model.pt",
    },
    {
        # the checkpoint the paper reports: last.pt of the delivered 8-epoch run
        # (test acc 0.9167 / macro F1 0.8854 in docs/paper_materials/results.md)
        "id": "delivered_last8",
        "preset": "full",
        "path": REPO / "outputs" / "hmsan_bsa_final" / "best_model_by_macro_f1.pt",
    },
]


MATRIX_ROOT = REPO / "outputs" / "path3_matrix"
MANIFEST = REPO / "experiments" / "path3_manifest.json"


def _row_preset(run_dir: Path, manifest_presets: dict[str, str]) -> str:
    """The preset a row was actually trained with.

    ``run_meta.json`` is written by train.py itself and carries the preset and
    the ablation signature, so it beats a hand-kept central manifest -- and it
    is what makes this script work on a matrix the manifest does not know
    about.  The manifest stays as the fallback for older runs.
    """
    meta = run_dir / "run_meta.json"
    if meta.is_file():
        try:
            preset = json.loads(meta.read_text(encoding="utf-8")).get(
                "ablation_preset")
        except (OSError, ValueError):
            preset = None
        if preset:
            return preset
    return manifest_presets.get(run_dir.name, "full")


def matrix_checkpoints(matrix_root: Path = MATRIX_ROOT,
                       manifest_path: Path = MANIFEST) -> list[dict]:
    """Every matrix row contributes both of its surviving checkpoints.

    ``best_model.pt`` is the epoch the current protocol reports (highest val
    accuracy on the dirty val split); ``last.pt`` is the final epoch.  Scoring
    both is what shows whether the choice of selection rule, rather than the
    architecture, moves a verdict.
    """
    presets: dict[str, str] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        presets = {run["id"]: run.get("ablation_preset", "full")
                   for run in manifest.get("runs", [])}
    discovered: list[dict] = []
    if not matrix_root.is_dir():
        return discovered
    for run_dir in sorted(p for p in matrix_root.iterdir() if p.is_dir()):
        preset = _row_preset(run_dir, presets)
        for suffix, filename in (("best", "best_model.pt"), ("last", "last.pt")):
            path = run_dir / filename
            if path.is_file():
                discovered.append({"id": f"{run_dir.name}/{suffix}",
                                   "preset": preset, "path": path})
    return discovered


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
    macro_all = float(np.mean(f1s))
    macro_present = float(np.mean([item["f1"] for item in present])) if present else 0.0

    def macro_min_support(threshold: int) -> float:
        pool = [item["f1"] for item in per_class if item["support"] >= threshold]
        return float(np.mean(pool)) if pool else 0.0

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for cid in range(num_classes):
        rows = target == cid
        if rows.any():
            matrix[cid] = np.bincount(pred[rows], minlength=num_classes)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_all,
        "macro_f1_present_classes": macro_present,
        "macro_f1_support_ge_100": macro_min_support(100),
        "macro_f1_support_ge_500": macro_min_support(500),
        "weighted_f1": weighted / total if total else 0.0,
        "valid_count": total,
        "present_classes": len(present),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def load_segments(pool_specs: dict, data_dir: Path, max_blocks: int,
                  max_pages: int) -> dict:
    """Materialise the requested segments, reading each source ZIP once."""
    pool = {}
    by_zip: "OrderedDict[str, list]" = OrderedDict()
    for key, spec in pool_specs.items():
        by_zip.setdefault(spec["zip"], []).append(spec)

    for zip_name, specs in by_zip.items():
        zip_path = data_dir / zip_name
        if not zip_path.is_file():
            raise SystemExit(f"missing export {zip_path}")
        documents = load_documents_from_zips([str(zip_path)], max_blocks, max_pages)
        groups: "OrderedDict[str, list]" = OrderedDict()
        for document in documents:
            groups.setdefault(document.pdf_name, []).append(document)
        for spec in specs:
            siblings = groups.get(spec["pdf_name"])
            if not siblings:
                raise SystemExit(f"{zip_name}: no document named {spec['pdf_name']!r}")
            document = siblings[spec["seg_index"]]
            actual = sum(len(page.blocks) for page in document.pages)
            if actual != spec["blocks"]:
                raise SystemExit(
                    f"{zip_name}:{spec['pdf_name']} segment {spec['seg_index']} has "
                    f"{actual} blocks, index says {spec['blocks']}"
                )
            pool[(spec["zip"], spec["pdf_name"], spec["seg_index"])] = document
        del documents, groups
        gc.collect()
    return pool


def write_report(report: dict, out_path: Path) -> None:
    """Persist the report after every checkpoint, so a kill is not a total loss."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--vit_cache_dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--label_config", default="configs/labels.yaml")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="limit to these checkpoint ids (repeatable)")
    parser.add_argument("--matrix_root", default=str(MATRIX_ROOT),
                        help="directory holding one sub-directory per matrix "
                             "row (default: outputs/path3_matrix)")
    parser.add_argument("--manifest", default=str(MANIFEST),
                        help="fallback preset map for rows without run_meta")
    parser.add_argument("--no_builtin", action="store_true",
                        help="ignore the hard-coded legacy CHECKPOINTS list "
                             "and score only what --matrix_root discovers")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--limit_segments", type=int, default=0,
                        help="debug: score only the first N segments per set")
    parser.add_argument("--sets", default="",
                        help="comma-separated subset of eval sets to score")
    args = parser.parse_args()

    out_path = Path(args.out)
    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    protocol = index["protocol"]
    data_dir = Path(index["data_dir"])
    labels = index["class_labels"]
    num_classes = len(labels)

    sets = index["eval_sets"]
    wanted = dict(sets)
    if args.sets:
        keep = [name.strip() for name in args.sets.split(",") if name.strip()]
        unknown = [name for name in keep if name not in wanted]
        if unknown:
            raise SystemExit(f"unknown eval set(s) {unknown}; have {sorted(wanted)}")
        wanted = {name: wanted[name] for name in keep}

    # One inference per unique segment, reused by every set and checkpoint.
    pool_specs = {}
    for name, specs in wanted.items():
        for spec in specs:
            pool_specs[(spec["zip"], spec["pdf_name"], spec["seg_index"])] = spec
    print(f"unique segments to score: {len(pool_specs)} "
          f"(sets: " + ", ".join(f"{k}={len(v)}" for k, v in wanted.items()) + ")")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"device={device} amp={use_amp}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    label_schema = LabelSchema.from_yaml(args.label_config)
    if list(label_schema.class_labels) != labels:
        raise SystemExit("labels.yaml disagrees with the corpus index")

    cache = ViTFeatureCache.open(args.vit_cache_dir)
    print(f"vit cache: {cache.describe()}")

    t_load = time.time()
    pool = load_segments(
        dict(sorted(pool_specs.items())),
        data_dir,
        protocol["max_blocks_per_sample"],
        protocol["max_pages_per_sample"],
    )
    print(f"loaded {len(pool)} documents in {time.time() - t_load:.1f}s")

    catalogue = ([] if args.no_builtin else CHECKPOINTS) + matrix_checkpoints(
        Path(args.matrix_root), Path(args.manifest))
    checkpoints: list[dict] = []
    seen_paths: set[str] = set()
    for item in catalogue:
        if args.checkpoint and item["id"] not in args.checkpoint:
            continue
        key = str(Path(item["path"]).resolve()).lower()
        if key in seen_paths:
            continue
        seen_paths.add(key)
        checkpoints.append(item)
    print(f"checkpoints selected: {len(checkpoints)} "
          f"({'all' if not args.checkpoint else ', '.join(args.checkpoint)})")
    if not checkpoints:
        raise SystemExit("no checkpoint selected")

    # dirty_val is the delivered split; the rest peel off one leak at a time.
    preferred = [
        "dirty_val", "val_no_ft3", "val_no_straddle", "val_clean",
        "test_clean", "clean_val_test",
    ]
    set_order = [name for name in preferred if name in wanted]
    set_order += [name for name in sorted(wanted) if name not in set_order]
    report = {
        "schema": 1,
        "generated_by": "scripts/paper/leak_diag_eval.py",
        "corpus_index": str(args.index),
        "vit_cache": cache.to_dict(),
        "amp": use_amp,
        "sets": {},
        "models": [],
    }
    # Carry earlier models over, so an incremental pass (one row at a time as
    # the matrix completes) does not clobber the previous results.
    if out_path.is_file():
        try:
            previous = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        fresh_ids = {item["id"] for item in checkpoints}
        report["models"] = [entry for entry in previous.get("models", [])
                            if entry.get("id") not in fresh_ids]
        if report["models"]:
            print(f"kept {len(report['models'])} model(s) from {out_path.name}")

    for name in set_order:
        specs = wanted.get(name)
        if not specs:
            continue
        keys = [(s["zip"], s["pdf_name"], s["seg_index"]) for s in specs]
        targets = np.concatenate([
            np.array([b.label_id for page in pool[k].pages for b in page.blocks],
                     dtype=np.int16)
            for k in keys
        ])
        support = np.bincount(targets[targets != IGNORE_INDEX], minlength=num_classes)
        report["sets"][name] = {
            "segments": len(specs),
            "blocks": int(targets.size),
            "labeled": int((targets != IGNORE_INDEX).sum()),
            "class_support": [int(x) for x in support],
        }
        print(f"set {name:<15} segments={len(specs):>3} labeled={int((targets != IGNORE_INDEX).sum())}")

    for item in checkpoints:
        path = Path(item["path"])
        if not path.is_file():
            print(f"skip {item['id']}: missing {path}")
            continue

        try:
            # mmap keeps the 1.6 GB checkpoint on disk instead of in RAM, which
            # matters on a 7.95 GB box while a training run holds the rest.
            checkpoint = torch.load(
                path, map_location="cpu", mmap=True, weights_only=False
            )
        except (RuntimeError, TypeError, ValueError):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        _, flags = resolve_preset(item["preset"])
        signature = checkpoint.get("ablation_signature")
        if signature and signature != flags.signature():
            raise SystemExit(
                f"{item['id']}: checkpoint signature {signature} != preset "
                f"{item['preset']} signature {flags.signature()}"
            )
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
            "checkpoint_val_accuracy": float(checkpoint.get("val_accuracy", float("nan"))),
            "parameters_total": int(sum(p.numel() for p in model.parameters())),
            "sets": {},
        }
        # The checkpoint also carries optimizer state (~1.6 GB on this box);
        # release it before the pass instead of holding it for 20 minutes.
        del checkpoint
        gc.collect()

        print()
        print(f"=== {item['id']}  ({path.name}, epoch {entry['source_epoch']}, "
              f"training val_acc={entry['checkpoint_val_accuracy']:.4f}) ===")

        predictions = {}
        skipped = []
        if args.limit_segments:
            keys = sorted(pool)[: args.limit_segments]
        else:
            keys = sorted(pool)
        started = time.time()
        for position, key in enumerate(keys, 1):
            document = pool[key]
            try:
                predictions[key] = predict_document(model, document, device, use_amp)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    predictions[key] = predict_document(model, document, device, use_amp)
                except torch.cuda.OutOfMemoryError:
                    skipped.append(key)
                    print(f"  [oom] {key} - skipped")
            if position % 10 == 0 or position == len(keys):
                elapsed = time.time() - started
                print(f"  {position}/{len(keys)} documents  {elapsed:.0f}s "
                      f"({elapsed / position:.1f}s/doc)", flush=True)

        entry["inference_seconds"] = round(time.time() - started, 1)
        entry["vit_cache_misses"] = int(model.block_encoder.vit_cache_misses)
        entry["skipped_segments"] = [list(k) for k in skipped]
        if entry["vit_cache_misses"]:
            print(f"  WARNING: {entry['vit_cache_misses']} image blocks missed the cache")

        for name in set_order:
            specs = wanted.get(name)
            if not specs:
                continue
            keys = [(s["zip"], s["pdf_name"], s["seg_index"]) for s in specs
                    if (s["zip"], s["pdf_name"], s["seg_index"]) in predictions]
            if not keys:
                continue
            pred = np.concatenate([predictions[k][0] for k in keys])
            target = np.concatenate([predictions[k][1] for k in keys])
            result = metrics(pred, target, num_classes, labels)
            result["segments_scored"] = len(keys)
            entry["sets"][name] = result
            print(
                f"  {name:<15} acc={result['accuracy']:.4f} "
                f"macroF1={result['macro_f1']:.4f} "
                f"wF1={result['weighted_f1']:.4f} "
                f"macroF1(sup>=500)={result['macro_f1_support_ge_500']:.4f} "
                f"({result['valid_count']} blocks)"
            )

        report["models"].append(entry)
        write_report(report, out_path)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_report(report, out_path)
    print()
    print(f"wrote {out_path}")

    # ── verdict table ──
    print()
    header = f"{'model':<20}" + "".join(f"{name:>26}" for name in set_order
                                         if name in report["sets"])
    print(header)
    for entry in report["models"]:
        row = f"{entry['id']:<20}"
        for name in set_order:
            result = entry["sets"].get(name)
            row += (f"{result['accuracy']:>13.4f}/{result['macro_f1']:.4f}"
                    if result else f"{'-':>26}")
        print(row)
    print(f"{'':<20}" + "".join(f"{'acc / macroF1':>26}" for name in set_order
                                 if name in report["sets"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())