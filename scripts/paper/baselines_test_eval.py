"""Score the table-3 baseline rows on the held-out test split -- once.

Why this file exists
--------------------
Table 3 of ``docs/path3_thesis_draft_2026-09-30.md`` mixes denominators: the
main model is read off validation while each baseline carries its own
``history.json`` reading.  The clean matrix now has a test column
(``scripts/paper/eval_clean_test.py``), so the baselines need one too -- the
finished table must not compare a test number against a validation number.

How it works
------------
Reuses ``scripts/paper/baselines.py`` verbatim (models, datasets, collate
function, metric function) by importing it as a module.  That file is left
byte-identical on purpose: a local re-run may be training from it right now.
Nothing here trains, and no run directory is written to -- the only output is
the JSON report.

B3 gets two readings.  ``PageDataset`` truncates every page to
``--max_blocks_per_page`` (default 48), so B3's natural test denominator is
smaller than every other row's; the ``--max_blocks_per_page 3000`` pass
re-scores the same frozen weights over the full denominator.  Both are recorded
under separate keys, and neither is allowed to sit in the same column as the
other.

    python scripts/paper/baselines_test_eval.py --baseline b1 \
        --row_dir outputs/path3_clean/b1_roberta_mlp
    python scripts/paper/baselines_test_eval.py --baseline b3 \
        --row_dir outputs/path3_clean/b3_page_bigru --max_blocks_per_page 3000
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from bid_slicing.data.dataset import (  # noqa: E402
    BidDocumentDataset,
    load_documents_from_zips,
)
from bid_slicing.data.label_schema import LabelSchema  # noqa: E402

import train as train_mod  # noqa: E402

DEFAULT_DATA_DIR = r"C:\Users\Administrator\Desktop\训练数据\final_train"
DEFAULT_OUT = REPO / "outputs" / "path3_diag" / "test_eval_baselines.json"
SPLIT_MANIFEST = REPO / "outputs" / "path3_diag" / "split_manifest.json"
EXPECTED_SPLIT = (306, 66, 62)

TIERS = [("<100", 0, 99), ("100-999", 100, 999),
         ("1k-5k", 1000, 4999), ("5k+", 5000, 10 ** 9)]


def load_baselines_module():
    path = REPO / "scripts" / "paper" / "baselines.py"
    spec = importlib.util.spec_from_file_location("path3_baselines", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def per_class_from_metrics(result: dict, labels: list) -> list:
    """baselines.py returns train.compute_metrics' flat f1_i / support_i keys."""
    return [{
        "id": i,
        "label": labels[i],
        "f1": float(result.get(f"f1_{i}", 0.0)),
        "precision": float(result.get(f"precision_{i}", 0.0)),
        "recall": float(result.get(f"recall_{i}", 0.0)),
        "support": int(result.get(f"support_{i}", 0)),
    } for i in range(len(labels))]


def tier_macro_f1(per_class: list) -> dict:
    tiers, classes, blocks = {}, {}, {}
    for name, lo, hi in TIERS:
        idx = [i for i, item in enumerate(per_class)
               if lo <= item["support"] <= hi]
        tiers[name] = float(np.mean([per_class[i]["f1"] for i in idx])) if idx else None
        classes[name] = len(idx)
        blocks[name] = int(sum(per_class[i]["support"] for i in idx))
    return {"macro_f1": tiers, "classes": classes, "blocks": blocks}


def history_last_epoch(row_dir: Path):
    path = row_dir / "history.json"
    if not path.is_file():
        return None
    records = json.loads(path.read_text(encoding="utf-8"))
    epoch_records = [r for r in records if r.get("phase") != "mid_epoch"]
    if not epoch_records:
        return None
    last = max(epoch_records, key=lambda r: int(r["epoch"]))
    val = last.get("val", {})
    return {"epoch": int(last["epoch"]),
            "accuracy": float(val["accuracy"]),
            "macro_f1": float(val["macro_f1"]),
            "valid_count": int(val.get("valid_count", 0))}


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
    return ok


def write_report(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                        encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True, choices=("b1", "b2", "b3"))
    p.add_argument("--row_dir", required=True,
                   help="the trained row, e.g. outputs/path3_clean/b1_roberta_mlp")
    p.add_argument("--checkpoint", default="",
                   help="default: last.pt if present, else best_model.pt")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--label_config", default="configs/labels.yaml")
    p.add_argument("--encoder_name", default="hfl/chinese-roberta-wwm-ext")
    p.add_argument("--text_max_length", type=int, default=510)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_blocks", type=int, default=32)
    p.add_argument("--batch_pages", type=int, default=8)
    p.add_argument("--max_blocks_per_page", type=int, default=48)
    p.add_argument("--encode_chunk", type=int, default=16)
    p.add_argument("--max_blocks_per_sample", type=int, default=3000)
    p.add_argument("--max_pages_per_sample", type=int, default=150)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--tolerance", type=float, default=1e-3,
                   help="max |diff| allowed between the replayed val metrics and "
                        "the row's own history.json last epoch")
    p.add_argument("--limit_docs", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    baselines = load_baselines_module()
    data_dir = Path(args.data_dir)
    row_dir = Path(args.row_dir)
    out_path = Path(args.out)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        row_dir / "last.pt" if (row_dir / "last.pt").is_file()
        else row_dir / "best_model.pt")
    if not checkpoint_path.is_file():
        raise SystemExit(f"no checkpoint at {checkpoint_path}")

    zip_paths = sorted(str(z) for z in data_dir.glob("training_data_*.zip"))
    if not zip_paths:
        raise SystemExit(f"no training_data_*.zip under {data_dir}")
    print(f"loading {len(zip_paths)} ZIP(s) from {data_dir}")
    documents = load_documents_from_zips(
        zip_paths, args.max_blocks_per_sample, args.max_pages_per_sample)
    splits = BidDocumentDataset(documents).split_by_documents(
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    split = (len(splits[0]), len(splits[1]), len(splits[2]))
    print(f"split: train={split[0]} val={split[1]} test={split[2]}")
    if not verify_split(splits):
        return 2
    if args.limit_docs <= 0 and split != EXPECTED_SPLIT:
        raise SystemExit(f"split {split} != frozen manifest {EXPECTED_SPLIT}")
    val_docs = list(splits[1].documents)
    test_docs = list(splits[2].documents)
    if args.limit_docs:
        val_docs = val_docs[: args.limit_docs]
        test_docs = test_docs[: args.limit_docs]

    print(f"baseline={args.baseline} row={row_dir} checkpoint={checkpoint_path.name} "
          f"cap={args.max_blocks_per_page}")
    if args.dry_run:
        print(f"  val docs={len(val_docs)} test docs={len(test_docs)}")
        print(f"  history last epoch={history_last_epoch(row_dir)}")
        print("dry-run: nothing scored")
        return 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    label_schema = LabelSchema.from_yaml(args.label_config)
    labels = list(label_schema.class_labels)
    num_classes = label_schema.num_classes

    use_layout = args.baseline == "b2"
    if args.baseline in ("b1", "b2"):
        model = baselines.BlockClassifier(
            args.encoder_name, args.text_max_length, num_classes,
            layout_dim=baselines.LAYOUT_DIM if use_layout else 0,
            dropout=args.dropout).to(device)
        dataset = lambda docs: baselines.BlockDataset(docs, use_layout)
        collate = baselines.collate_blocks
        batch_size = args.batch_blocks
        evaluate = baselines.evaluate_blocks
    else:
        model = baselines.PageSequenceClassifier(
            args.encoder_name, args.text_max_length, num_classes,
            dropout=args.dropout, encode_chunk=args.encode_chunk).to(device)
        dataset = lambda docs: baselines.PageDataset(docs, args.max_blocks_per_page)
        collate = lambda batch: batch
        batch_size = args.batch_pages
        evaluate = baselines.evaluate_pages

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    source_epoch = int(state.get("epoch", -1))
    del state
    gc.collect()

    report = {
        "schema": 1,
        "generated_by": "scripts/paper/baselines_test_eval.py",
        "data_dir": str(data_dir),
        "test_policy": ("the test split is scored once, after all training has "
                        "stopped, over frozen checkpoints; this script never "
                        "trains and never writes into a run directory"),
        "val_reference": ("the row's history.json last-epoch record; the replayed "
                          "val column must reproduce it"),
        "split": {"train": split[0], "val": split[1], "test": split[2],
                  "seed": args.seed},
        "models": [],
    }
    key = f"{args.baseline}|{row_dir.name}|{checkpoint_path.name}|cap{args.max_blocks_per_page}"
    if out_path.is_file():
        try:
            previous = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        report["models"] = [e for e in previous.get("models", []) if e.get("key") != key]

    entry = {
        "key": key,
        "baseline": args.baseline,
        "row": row_dir.name,
        "checkpoint": str(checkpoint_path),
        "source_epoch": source_epoch,
        "max_blocks_per_page": args.max_blocks_per_page,
        "sets": {},
    }

    for set_name, docs in (("val", val_docs), ("test", test_docs)):
        loader = DataLoader(
            dataset(docs), batch_size=batch_size, shuffle=False,
            collate_fn=collate, num_workers=0)
        started = time.time()
        result = evaluate(model, loader, device, num_classes, use_amp)
        per_class = per_class_from_metrics(result, labels)
        entry["sets"][set_name] = {
            "accuracy": float(result["accuracy"]),
            "macro_f1": float(result["macro_f1"]),
            "valid_count": int(result["valid_count"]),
            "total_count": int(result.get("total_count", 0)),
            "examples": len(loader.dataset),
            "inference_seconds": round(time.time() - started, 1),
            "per_class": per_class,
            "tiers": tier_macro_f1(per_class),
        }
        print(f"  {set_name:<5} acc={result['accuracy']:.4f} "
              f"macroF1={result['macro_f1']:.4f} "
              f"({result['valid_count']} blocks)")

    reference = history_last_epoch(row_dir)
    if reference and "val" in entry["sets"]:
        got = entry["sets"]["val"]
        deltas = {"accuracy": abs(got["accuracy"] - reference["accuracy"]),
                  "macro_f1": abs(got["macro_f1"] - reference["macro_f1"]),
                  "valid_count": abs(got["valid_count"] - reference["valid_count"])}
        ok = (deltas["accuracy"] <= args.tolerance
              and deltas["macro_f1"] <= args.tolerance
              and deltas["valid_count"] == 0)
        entry["val_fidelity"] = {"reference": reference, "abs_delta": deltas,
                                 "ok": bool(ok), "tolerance": args.tolerance}
        print(f"  fidelity vs history.json ep{reference['epoch']}: "
              f"d_acc={deltas['accuracy']:.2e} d_macroF1={deltas['macro_f1']:.2e} "
              f"d_blocks={deltas['valid_count']}  {'OK' if ok else 'FAIL'}")
    else:
        ok = False
        entry["val_fidelity"] = {"reference": None, "ok": False,
                                 "tolerance": args.tolerance}
        print("  fidelity: no history.json last-epoch record")

    report["models"].append(entry)
    write_report(report, out_path)
    print(f"wrote {out_path}  ({len(report['models'])} entr(ies))")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())