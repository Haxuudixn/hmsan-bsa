"""Dump per-document, per-block predictions for the frozen clean split.

Why this file exists
--------------------
Every evaluation script in this directory aggregates on the fly:
``eval_clean_test.py`` and ``baselines_test_eval.py`` keep accuracy, macro F1
and one 19x19 confusion matrix per split, and throw the per-block predictions
away.  That is enough to score a row but not enough to ask *how much of a
difference is explained by which documents the model got right*.  The
independent unit here is the document -- 62 of them on test, not 79,851 blocks
-- and no stored artifact carries a document id, so any document-level test has
to be re-derived from a fresh inference pass.

What it does
------------
Rebuilds the split exactly as ``train.py`` does (same ZIPs, same caps, same
ratios, same seed), asserts it against
``outputs/path3_diag/split_manifest.json``, then replays inference for each
requested checkpoint and writes, per split::

    <name>__<split>.npz        pred (int16), target (int16), doc_index (int32)
    <name>__<split>.meta.json  checkpoint, protocol, doc names, per-doc counts

``doc_index`` indexes the ``doc_names`` list in the meta file, so a paired
resample of documents is a resample of that index.  Blocks that carry
``IGNORE_INDEX`` are kept: the metric mask is applied downstream, exactly as
``train.compute_metrics`` does.

Nothing is trained, nothing is written into a run directory, and no checkpoint
is selected by its test score.

Usage
-----
    python scripts/paper/dump_block_preds.py --split test \
        --main delivered_ep16=outputs/path3_local_calib/delivered_ext16/last.pt:full \
        --page b3_cap3000=outputs/path3_local_calib/b3_page_bigru_cap3000/last.pt:3000
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from bid_slicing.data.dataset import (  # noqa: E402
    IGNORE_INDEX,
    BidDocumentDataset,
    load_documents_from_zips,
)
from bid_slicing.data.label_schema import LabelSchema  # noqa: E402
from bid_slicing.data.vit_cache import ViTFeatureCache  # noqa: E402
from bid_slicing.models.ablation import resolve_preset  # noqa: E402

import baselines  # noqa: E402
import eval_clean_test as ect  # noqa: E402

train_mod = ect.train_mod
DEFAULT_DATA_DIR = ect.DEFAULT_DATA_DIR
DEFAULT_CACHE = ect.DEFAULT_CACHE
SPLIT_MANIFEST = ect.SPLIT_MANIFEST
DEFAULT_OUT_DIR = REPO / "outputs" / "path3_diag" / "block_preds"


# ---------------------------------------------------------------------------
# argument helpers
# ---------------------------------------------------------------------------

def parse_spec(text: str) -> tuple[str, Path, str | None]:
    """``name=path[:mode]`` -> (name, path, mode).

    A Windows path contains a colon too, so the trailing group only counts as
    the mode when it holds no path separator.
    """
    if "=" not in text:
        raise SystemExit(f"bad spec {text!r}; want name=path[:mode]")
    name, _, rest = text.partition("=")
    mode = None
    head, sep, tail = rest.rpartition(":")
    if sep and tail and "\\" not in tail and "/" not in tail:
        rest, mode = head, tail
    return name.strip(), Path(rest).expanduser(), mode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--label_config", default="configs/labels.yaml")
    p.add_argument("--vit_cache_dir", default=str(DEFAULT_CACHE))
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--max_blocks_per_sample", type=int, default=3000)
    p.add_argument("--max_pages_per_sample", type=int, default=150)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", action="append", default=[],
                   choices=("val", "test"),
                   help="repeatable; default is val and test")
    p.add_argument("--main", action="append", default=[],
                   help="HMSAN-BSA checkpoint: name=path[:preset]")
    p.add_argument("--page", action="append", default=[],
                   help="page-sequence baseline (B3): name=path[:max_blocks_per_page]")
    p.add_argument("--encoder_name", default="hfl/chinese-roberta-wwm-ext")
    p.add_argument("--text_max_length", type=int, default=510)
    p.add_argument("--encode_chunk", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--limit_docs", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_split(args):
    data_dir = Path(args.data_dir)
    zip_paths = sorted(str(z) for z in data_dir.glob("training_data_*.zip"))
    if not zip_paths:
        raise SystemExit(f"no training_data_*.zip under {data_dir}")
    print(f"loading {len(zip_paths)} ZIP(s) from {data_dir}")
    documents = load_documents_from_zips(
        zip_paths, args.max_blocks_per_sample, args.max_pages_per_sample)
    splits = BidDocumentDataset(documents).split_by_documents(
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    print(f"split: train={len(splits[0])} val={len(splits[1])} test={len(splits[2])}")
    if not ect.verify_split(splits):
        raise SystemExit("rebuilt split disagrees with split_manifest.json")
    del documents
    gc.collect()
    return splits


# ---------------------------------------------------------------------------
# model builders
# ---------------------------------------------------------------------------

def load_main(name: str, path: Path, preset: str, label_schema, cache, device):
    try:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (RuntimeError, TypeError, ValueError):
        state = torch.load(path, map_location="cpu", weights_only=False)
    _, flags = resolve_preset(preset)
    signature = state.get("ablation_signature")
    if signature and signature != flags.signature():
        raise SystemExit(f"{name}: checkpoint signature {signature} != preset "
                         f"{preset} signature {flags.signature()}")
    model = ect.build_model(label_schema, flags, cache, device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    model.block_encoder.vit_cache_misses = 0
    return model, int(state.get("epoch", -1)), flags.signature()


def load_page(args, path: Path, num_classes, device):
    try:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except (RuntimeError, TypeError, ValueError):
        state = torch.load(path, map_location="cpu", weights_only=False)
    model = baselines.PageSequenceClassifier(
        args.encoder_name, args.text_max_length, num_classes,
        dropout=args.dropout, encode_chunk=args.encode_chunk).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, int(state.get("epoch", -1)), "page-bigru"


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

def _collect(docs, forward_one, device, tag: str):
    """Run ``forward_one(document) -> (pred, target)``; keep block order."""
    preds, targets, indices = [], [], []
    doc_names, skipped = [], []
    started = time.time()
    for position, document in enumerate(docs, 1):
        try:
            pred, target = forward_one(document)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                pred, target = forward_one(document)
            except torch.cuda.OutOfMemoryError:
                skipped.append(document.pdf_name)
                print(f"  [oom] {tag}/{document.pdf_name} skipped", flush=True)
                continue
        pred = np.asarray(pred, dtype=np.int16).reshape(-1)
        target = np.asarray(target, dtype=np.int16).reshape(-1)
        if pred.shape != target.shape:
            raise SystemExit(f"{tag}/{document.pdf_name}: pred {pred.shape} != "
                             f"target {target.shape}")
        preds.append(pred)
        targets.append(target)
        indices.append(np.full(pred.shape, len(doc_names), dtype=np.int32))
        doc_names.append(document.pdf_name)
        if position % 20 == 0 or position == len(docs):
            elapsed = time.time() - started
            print(f"  {tag} {position}/{len(docs)} docs {elapsed:.0f}s "
                  f"({elapsed / position:.1f}s/doc)", flush=True)
    if not preds:
        raise SystemExit(f"{tag}: no document scored")
    return (np.concatenate(preds), np.concatenate(targets),
            np.concatenate(indices), doc_names, skipped,
            round(time.time() - started, 1))


def dump_split(name, kind, path, mode, model, epoch, signature, args,
               docs, split, label_schema, device, use_amp, out_dir, num_classes,
               labels):
    if kind == "main":
        forward_one = lambda document: ect.predict_document(  # noqa: E731
            model, document, device, use_amp)
    else:
        cap = int(mode) if mode else 48

        def forward_one(document):
            doc_pred, doc_target = [], []
            for page in document.pages:
                blocks = page.blocks[:cap]
                if not blocks:
                    continue
                texts = [baselines.block_text(b) for b in blocks]
                with train_mod._autocast(device, use_amp):
                    logits = model.forward_embeddings(model.encode_page(texts))
                doc_pred.extend(logits.argmax(dim=-1).cpu().tolist())
                doc_target.extend(b.label_id for b in blocks)
            return doc_pred, doc_target

    pred, target, doc_index, doc_names, skipped, seconds = _collect(
        docs, forward_one, device, f"{name}/{split}")
    result = ect.metrics(pred, target, num_classes, labels)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{name}__{split}.npz",
                        pred=pred, target=target, doc_index=doc_index)
    per_doc = [int(((doc_index == i) & (target != IGNORE_INDEX)).sum())
               for i in range(len(doc_names))]
    meta = {
        "schema": 1,
        "generated_by": "scripts/paper/dump_block_preds.py",
        "name": name,
        "kind": kind,
        "checkpoint": str(path),
        "mode": mode,
        "preset_or_signature": signature,
        "source_epoch": epoch,
        "split": split,
        "segments_scored": len(doc_names),
        "skipped_segments": skipped,
        "blocks_total": int(pred.size),
        "blocks_valid": int(result["valid_count"]),
        "blocks_per_doc": per_doc,
        "doc_names": doc_names,
        "accuracy": float(result["accuracy"]),
        "macro_f1": float(result["macro_f1"]),
        "tiers": ect.tier_macro_f1(result),
        "ignored_index": int(IGNORE_INDEX),
        "protocol": {
            "max_blocks_per_sample": args.max_blocks_per_sample,
            "max_pages_per_sample": args.max_pages_per_sample,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "amp": bool(use_amp),
            "device": str(device),
        },
        "inference_seconds": seconds,
    }
    meta_path = out_dir / f"{name}__{split}.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print(f"  {split:<5} acc={result['accuracy']:.4f} "
          f"macroF1={result['macro_f1']:.4f} ({result['valid_count']} blocks, "
          f"{len(doc_names)} docs, {seconds:.0f}s)")
    print(f"  wrote {meta_path.name}")
    return meta


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    splits_wanted = args.split or ["val", "test"]
    main_specs = [parse_spec(s) for s in args.main]
    page_specs = [parse_spec(s) for s in args.page]
    if not main_specs and not page_specs:
        raise SystemExit("pass at least one --main or --page checkpoint")

    label_config = Path(args.label_config)
    if not label_config.is_absolute():
        label_config = REPO / label_config
    label_schema = LabelSchema.from_yaml(str(label_config))
    labels = list(label_schema.class_labels)
    num_classes = label_schema.num_classes

    if args.dry_run:
        splits = build_split(args)
        for name, path, mode in main_specs:
            print(f"  [main] {name:<22} {path} preset={mode or 'full'} "
                  f"exists={path.is_file()}")
        for name, path, mode in page_specs:
            print(f"  [page] {name:<22} {path} cap={mode or 48} "
                  f"exists={path.is_file()}")
        print("dry-run: no inference")
        return 0

    splits = build_split(args)
    docs_by_split = {"val": list(splits[1].documents),
                     "test": list(splits[2].documents)}
    if args.limit_docs:
        docs_by_split = {k: v[: args.limit_docs] for k, v in docs_by_split.items()}

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"device={device} amp={use_amp}")
    cache = ViTFeatureCache.open(args.vit_cache_dir)
    print(f"vit cache: {cache.describe()}")

    for name, path, mode in main_specs:
        if not path.is_file():
            raise SystemExit(f"{name}: missing checkpoint {path}")
        wanted = [s for s in splits_wanted
                  if not (args.skip_existing
                          and (out_dir / f"{name}__{s}.npz").is_file())]
        if not wanted:
            print(f"skip {name}: all splits already dumped")
            continue
        print(f"[main] {name} <- {path} (preset {mode or 'full'})")
        model, epoch, signature = load_main(name, path, mode or "full",
                                            label_schema, cache, device)
        for split in wanted:
            dump_split(name, "main", path, mode, model, epoch, signature, args,
                       docs_by_split[split], split, label_schema, device,
                       use_amp, out_dir, num_classes, labels)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for name, path, mode in page_specs:
        if not path.is_file():
            raise SystemExit(f"{name}: missing checkpoint {path}")
        wanted = [s for s in splits_wanted
                  if not (args.skip_existing
                          and (out_dir / f"{name}__{s}.npz").is_file())]
        if not wanted:
            print(f"skip {name}: all splits already dumped")
            continue
        print(f"[page] {name} <- {path} (cap {mode or 48})")
        model, epoch, signature = load_page(args, path, num_classes, device)
        for split in wanted:
            dump_split(name, "page", path, mode, model, epoch, signature, args,
                       docs_by_split[split], split, label_schema, device,
                       use_amp, out_dir, num_classes, labels)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print(f"done; artifacts under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())