"""Verify the train/val/test split never cuts through a source PDF.

Reproduces train.py's dataset construction (same ZIPs, same block/page caps,
same ratios, same seed) and asserts that no ``pdf_name`` appears in more than
one split. A source PDF that straddles two splits leaks content across the
boundary, so this must exit 0 before any matrix row is launched.

Usage:
    python scripts/verify_split_clean.py --data_dir "<final_train>" \
        --manifest_out outputs/path3_diag/split_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bid_slicing.data.dataset import (  # noqa: E402
    BidDocumentDataset,
    IGNORE_INDEX,
    load_documents_from_zips,
)

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_blocks_per_sample", type=int, default=3000)
    p.add_argument("--max_pages_per_sample", type=int, default=150)
    p.add_argument("--manifest_out", type=str, default="")
    return p.parse_args()


def zip_fingerprint(zip_paths: list[str]) -> str:
    h = hashlib.sha256()
    for path in sorted(zip_paths):
        p = Path(path)
        h.update(p.name.encode("utf-8"))
        h.update(str(p.stat().st_size).encode("utf-8"))
    return h.hexdigest()[:16]


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    zip_paths = sorted(str(z) for z in data_dir.glob("training_data_*.zip"))
    if not zip_paths:
        print(f"ERROR: no training_data_*.zip under {data_dir}")
        return 2

    print(f"Loading {len(zip_paths)} ZIP(s)...")
    documents = load_documents_from_zips(
        zip_paths, args.max_blocks_per_sample, args.max_pages_per_sample
    )
    total_blocks = sum(len(p.blocks) for d in documents for p in d.pages)
    print(f"Corpus: {len(documents)} segments, {total_blocks} blocks, "
          f"{len({d.pdf_name for d in documents})} source PDFs")

    dataset = BidDocumentDataset(documents)
    splits = dataset.split_by_documents(
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )

    manifest: dict = {
        "schema": 1,
        "data_dir": str(data_dir),
        "zip_count": len(zip_paths),
        "zips_sha256_16": zip_fingerprint(zip_paths),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "max_blocks_per_sample": args.max_blocks_per_sample,
        "max_pages_per_sample": args.max_pages_per_sample,
        "corpus_segments": len(documents),
        "corpus_blocks": total_blocks,
        "splits": {},
    }

    owners: dict[str, list[str]] = {}
    ok = True

    for name, ds in zip(SPLITS, splits):
        names = [d.pdf_name for d in ds.documents]
        for pdf_name in set(names):
            owners.setdefault(pdf_name, []).append(name)
        blocks = sum(len(p.blocks) for d in ds.documents for p in d.pages)
        labeled = sum(
            1
            for d in ds.documents
            for p in d.pages
            for b in p.blocks
            if b.label_id != IGNORE_INDEX
        )
        support = Counter(
            b.label_id
            for d in ds.documents
            for p in d.pages
            for b in p.blocks
            if b.label_id != IGNORE_INDEX
        )
        manifest["splits"][name] = {
            "segments": len(ds.documents),
            "pdfs": sorted(set(names)),
            "blocks": blocks,
            "labeled_blocks": labeled,
            "class_support": [support.get(i, 0) for i in range(19)],
        }
        if not ds.documents:
            print(f"ERROR: split '{name}' is empty")
            ok = False

    # The whole point of the check: a source PDF must live in exactly one split.
    straddling = {k: v for k, v in owners.items() if len(v) > 1}
    manifest["straddling_pdfs"] = sorted(straddling)

    print()
    header = f"{'split':<8}{'segs':>6}{'pdfs':>6}{'blocks':>10}{'labeled':>10}"
    print(header)
    for name in SPLITS:
        s = manifest["splits"][name]
        print(f"{name:<8}{s['segments']:>6}{len(s['pdfs']):>6}"
              f"{s['blocks']:>10}{s['labeled_blocks']:>10}")

    total_segs = sum(manifest["splits"][n]["segments"] for n in SPLITS)
    print()
    for name in SPLITS:
        s = manifest["splits"][name]
        print(f"  {name:<6} actual segment share = "
              f"{s['segments'] / max(total_segs, 1):.4f}")

    print()
    if straddling:
        ok = False
        print(f"FAIL: {len(straddling)} source PDF(s) appear in more than one split:")
        for pdf_name, where in sorted(straddling.items())[:20]:
            print(f"  {pdf_name}: {where}")
    else:
        print("PASS: no source PDF straddles two splits.")

    if args.manifest_out:
        out = Path(args.manifest_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
        print(f"\nmanifest -> {out}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
