"""OCR enrichment for annotated training data.

Reads the annotation tool's ZIP exports, runs OCR on image/mixed blocks whose
`ocr_text` field is empty, and writes an enriched ZIP with the same layout.

Usage:
    python scripts/ocr_preprocess.py --input <in.zip> [--output <out.zip>]
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

from bid_slicing.features.ocr import extract_text


def enrich_zip(input_path: Path, output_path: Path, limit: int = 0) -> dict:
    """Run OCR on image blocks with empty ocr_text and write enriched ZIP."""
    with zipfile.ZipFile(input_path, "r") as zin:
        # Read CSV rows
        csv_bytes = zin.read("annotations.csv")
        try:
            csv_content = csv_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            csv_content = csv_bytes.decode("utf-8")

        reader = csv.DictReader(io.StringIO(csv_content))
        fieldnames = reader.fieldnames
        rows = list(reader)

        stats = {"total": 0, "ocr_run": 0, "skipped_empty": 0, "skipped_existing": 0}

        for i, row in enumerate(rows):
            stats["total"] += 1

            block_type = (row.get("block_type") or "").strip().lower()
            if block_type not in ("image", "mixed"):
                continue

            existing = (row.get("ocr_text") or "").strip()
            if existing:
                stats["skipped_existing"] += 1
                continue

            block_file = row.get("block_file") or row.get("content") or ""
            if not block_file:
                stats["skipped_empty"] += 1
                continue

            # Load image from ZIP
            try:
                data = zin.read(block_file)
                img = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception as exc:
                print(f"  [warn] cannot read {block_file}: {exc}")
                stats["skipped_empty"] += 1
                continue

            try:
                text = extract_text(img)
            except Exception as exc:
                print(f"  [warn] OCR failed for {block_file}: {exc}")
                continue

            if text.strip():
                row["ocr_text"] = text.strip().replace("\r", " ")
                stats["ocr_run"] += 1
                print(f"  [{i + 1}/{len(rows)}] {block_file}: {text.strip()[:40]}")

            if limit and stats["ocr_run"] >= limit:
                print(f"  [limit] reached {limit} OCR runs, stopping early")
                break

    # Write enriched ZIP
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_path, "r") as zin, \
         zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zout:

        # Copy all entries except annotations.csv
        for name in zin.namelist():
            if name != "annotations.csv":
                zout.writestr(name, zin.read(name))

        # Write enriched CSV
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        zout.writestr("annotations.csv", "\ufeff" + buf.getvalue())

    return stats


def main():
    parser = argparse.ArgumentParser(description="OCR enrichment for training data")
    parser.add_argument("--input", required=True, help="Input training data ZIP")
    parser.add_argument("--output", help="Output enriched ZIP (default: <input>_ocr.zip)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of OCR runs (0 = unlimited, for testing)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else \
        input_path.with_name(input_path.stem + "_ocr.zip")

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("Running OCR...")

    stats = enrich_zip(input_path, output_path, limit=args.limit)

    print("\n=== Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nDone. Enriched data written to {output_path}")


if __name__ == "__main__":
    main()