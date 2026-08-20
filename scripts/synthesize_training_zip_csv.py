"""Synthesize a training-data ZIP from the annotation tool's CSV export.

CSV fields:
    pdf_name, page_num, block_type, bbox, content, font, font_size,
    is_bold, is_italic, font_color, label, boundary_label, ocr_text, block_id

The resulting ZIP has the same layout used by train.py:
    annotations.csv, annotations.json, blocks/block_NNNN.png, blocks/block_NNNN.txt
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

BBOX_SCALE = 1.5    # CSV bbox is PDF.js pixel coords at scale 1.5
RENDER_SCALE = 3.0  # render pages at 3x for quality (2x bbox)
SCALE_RATIO = RENDER_SCALE / BBOX_SCALE  # 2.0

CSV_FIELDS = [
    "pdf_name", "page_num", "block_type", "bbox", "content", "font",
    "font_size", "is_bold", "is_italic", "font_color", "label",
    "boundary_label", "ocr_text", "block_id", "block_file",
]


def parse_bbox(bbox):
    s = str(bbox or "(0,0,0,0)").strip("()")
    vals = []
    for p in s.split(",")[:4]:
        try:
            vals.append(float(p.strip()))
        except ValueError:
            vals.append(0.0)
    while len(vals) < 4:
        vals.append(0.0)
    return vals


def crop_image_from_samples(samples, pw, ph, bbox):
    x0, y0, x1, y1 = [v * SCALE_RATIO for v in bbox]
    x0 = max(0, int(round(x0)))
    y0 = max(0, int(round(y0)))
    x1 = min(pw, max(x0 + 1, int(round(x1))))
    y1 = min(ph, max(y0 + 1, int(round(y1))))
    if x1 <= x0 or y1 <= y0:
        return None
    arr = np.frombuffer(samples, dtype=np.uint8).reshape(ph, pw, 3)
    crop = arr[y0:y1, x0:x1, :].copy()
    img = Image.fromarray(crop, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_csv(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def synthesize(csv_path: str, pdf_root: str, output_path: str, limit_pages: int = 0):
    csv_path = Path(csv_path)
    pdf_root = Path(pdf_root)
    output_path = Path(output_path)

    records = load_csv(csv_path)
    pdf_map = {}
    for p in pdf_root.rglob("*.pdf"):
        pdf_map[p.name] = str(p)

    rows = []
    img_groups = defaultdict(list)
    missing_pdfs = set()

    for i, rec in enumerate(records):
        pdf_name = rec.get("pdf_name", "") or ""
        if pdf_name not in pdf_map:
            missing_pdfs.add(pdf_name)
            continue

        block_type = (rec.get("block_type") or "text").strip().lower()
        seq = i + 1
        if block_type == "image":
            file_name = f"blocks/block_{seq:04d}.png"
            content = file_name
        else:
            file_name = f"blocks/block_{seq:04d}.txt"
            content = rec.get("content", "") or ""

        row = {
            "pdf_name": pdf_name,
            "page_num": rec.get("page_num", 1),
            "block_type": block_type,
            "bbox": rec.get("bbox", "(0,0,0,0)"),
            "content": content,
            "font": rec.get("font", "") or "",
            "font_size": rec.get("font_size", "") or "0",
            "is_bold": rec.get("is_bold", "") or "0",
            "is_italic": rec.get("is_italic", "") or "0",
            "font_color": rec.get("font_color", "") or "",
            "label": rec.get("label", "") or "",
            "boundary_label": rec.get("boundary_label", "O") or "O",
            "ocr_text": rec.get("ocr_text", "") or "",
            "block_id": rec.get("block_id", "") or "",
            "block_file": file_name,
        }
        row_idx = len(rows)
        rows.append(row)
        if block_type == "image":
            img_groups[(pdf_map[pdf_name], int(float(row["page_num"])))].append(
                (row_idx, parse_bbox(rec.get("bbox")))
            )

    n_img = sum(len(v) for v in img_groups.values())
    n_txt = len(rows) - n_img
    print(f"CSV rows: {len(records)}")
    print(f"Text blocks: {n_txt}, Image blocks: {n_img}")
    print(f"Unique pages to render: {len(img_groups)}")
    if missing_pdfs:
        print(f"Missing PDFs: {len(missing_pdfs)}: {sorted(missing_pdfs)[:10]}")

    rendered = 0
    failed = 0
    page_processed = 0
    image_bytes = {}

    for (pdf_path, page_num), blocks in img_groups.items():
        if limit_pages and page_processed >= limit_pages:
            break
        page_processed += 1
        try:
            doc = fitz.open(pdf_path)
            if page_num < 1 or page_num > doc.page_count:
                doc.close()
                failed += len(blocks)
                continue
            page = doc[page_num - 1]
            mat = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            samples = pix.samples
            pw, ph = pix.width, pix.height
            doc.close()
        except Exception as exc:
            failed += len(blocks)
            print(f"  [warn] page render failed {pdf_path}:{page_num}: {exc}")
            continue

        for row_idx, bbox in blocks:
            try:
                png = crop_image_from_samples(samples, pw, ph, bbox)
                if png is None:
                    failed += 1
                    continue
                image_bytes[row_idx] = png
                rendered += 1
            except Exception as exc:
                failed += 1
                print(f"  [warn] crop failed {pdf_path}:{page_num}: {exc}")

        if page_processed % 100 == 0:
            print(f"  rendered {page_processed}/{len(img_groups)} pages, {rendered} images")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as z:
        for i, row in enumerate(rows):
            if row["block_type"] == "image":
                png = image_bytes.get(i)
                if png is not None:
                    z.writestr(row["block_file"], png)
            else:
                z.writestr(row["block_file"], (row["content"] or "").encode("utf-8"))

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        z.writestr("annotations.csv", "\ufeff" + buf.getvalue())
        z.writestr("annotations.json", json.dumps(rows, ensure_ascii=False, indent=2))

    print(f"\nRendered images: {rendered}, failed: {failed}")
    print(f"Output: {output_path} ({output_path.stat().st_size/1024/1024:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="C:/Users/lizh2/Downloads/annotations_2026-08-19.csv")
    ap.add_argument("--pdf-root", default="C:/Users/lizh2/Desktop/022505/商务文件/022505-2805001-9999-智能一体化电源系统-1")
    ap.add_argument("--output", default="C:/Users/lizh2/Desktop/training_data_2026-08-19.zip")
    ap.add_argument("--limit-pages", type=int, default=0)
    args = ap.parse_args()
    synthesize(args.csv, args.pdf_root, args.output, args.limit_pages)


if __name__ == "__main__":
    main()
