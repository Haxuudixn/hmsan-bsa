"""OCR enrichment for annotated training data.

Reads the annotation tool's ZIP exports, runs OCR on image/mixed blocks whose
`ocr_text` field is empty, and writes an enriched ZIP with the same layout.

Usage:
    python scripts/ocr_preprocess.py --input <in.zip> [--output <out.zip>] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

# 环境优化
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PADDLEOCR_LOG_LEVEL"] = "ERROR"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("ERROR: paddleocr not installed. Run: pip install paddleocr")
    sys.exit(1)

_ocr = None


def get_ocr():
    """获取单例 OCR 实例，优先使用 ONNX Runtime，回退到 Paddle 引擎"""
    global _ocr
    if _ocr is None:
        try:
            _ocr = PaddleOCR(use_textline_orientation=True, lang='ch', engine='onnxruntime')
        except Exception:
            print("[info] ONNX Runtime not available, falling back to Paddle engine with MKLDNN disabled.")
            os.environ["FLAGS_use_mkldnn"] = "0"
            _ocr = PaddleOCR(use_textline_orientation=True, lang='ch')
    return _ocr


def extract_text(img: Image.Image) -> str:
    """从 PIL Image 提取文本，支持新版（返回 rec_texts）和旧版格式"""
    ocr = get_ocr()
    img_np = np.array(img)

    try:
        results = ocr.predict(img_np)
    except Exception:
        try:
            results = ocr.ocr(img_np)
        except Exception:
            return ""

    if not results:
        return ""

    texts = []
    for res in results:
        if isinstance(res, dict):
            if 'rec_texts' in res:
                rec_list = res['rec_texts']
                if isinstance(rec_list, list):
                    texts.extend(rec_list)
                elif isinstance(rec_list, str):
                    texts.append(rec_list)
            elif 'rec_text' in res:
                rec = res['rec_text']
                if isinstance(rec, str):
                    texts.append(rec)
                elif isinstance(rec, list):
                    texts.extend(rec)
        elif isinstance(res, list):
            for item in res:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    if isinstance(item[1], (list, tuple)) and len(item[1]) >= 1:
                        texts.append(item[1][0])
                elif isinstance(item, dict):
                    if 'rec_text' in item:
                        texts.append(item['rec_text'])
                    elif 'text' in item:
                        texts.append(item['text'])
    return " ".join(texts).strip()


def enrich_zip(input_path: Path, output_path: Path, limit: int = 0) -> dict:
    """Run OCR on image/mixed blocks and write enriched ZIP."""
    with zipfile.ZipFile(input_path, "r") as zin:
        csv_bytes = zin.read("annotations.csv")
        try:
            csv_content = csv_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            csv_content = csv_bytes.decode("utf-8")

        reader = csv.DictReader(io.StringIO(csv_content))
        fieldnames = reader.fieldnames
        rows = list(reader)

        stats = {
            "total": 0,
            "ocr_run": 0,
            "skipped_empty": 0,
            "skipped_existing": 0,
            "total_time": 0.0,
            "ocr_count": 0,
        }

        # 预先统计需要处理的块总数（用于进度显示）
        image_mixed_indices = []
        for idx, row in enumerate(rows):
            block_type = (row.get("block_type") or "").strip().lower()
            if block_type in ("image", "mixed") and not (row.get("ocr_text") or "").strip():
                image_mixed_indices.append(idx)
        total_target = len(image_mixed_indices)
        if limit and total_target > limit:
            total_target = limit

        processed_count = 0
        start_time_all = time.time()
        last_progress_time = start_time_all

        for i, row in enumerate(rows):
            stats["total"] += 1

            block_type = (row.get("block_type") or "").strip().lower()
            if block_type not in ("image", "mixed"):
                continue

            existing = (row.get("ocr_text") or "").strip()
            if existing:
                stats["skipped_existing"] += 1
                continue

            processed_count += 1
            if limit and processed_count > limit:
                break

            block_file = row.get("block_file") or row.get("content") or ""
            if not block_file:
                stats["skipped_empty"] += 1
                continue

            try:
                data = zin.read(block_file)
                img = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                stats["skipped_empty"] += 1
                continue

            start_time = time.time()
            try:
                text = extract_text(img)
            except Exception:
                text = ""
            elapsed = time.time() - start_time
            stats["total_time"] += elapsed
            stats["ocr_count"] += 1

            if text.strip():
                row["ocr_text"] = text.strip().replace("\r", " ")
                stats["ocr_run"] += 1

            # 进度显示：每处理10张或最后一张时输出
            if processed_count % 10 == 0 or processed_count == total_target:
                now = time.time()
                elapsed_total = now - start_time_all
                avg_time = elapsed_total / processed_count
                remaining = (total_target - processed_count) * avg_time
                progress_pct = (processed_count / total_target * 100) if total_target else 0
                print(f"  Progress: {processed_count}/{total_target} ({progress_pct:.1f}%) | "
                      f"Avg {avg_time:.2f}s/img | ETA {remaining:.1f}s | "
                      f"Last img {block_file} took {elapsed:.2f}s")
                if text.strip():
                    print(f"    -> OCR text: {text.strip()[:60]}...")

        stats["avg_time"] = stats["total_time"] / stats["ocr_count"] if stats["ocr_count"] else 0.0

    # 写入输出 ZIP
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_path, "r") as zin, \
         zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in zin.namelist():
            if name != "annotations.csv":
                zout.writestr(name, zin.read(name))
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
                        help="Limit number of image/mixed blocks to process (0 = unlimited)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else \
        input_path.with_name(input_path.stem + "_ocr.zip")

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("Running OCR with PaddleOCR 3.7.0 (using ONNX Runtime)...")

    stats = enrich_zip(input_path, output_path, limit=args.limit)

    print("\n=== Summary ===")
    for k, v in stats.items():
        if k == "total_time":
            print(f"  total_time: {v:.2f}s")
        elif k == "avg_time":
            print(f"  avg_time: {v:.2f}s")
        else:
            print(f"  {k}: {v}")
    print(f"\nDone. Enriched data written to {output_path}")


if __name__ == "__main__":
    main()