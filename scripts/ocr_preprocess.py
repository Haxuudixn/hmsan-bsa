"""OCR enrichment for annotated training data.

Reads the annotation tool's ZIP exports, runs OCR on image/mixed blocks whose
`ocr_text` field is empty, and writes an enriched ZIP with the same layout.

Usage:
    python scripts/ocr_preprocess.py --input <in.zip> [--output <out.zip>] [--limit N]

Notes:
    * Always pass an explicit --device. PaddleOCR resolves the default device by
      importing paddlepaddle; on this machine the cuDNN build that onnxruntime
      needs conflicts with paddle's, so the implicit import crashes.
    * --device cpu is faster than --device gpu:0 for this workload. Block images
      vary wildly in size, and the onnxruntime CUDA EP spends 3.5-4.4s compiling
      a cuDNN engine for every new input shape, which never amortizes across
      50k distinct crops.
    * --progress writes a JSONL checkpoint so an interrupted run can resume
      without redoing finished blocks.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

# 环境优化
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["PADDLEOCR_LOG_LEVEL"] = "ERROR"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
logging.getLogger("paddlex").setLevel(logging.ERROR)

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("ERROR: paddleocr not installed. Run: pip install paddleocr")
    sys.exit(1)

_ocr_cache: dict = {}


def get_ocr(device: str = "cpu", engine: str = "onnxruntime",
            textline_orientation: bool = True, doc_preprocess: bool = False):
    """获取（并缓存）OCR 实例，优先使用 ONNX Runtime，回退到 Paddle 引擎。"""
    key = (device, engine, textline_orientation, doc_preprocess)
    if key in _ocr_cache:
        return _ocr_cache[key]

    kwargs = dict(
        lang="ch",
        device=device,
        use_textline_orientation=textline_orientation,
        use_doc_orientation_classify=doc_preprocess,
        use_doc_unwarping=doc_preprocess,
    )

    if engine == "onnxruntime":
        try:
            ocr = PaddleOCR(engine="onnxruntime", **kwargs)
        except Exception as exc:
            try:
                import paddle  # noqa: F401
            except ImportError:
                print(f"[error] ONNX Runtime init failed ({exc}) and paddlepaddle is not installed.")
                raise
            print(f"[warn] ONNX Runtime init failed ({exc}); falling back to Paddle engine with MKLDNN disabled.")
            os.environ["FLAGS_use_mkldnn"] = "0"
            ocr = PaddleOCR(**kwargs)
    else:
        ocr = PaddleOCR(**kwargs)

    _ocr_cache[key] = ocr
    return ocr


def extract_text(img: Image.Image, ocr) -> str:
    """从 PIL Image 提取文本，支持新版（返回 rec_texts）和旧版格式"""
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


def _load_progress(progress_path):
    """读取 JSONL 断点文件，返回 {block_file: ocr_text}"""
    done: dict = {}
    if progress_path and progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[rec["f"]] = rec.get("t", "")
                except (ValueError, KeyError, TypeError):
                    continue
    return done


def _block_file_of(row) -> str:
    return (row.get("block_file") or row.get("content") or "").strip()


def enrich_zip(input_path: Path, output_path: Path, limit: int = 0, device: str = "cpu",
               engine: str = "onnxruntime", textline_orientation: bool = True,
               doc_preprocess: bool = False, progress_path=None) -> dict:
    """Run OCR on image/mixed blocks and write enriched ZIP."""
    done = _load_progress(progress_path)

    stats = {
        "total": 0,
        "ocr_run": 0,
        "skipped_empty": 0,
        "skipped_existing": 0,
        "resumed": 0,
        "total_time": 0.0,
        "ocr_count": 0,
    }

    prog_fh = None
    if progress_path:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        prog_fh = open(progress_path, "a", encoding="utf-8")

    try:
        with zipfile.ZipFile(input_path, "r") as zin:
            csv_bytes = zin.read("annotations.csv")
            try:
                csv_content = csv_bytes.decode("utf-8-sig")
            except UnicodeDecodeError:
                csv_content = csv_bytes.decode("utf-8")

            reader = csv.DictReader(io.StringIO(csv_content))
            fieldnames = reader.fieldnames
            rows = list(reader)

            # 预先统计仍需处理的块总数（用于进度显示）
            total_target = 0
            for row in rows:
                block_type = (row.get("block_type") or "").strip().lower()
                if block_type not in ("image", "mixed"):
                    continue
                if (row.get("ocr_text") or "").strip():
                    continue
                bf = _block_file_of(row)
                if not bf or bf in done:
                    continue
                total_target += 1
            if limit and total_target > limit:
                total_target = limit

            processed_count = 0
            start_time_all = time.time()

            for row in rows:
                stats["total"] += 1

                block_type = (row.get("block_type") or "").strip().lower()
                if block_type not in ("image", "mixed"):
                    continue

                if (row.get("ocr_text") or "").strip():
                    stats["skipped_existing"] += 1
                    continue

                block_file = _block_file_of(row)
                if not block_file:
                    stats["skipped_empty"] += 1
                    continue

                # 断点续跑：该块之前已尝试过（即使识别结果为空也视为已完成）
                if block_file in done:
                    if done[block_file]:
                        row["ocr_text"] = done[block_file]
                        stats["ocr_run"] += 1
                    stats["resumed"] += 1
                    continue

                processed_count += 1
                if limit and processed_count > limit:
                    break

                try:
                    data = zin.read(block_file)
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception:
                    stats["skipped_empty"] += 1
                    if prog_fh:
                        prog_fh.write(json.dumps({"f": block_file, "t": ""}, ensure_ascii=False) + "\n")
                    continue

                start_time = time.time()
                try:
                    text = extract_text(img, get_ocr(device=device, engine=engine,
                                                     textline_orientation=textline_orientation,
                                                     doc_preprocess=doc_preprocess))
                except Exception as exc:
                    print(f"    [warn] OCR failed for {block_file}: {exc}")
                    text = ""
                elapsed = time.time() - start_time
                stats["total_time"] += elapsed
                stats["ocr_count"] += 1

                text = (text or "").strip().replace("\r", " ")
                if prog_fh:
                    prog_fh.write(json.dumps({"f": block_file, "t": text}, ensure_ascii=False) + "\n")
                    if processed_count % 10 == 0:
                        prog_fh.flush()

                if text:
                    row["ocr_text"] = text
                    stats["ocr_run"] += 1

                # 进度显示：每处理10张或最后一张时输出
                if processed_count % 10 == 0 or processed_count == total_target:
                    now = time.time()
                    elapsed_total = now - start_time_all
                    avg_time = elapsed_total / processed_count
                    remaining = (total_target - processed_count) * avg_time
                    progress_pct = (processed_count / total_target * 100) if total_target else 0
                    print(f"  Progress: {processed_count}/{total_target} ({progress_pct:.1f}%) | "
                          f"Avg {avg_time:.2f}s/img | ETA {remaining/60:.1f}min | "
                          f"Last img {block_file} took {elapsed:.2f}s", flush=True)
                    if text:
                        print(f"    -> OCR text: {text[:60]}...", flush=True)

            stats["avg_time"] = stats["total_time"] / stats["ocr_count"] if stats["ocr_count"] else 0.0
    finally:
        if prog_fh:
            prog_fh.close()

    if stats["ocr_count"] == 0 and stats["resumed"] == 0:
        print("[info] Nothing to OCR; skipping output write.")
        return stats

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
    parser.add_argument("--device", default="cpu",
                        help="OCR device, e.g. cpu or gpu:0 (default: cpu). Must be explicit.")
    parser.add_argument("--engine", default="onnxruntime", choices=["onnxruntime", "paddle"],
                        help="OCR engine (default: onnxruntime)")
    parser.add_argument("--no-textline-orientation", dest="textline_orientation",
                        action="store_false", help="Disable per-line orientation classifier")
    parser.add_argument("--doc-preprocess", dest="doc_preprocess", action="store_true",
                        help="Enable document orientation classify + unwarping (slow, for full scans)")
    parser.add_argument("--progress", help="JSONL checkpoint file for resumable runs")
    parser.set_defaults(textline_orientation=True, doc_preprocess=False)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else \
        input_path.with_name(input_path.stem + "_ocr.zip")

    progress_path = Path(args.progress) if args.progress else None

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Device: {args.device} | Engine: {args.engine} | "
          f"TextlineOrientation: {args.textline_orientation} | DocPreprocess: {args.doc_preprocess}")
    if progress_path:
        print(f"Progress checkpoint: {progress_path}")

    stats = enrich_zip(input_path, output_path, limit=args.limit, device=args.device,
                       engine=args.engine, textline_orientation=args.textline_orientation,
                       doc_preprocess=args.doc_preprocess, progress_path=progress_path)

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