"""Collect corpus statistics for the HMSAN-BSA paper materials.

Pure-stdlib replication of the data pipeline in
``src/bid_slicing/data/dataset.py`` (load -> page-range segmentation -> grouped
train/val/test split), so the numbers reported here match what training sees.

Outputs (into --out_dir):
  package_stats.csv      per-ZIP package aggregates
  class_distribution.csv per-class block counts for the whole corpus and splits
  dataset_stats.json     machine-readable dump of everything below
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import zipfile
from pathlib import Path

VALID_LABELS = [
    "封面页", "目录", "商务偏差表", "投标保证金", "关系说明",
    "基本情况表", "营业执照", "税务证明", "资格证明文件", "财务状况",
    "财务凭证单", "资质业绩凭证单", "评分支撑材料", "名称变更",
    "一致性承诺函", "十不准", "公章授权书", "其他", "法定代表人授权委托书",
]
LABEL_TO_ID = {label: i for i, label in enumerate(VALID_LABELS)}
IGNORE_INDEX = -100


def _safe_int(value, default=0):
    try:
        return int(value) if value not in ("", None) else default
    except (TypeError, ValueError):
        return default


def block_type_id(row):
    """Mirror Block.block_type_id: 0=text, 1=image, 2=mixed/other."""
    kind = (row.get("block_type") or "text").lower()
    if kind == "text":
        return 0
    if kind == "image":
        return 2 if (row.get("ocr_text") or "").strip() else 1
    return 2


def label_id(row):
    label = (row.get("label") or "").strip()
    return LABEL_TO_ID.get(label, IGNORE_INDEX)


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def summarise(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": float(min(values)),
        "mean": round(sum(values) / len(values), 2),
        "p50": round(percentile(values, 0.5), 1),
        "p90": round(percentile(values, 0.9), 1),
        "p99": round(percentile(values, 0.99), 1),
        "max": float(max(values)),
    }


class PageAgg:
    """Per-page aggregate; enough to replay segmentation without keeping rows."""

    __slots__ = ("n_blocks", "label_counts", "type_counts", "ocr_filled", "labeled")

    def __init__(self):
        self.n_blocks = 0
        self.label_counts = [0] * len(VALID_LABELS)
        self.type_counts = {"text": 0, "image": 0, "mixed": 0}
        self.ocr_filled = 0
        self.labeled = 0

    def add(self, row, type_id, lbl):
        self.n_blocks += 1
        kind = "text" if type_id == 0 else ("image" if type_id == 1 else "mixed")
        self.type_counts[kind] += 1
        if (row.get("ocr_text") or "").strip():
            self.ocr_filled += 1
        if lbl != IGNORE_INDEX:
            self.label_counts[lbl] += 1
            self.labeled += 1


def segment_pages(pages, max_blocks, max_pages):
    """Replay _split_long_document over (page_num, PageAgg) pairs, ordered."""
    total_blocks = sum(page.n_blocks for _, page in pages)
    too_big = (max_blocks > 0 and total_blocks > max_blocks) or (
        max_pages > 0 and len(pages) > max_pages
    )
    if not too_big:
        return [list(pages)]
    segments = []
    current = []
    current_blocks = 0
    for entry in pages:
        n_blocks = entry[1].n_blocks
        over_blocks = max_blocks > 0 and current_blocks + n_blocks > max_blocks
        over_pages = max_pages > 0 and len(current) + 1 > max_pages
        if current and (over_blocks or over_pages):
            segments.append(current)
            current = []
            current_blocks = 0
        current.append(entry)
        current_blocks += n_blocks
    if current:
        segments.append(current)
    return segments


def read_package(zip_path, text_len_hist, ocr_len_hist):
    """Parse one ZIP; return (docs -> {page_num: PageAgg}, package counters)."""
    docs = {}
    counters = {
        "zipname": Path(zip_path).name,
        "docs": 0,
        "pages": 0,
        "rows": 0,
        "text": 0,
        "image": 0,
        "mixed": 0,
        "ocr_filled": 0,
        "image_no_ocr": 0,
        "labeled": 0,
        "unlabeled": 0,
        "text_chars": 0,
        "ocr_chars": 0,
        "label_counts": [0] * len(VALID_LABELS),
    }
    with zipfile.ZipFile(zip_path, "r") as handle:
        raw = handle.read("annotations.csv")
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw.decode("utf-8")

    for row in csv.DictReader(io.StringIO(content)):
        pdf_name = row.get("pdf_name", "unknown")
        page_num = _safe_int(row.get("page_num", "1"), 1)
        type_id = block_type_id(row)
        lbl = label_id(row)

        pages = docs.setdefault(pdf_name, {})
        page = pages.get(page_num)
        if page is None:
            page = pages[page_num] = PageAgg()
            counters["pages"] += 1
        page.add(row, type_id, lbl)

        counters["rows"] += 1
        if type_id == 0:
            counters["text"] += 1
            text = row.get("content") or ""
            text_len_hist.append(len(text))
            counters["text_chars"] += len(text)
        elif type_id == 1:
            counters["image"] += 1
            counters["image_no_ocr"] += 1
        else:
            counters["mixed"] += 1

        ocr = (row.get("ocr_text") or "").strip()
        if ocr:
            counters["ocr_filled"] += 1
            ocr_len_hist.append(len(ocr))
            counters["ocr_chars"] += len(ocr)

        if lbl != IGNORE_INDEX:
            counters["labeled"] += 1
            counters["label_counts"][lbl] += 1
        else:
            counters["unlabeled"] += 1

    counters["docs"] = len(docs)
    return docs, counters


def main():
    parser = argparse.ArgumentParser(description="collect corpus statistics")
    parser.add_argument(
        "--data_dir",
        default=r"C:\Users\Administrator\Desktop\训练数据\final_train",
    )
    parser.add_argument("--out_dir", default="docs/paper_materials")
    parser.add_argument("--max_blocks_per_sample", type=int, default=3000)
    parser.add_argument("--max_pages_per_sample", type=int, default=150)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_paths = sorted(data_dir.glob("training_data_*.zip"))
    print(f"packages: {len(zip_paths)}")

    text_len_hist, ocr_len_hist = [], []
    package_rows = []
    ordered_segments = []  # (pdf_name, [pages], zipname), encounter order
    total_pages = 0
    total_blocks = 0
    source_document_keys = set()
    raw_blocks_per_document = []  # pre-segmentation document sizes
    raw_pages_per_document = []

    for zip_path in zip_paths:
        docs, counters = read_package(str(zip_path), text_len_hist, ocr_len_hist)
        package_rows.append(counters)
        total_pages += counters["pages"]
        total_blocks += counters["rows"]
        print(f"  {counters['zipname']}: docs={counters['docs']} "
              f"rows={counters['rows']} ocr={counters['ocr_filled']} "
              f"pages={counters['pages']}")
        for pdf_name, pages_dict in docs.items():
            source_document_keys.add((counters["zipname"], pdf_name))
            pages = sorted(pages_dict.items())
            raw_blocks_per_document.append(sum(p.n_blocks for _, p in pages))
            raw_pages_per_document.append(len(pages))
            for segment in segment_pages(
                pages, args.max_blocks_per_sample, args.max_pages_per_sample
            ):
                ordered_segments.append((pdf_name, segment, counters["zipname"]))

    # Grouped split, replicating BidDocumentDataset.split_by_documents.
    groups = {}
    for pdf_name, segment, source in ordered_segments:
        groups.setdefault(pdf_name, []).append((segment, source))
    keys = sorted(groups)
    random.seed(args.seed)
    random.shuffle(keys)

    ordered = []
    for key in keys:
        for segment, source in groups[key]:
            ordered.append((key, segment, source))

    total_segments = len(ordered)
    n_train = int(total_segments * args.train_ratio)
    n_val = int(total_segments * args.val_ratio)
    split_index = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, total_segments),
    }

    split_stats = {}
    for split, (start, stop) in split_index.items():
        segments = ordered[start:stop]
        label_counts = [0] * len(VALID_LABELS)
        type_counts = {"text": 0, "image": 0, "mixed": 0}
        blocks = pages = labeled = ocr_filled = 0
        for _, segment, _ in segments:
            pages += len(segment)
            for _, page in segment:
                blocks += page.n_blocks
                labeled += page.labeled
                ocr_filled += page.ocr_filled
                for name, value in page.type_counts.items():
                    type_counts[name] += value
                for idx, value in enumerate(page.label_counts):
                    label_counts[idx] += value
        split_stats[split] = {
            "segments": len(segments),
            "unique_documents": len({pdf for pdf, _, _ in segments}),
            "pages": pages,
            "blocks": blocks,
            "labeled_blocks": labeled,
            "label_rate": round(labeled / max(blocks, 1), 4),
            "ocr_filled_blocks": ocr_filled,
            "block_type_counts": type_counts,
            "label_counts": label_counts,
        }

    blocks_per_segment = [sum(p.n_blocks for _, p in segment) for _, segment, _ in ordered]
    pages_per_segment = [len(segment) for _, segment, _ in ordered]

    # Raw (pre-segmentation) document sizes, used to justify the page-range caps.

    docs_over_block_cap = sum(
        1 for value in raw_blocks_per_document
        if args.max_blocks_per_sample > 0 and value > args.max_blocks_per_sample
    )
    docs_over_page_cap = sum(
        1 for value in raw_pages_per_document
        if args.max_pages_per_sample > 0 and value > args.max_pages_per_sample
    )

    total_label_counts = [0] * len(VALID_LABELS)
    total_labeled = total_ocr = 0
    total_type = {"text": 0, "image": 0, "mixed": 0}
    for counters in package_rows:
        total_labeled += counters["labeled"]
        total_ocr += counters["ocr_filled"]
        for name in total_type:
            total_type[name] += counters[name]
        for idx, value in enumerate(counters["label_counts"]):
            total_label_counts[idx] += value

    image_like = total_type["image"] + total_type["mixed"]
    image_with_ocr = total_type["mixed"]
    stats = {
        "corpus": {
            "packages": len(zip_paths),
            "source_documents": len(source_document_keys),
            "unique_pdf_names": len(groups),
            "segments": total_segments,
            "pages": total_pages,
            "blocks": total_blocks,
            "labeled_blocks": total_labeled,
            "label_rate": round(total_labeled / max(total_blocks, 1), 4),
            "blocks_with_ocr_text": total_ocr,
            "image_blocks_total": image_like,
            "image_blocks_with_ocr": image_with_ocr,
            "ocr_coverage_of_image_blocks": round(image_with_ocr / max(image_like, 1), 4),
            "text_blocks_with_ocr_text": total_ocr - image_with_ocr,
            "block_type_counts": total_type,
            "text_chars": sum(c["text_chars"] for c in package_rows),
            "ocr_chars": sum(c["ocr_chars"] for c in package_rows),
        },
        "segmentation": {
            "max_blocks_per_sample": args.max_blocks_per_sample,
            "max_pages_per_sample": args.max_pages_per_sample,
            "blocks_per_segment": summarise(blocks_per_segment),
            "pages_per_segment": summarise(pages_per_segment),
            "raw_blocks_per_document": summarise(raw_blocks_per_document),
            "raw_pages_per_document": summarise(raw_pages_per_document),
            "documents_over_block_cap": docs_over_block_cap,
            "documents_over_page_cap": docs_over_page_cap,
        },
        "split": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "grouping_key": "pdf_name",
            "stats": split_stats,
        },
        "classes": [
            {
                "id": idx,
                "label": label,
                "blocks": total_label_counts[idx],
                "train": split_stats["train"]["label_counts"][idx],
                "val": split_stats["val"]["label_counts"][idx],
                "test": split_stats["test"]["label_counts"][idx],
            }
            for idx, label in enumerate(VALID_LABELS)
        ],
        "text_length_chars": summarise(text_len_hist),
        "ocr_length_chars": summarise(ocr_len_hist),
        "packages": package_rows,
    }

    with open(out_dir / "dataset_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=1)

    with open(out_dir / "package_stats.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "zipname", "docs", "pages", "rows", "text", "image", "mixed",
            "ocr_filled", "image_no_ocr", "labeled", "unlabeled",
            "text_chars", "ocr_chars",
        ])
        for counters in package_rows:
            writer.writerow([
                counters["zipname"], counters["docs"], counters["pages"],
                counters["rows"], counters["text"], counters["image"],
                counters["mixed"], counters["ocr_filled"],
                counters["image_no_ocr"], counters["labeled"],
                counters["unlabeled"], counters["text_chars"],
                counters["ocr_chars"],
            ])

    with open(out_dir / "class_distribution.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "label", "total_blocks", "train", "val", "test"])
        for entry in stats["classes"]:
            writer.writerow([
                entry["id"], entry["label"], entry["blocks"],
                entry["train"], entry["val"], entry["test"],
            ])

    print(json.dumps(stats["corpus"], ensure_ascii=False, indent=1))
    print(json.dumps({
        split: split_stats[split]["segments"] for split in ("train", "val", "test")
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()