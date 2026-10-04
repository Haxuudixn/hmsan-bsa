"""Step 0 leak audit, phase 1: rebuild the corpus partition on CPU only.

The delivered val/test split may be contaminated: the warm-start checkpoint
``outputs/hmsan_bsa_ft3/best_model.pt`` finished 2026-08-27 17:02, while the
current corpus holds 38 exports dated 08-04..09-17.  Anything the ft3 run
trained on is not a held-out document, yet the seed-42 split drew val/test from
the whole corpus without a time axis or stratification.

This script rebuilds the split with the *exact* logic of
``bid_slicing.data.dataset`` (same zip order, same (zip, pdf_name) grouping,
same page-capped segmentation, same ``random.seed(42)`` shuffle) while only ever
reading ``annotations.csv``.  No text, no images, no torch, no GPU: the phase
cannot disturb the training run that owns the GPU.

It writes ``outputs/path3_diag/corpus_index.json``, which phase 2
(``leak_diag_eval.py``) uses to locate a clean val set, and prints the audit
numbers: per-export counts, split sizes (checked against the values train.py
logged), how much of val/test comes from pre-ft3 exports, and which corpus
definition reproduces ft3's own logged split sizes.

    python scripts/paper/leak_diag_scan.py
    python scripts/paper/leak_diag_scan.py --out outputs/path3_diag/corpus_index.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = r"C:\Users\Administrator\Desktop\训练数据\final_train"
DEFAULT_OUT = REPO / "outputs" / "path3_diag" / "corpus_index.json"
DEFAULT_LABELS = REPO / "configs" / "labels.yaml"

# Locked by the 2026-09-20 training plan (docs/training_plan_2026-09-20.md).
MAX_BLOCKS_PER_SAMPLE = 3000
MAX_PAGES_PER_SAMPLE = 150
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# ft3 finished 2026-08-27 17:02 (outputs/hmsan_bsa_ft3/best_model.pt mtime).
# Exports dated on or before this day are treated as *possibly* already used by
# ft3: the day itself is kept on the contaminated side, because the 08-27
# exports carry that date and we cannot prove they post-date 17:02.
FT3_CUTOFF = "2026-08-27"

# Values train.py printed for the delivered 38-export corpus; the rebuild is
# only trusted if it reproduces them (see verify_split).
EXPECTED = {
    "segments": {"train": 303, "val": 65, "test": 66},
    "train_total_blocks": 310124,
    "train_labeled_blocks": 310123,
    "val_labeled_blocks": 62082,
}

DATE_RE = re.compile(r"training_data_(\d{4}-\d{2}-\d{2})")


# ── helpers mirroring src/bid_slicing/data/dataset.py ──


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val) if val != "" else default
    except (ValueError, TypeError):
        return default


def load_class_labels(path: Path) -> list[str]:
    """Class order from configs/labels.yaml (== VALID_LABELS in dataset.py)."""
    text = path.read_text(encoding="utf-8")
    labels, in_block = [], False
    for line in text.splitlines():
        if line.startswith("class_labels:"):
            in_block = True
            continue
        if in_block:
            stripped = line.strip()
            if stripped.startswith("- "):
                labels.append(stripped[2:].strip())
            elif stripped and not line.startswith(" "):
                break
    if not labels:
        raise SystemExit(f"could not read class_labels from {path}")
    return labels


def zip_date(name: str) -> str:
    match = DATE_RE.search(name)
    if not match:
        raise SystemExit(f"cannot parse a date out of {name!r}")
    return match.group(1)


# ── corpus scan ──


@dataclass
class PageAgg:
    page_num: int
    blocks: int = 0
    labeled: int = 0
    classes: dict = field(default_factory=dict)


@dataclass
class DocAgg:
    pdf_name: str
    source_zip: str
    pages: list


@dataclass
class SegAgg:
    pdf_name: str
    source_zip: str
    zip_date: str
    page_nums: list
    blocks: int
    labeled: int
    classes: dict

    def record(self, seg_index: int, seg_n: int, split: str) -> dict:
        return {
            "pdf_name": self.pdf_name,
            "zip": Path(self.source_zip).name,
            "zip_date": self.zip_date,
            "split": split,
            "seg_index": seg_index,
            "seg_n": seg_n,
            "page_first": self.page_nums[0],
            "page_last": self.page_nums[-1],
            "pages": len(self.page_nums),
            "blocks": self.blocks,
            "labeled": self.labeled,
            "classes": {str(k): v for k, v in sorted(self.classes.items())},
        }


def scan_zip(zip_path: Path, class_to_id: dict) -> list:
    """One Document per (pdf_name) inside this ZIP, in first-seen order."""
    with zipfile.ZipFile(zip_path, "r") as handle:
        raw = handle.read("annotations.csv")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("utf-8")

    docs: "OrderedDict[str, OrderedDict[int, PageAgg]]" = OrderedDict()
    for row in csv.DictReader(io.StringIO(text)):
        pdf_name = row.get("pdf_name", "unknown")
        page_num = _safe_int(row.get("page_num", "1"), 1)
        pages = docs.get(pdf_name)
        if pages is None:
            pages = docs[pdf_name] = OrderedDict()
        page = pages.get(page_num)
        if page is None:
            page = pages[page_num] = PageAgg(page_num)
        page.blocks += 1
        label = (row.get("label") or "").strip()
        if label and label in class_to_id:
            cid = class_to_id[label]
            page.labeled += 1
            page.classes[cid] = page.classes.get(cid, 0) + 1

    return [
        DocAgg(
            pdf_name=name,
            source_zip=str(zip_path),
            pages=sorted(pages.values(), key=lambda p: p.page_num),
        )
        for name, pages in docs.items()
    ]


def segment_document(doc: DocAgg, max_blocks: int, max_pages: int) -> list:
    """Replica of dataset._split_long_document (page-contiguous segments)."""
    def build(pages: list) -> SegAgg:
        classes: dict = {}
        for page in pages:
            for cid, count in page.classes.items():
                classes[cid] = classes.get(cid, 0) + count
        return SegAgg(
            pdf_name=doc.pdf_name,
            source_zip=doc.source_zip,
            zip_date=zip_date(Path(doc.source_zip).name),
            page_nums=[p.page_num for p in pages],
            blocks=sum(p.blocks for p in pages),
            labeled=sum(p.labeled for p in pages),
            classes=classes,
        )

    total_blocks = sum(p.blocks for p in doc.pages)
    too_big = (max_blocks > 0 and total_blocks > max_blocks) or (
        max_pages > 0 and len(doc.pages) > max_pages
    )
    if not too_big:
        return [build(doc.pages)]

    segments, current, current_blocks = [], [], 0
    for page in doc.pages:
        over_blocks = max_blocks > 0 and current_blocks + page.blocks > max_blocks
        over_pages = max_pages > 0 and len(current) + 1 > max_pages
        if current and (over_blocks or over_pages):
            segments.append(build(current))
            current, current_blocks = [], 0
        current.append(page)
        current_blocks += page.blocks
    if current:
        segments.append(build(current))
    return segments


def shuffle_documents(segments: list, seed: int = SEED) -> list:
    """Replica of BidDocumentDataset.split_by_documents ordering."""
    groups: "OrderedDict[str, list]" = OrderedDict()
    for seg in segments:
        groups.setdefault(seg.pdf_name, []).append(seg)
    keys = sorted(groups)
    random.seed(seed)
    random.shuffle(keys)
    return [seg for key in keys for seg in groups[key]]


def take(segments: list, train_ratio: float, val_ratio: float):
    n = len(segments)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        segments[:n_train],
        segments[n_train:n_train + n_val],
        segments[n_train + n_val:],
    )


def labelled(segments: list) -> int:
    return sum(s.labeled for s in segments)


def blocks(segments: list) -> int:
    return sum(s.blocks for s in segments)


def class_vector(segments: list, num_classes: int) -> list:
    counts = [0] * num_classes
    for seg in segments:
        for cid, count in seg.classes.items():
            counts[int(cid)] += count
    return counts


def best_prefix_match(ordered: list, target: int) -> tuple:
    """Document-aligned prefix whose labelled block count is closest to target."""
    running, best = 0, None
    for index, seg in enumerate(ordered):
        running += seg.labeled
        distance = abs(running - target)
        if best is None or distance < best[0]:
            best = (distance, index + 1, running)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--max_blocks_per_sample", type=int, default=MAX_BLOCKS_PER_SAMPLE)
    parser.add_argument("--max_pages_per_sample", type=int, default=MAX_PAGES_PER_SAMPLE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    zip_paths = sorted(data_dir.glob("training_data_*.zip"))
    if not zip_paths:
        raise SystemExit(f"no training_data_*.zip under {data_dir}")

    labels = load_class_labels(Path(args.labels))
    class_to_id = {label: i for i, label in enumerate(labels)}
    num_classes = len(labels)

    print(f"data_dir  : {data_dir}")
    print(f"exports   : {len(zip_paths)}")
    print(f"classes   : {num_classes}")
    print(f"caps      : blocks<={args.max_blocks_per_sample} pages<={args.max_pages_per_sample}")
    print()

    per_zip, all_segments = [], []
    for index, zp in enumerate(zip_paths, 1):
        docs = scan_zip(zp, class_to_id)
        date = zip_date(zp.name)
        segments = [
            seg for doc in docs for seg in segment_document(
                doc, args.max_blocks_per_sample, args.max_pages_per_sample
            )
        ]
        all_segments.extend(segments)
        per_zip.append({
            "zip": zp.name,
            "date": date,
            "docs": len(docs),
            "segments": len(segments),
            "blocks": blocks(segments),
            "labeled": labelled(segments),
        })
        print(
            f"  [{index:2d}/{len(zip_paths)}] {zp.name:<34} {date}  "
            f"docs={len(docs):>3} segs={len(segments):>3} "
            f"blocks={blocks(segments):>7} labeled={labelled(segments):>7}",
            flush=True,
        )

    # Segments per source document, in corpus order.  A locator must be a
    # document-relative index: when the split puts a document on two sides, the
    # split-relative index of its segments is not the index the loader uses.
    doc_segments: "OrderedDict[tuple, list]" = OrderedDict()
    for seg in all_segments:
        doc_segments.setdefault((Path(seg.source_zip).name, seg.pdf_name), []).append(seg)

    total_blocks = blocks(all_segments)
    total_labeled = labelled(all_segments)
    print()
    print(f"corpus: {len(all_segments)} segments, {total_blocks} blocks, {total_labeled} labeled")

    ordered = shuffle_documents(all_segments, args.seed)
    train, val, test = take(ordered, TRAIN_RATIO, VAL_RATIO)
    print(
        f"split  : train={len(train)} val={len(val)} test={len(test)}  "
        f"(block counts train={blocks(train)}/{labelled(train)} val={labelled(val)} "
        f"test={labelled(test)})"
    )

    checks = {
        "segments_train": len(train) == EXPECTED["segments"]["train"],
        "segments_val": len(val) == EXPECTED["segments"]["val"],
        "segments_test": len(test) == EXPECTED["segments"]["test"],
        "train_total_blocks": blocks(train) == EXPECTED["train_total_blocks"],
        "train_labeled_blocks": labelled(train) == EXPECTED["train_labeled_blocks"],
        "val_labeled_blocks": labelled(val) == EXPECTED["val_labeled_blocks"],
    }
    verified = all(checks.values())
    print(f"rebuild verified against train.py's logged split: {verified}")
    for name, ok in checks.items():
        if not ok:
            print(f"  MISMATCH {name}")
    if not verified:
        print("  refusing to write the index: the rebuild does not match the real run")
        return 2

    # ── contamination accounting ──
    def is_early(seg: SegAgg) -> bool:
        return seg.zip_date <= FT3_CUTOFF

    early_pdf_names = {s.pdf_name for s in all_segments if is_early(s)}
    both = early_pdf_names & {s.pdf_name for s in all_segments if not is_early(s)}

    # split_by_documents permutes whole pdf_name groups but then cuts the list at
    # fixed indices, so a group long enough to straddle a cut point lands on both
    # sides of the held-out line.  Those val segments are memorised, not
    # generalised, so they are counted as a second, independent leak.
    name_splits: dict = {}
    for split_name, split_segments in (("train", train), ("val", val), ("test", test)):
        for seg in split_segments:
            name_splits.setdefault(seg.pdf_name, set()).add(split_name)
    straddling = {name for name, splits in name_splits.items() if len(splits) > 1}

    def is_ft3_seen(seg: SegAgg) -> bool:
        return seg.pdf_name in early_pdf_names

    def is_clean(seg: SegAgg) -> bool:
        return not is_ft3_seen(seg) and seg.pdf_name not in straddling

    def audit(segments: list) -> dict:
        total = max(labelled(segments), 1)
        seen = sum(s.labeled for s in segments if is_ft3_seen(s))
        straddled = sum(s.labeled for s in segments if s.pdf_name in straddling)
        either = sum(
            s.labeled for s in segments
            if is_ft3_seen(s) or s.pdf_name in straddling
        )
        clean = [s for s in segments if is_clean(s)]
        return {
            "segments": len(segments),
            "labeled": labelled(segments),
            "ft3_seen_labeled": seen,
            "ft3_seen_share": seen / total,
            "straddle_labeled": straddled,
            "straddle_share": straddled / total,
            "either_labeled": either,
            "either_share": either / total,
            "clean_segments": len(clean),
            "clean_labeled": labelled(clean),
            "clean_share": labelled(clean) / total,
        }

    split_audit = {
        "train": audit(train),
        "val": audit(val),
        "test": audit(test),
    }

    print()
    print(f"pre-ft3 exports (date <= {FT3_CUTOFF}): "
          f"{len(early_pdf_names)} source documents "
          f"({len(both)} names also occur in a post-ft3 export)")
    print(f"documents whose segments the split put on both sides: {len(straddling)}")
    print()
    print(f"{'split':<7}{'segs':>5}{'labeled':>9}{'ft3seen':>9}{'straddle':>10}"
          f"{'either':>8}{'cleanSeg':>9}{'cleanBlk':>9}{'clean%':>8}")
    for name in ("train", "val", "test"):
        item = split_audit[name]
        print(
            f"{name:<7}{item['segments']:>5}{item['labeled']:>9}"
            f"{item['ft3_seen_share'] * 100:>8.1f}%"
            f"{item['straddle_share'] * 100:>9.1f}%"
            f"{item['either_share'] * 100:>7.1f}%"
            f"{item['clean_segments']:>9}{item['clean_labeled']:>9}"
            f"{item['clean_share'] * 100:>7.1f}%"
        )
    print("  ft3seen  = the segment's document also occurs in a pre-ft3 export")
    print("  straddle = the segment's document also has segments in another split")
    print("  clean    = neither, i.e. genuinely unseen and genuinely held out")

    # ── which export set reproduces ft3's logged split? ──
    # ft3's own history.json records train 92,116 / val 11,958 labeled blocks.
    ft3_train, ft3_val = 92116, 11958
    candidates = {}
    dates = sorted({z["date"] for z in per_zip})
    for date in dates:
        subset_zips = [z["zip"] for z in per_zip if z["date"] <= date]
        subset = [s for s in all_segments if Path(s.source_zip).name in set(subset_zips)]
        ordered_subset = shuffle_documents(subset, args.seed)
        sub_train, sub_val, sub_test = take(ordered_subset, TRAIN_RATIO, VAL_RATIO)
        prefix = best_prefix_match(ordered_subset, ft3_train)
        candidates[date] = {
            "zips": len(subset_zips),
            "documents": len({s.pdf_name for s in subset}),
            "labeled_total": labelled(subset),
            "ratio_0.7_0.15": {
                "train": labelled(sub_train),
                "val": labelled(sub_val),
                "test": labelled(sub_test),
            },
            "best_prefix": {
                "documents": prefix[1],
                "labeled": prefix[2],
                "distance_to_92116": prefix[0],
            },
        }

    print()
    print("ft3 logged train=92,116 / val=11,958 labeled blocks on SOME export set.")
    print(f"{'date<=x':<12}{'zips':>5}{'docs':>6}{'labeled':>9}{'0.7/0.15 train':>16}{'val':>8}"
          f"{'best prefix blk':>17}{'gap':>7}")
    for date in dates:
        item = candidates[date]
        ratio = item["ratio_0.7_0.15"]
        pre = item["best_prefix"]
        print(
            f"{date:<12}{item['zips']:>5}{item['documents']:>6}{item['labeled_total']:>9}"
            f"{ratio['train']:>16}{ratio['val']:>8}{pre['labeled']:>17}{pre['distance_to_92116']:>7}"
        )

    train_vec = class_vector(train, num_classes)
    val_vec = class_vector(val, num_classes)
    eval_sets = {
        # the delivered split, the one every checkpoint was selected on
        "dirty_val": val,
        # each leak isolated, to separate their effects
        "val_no_straddle": [s for s in val if s.pdf_name not in straddling],
        "val_no_ft3": [
            s for s in val if not is_ft3_seen(s)
        ],
        # both leaks removed
        "val_clean": [s for s in val if is_clean(s)],
        # the delivered test split, so the paper's own test number can be
        # reproduced before the clean subset is compared against it
        "dirty_test": test,
        "test_clean": [s for s in test if is_clean(s)],
        "clean_val_test": [s for s in (val + test) if is_clean(s)],
    }
    clean_val = eval_sets["val_clean"]
    clean_val_vec = class_vector(clean_val, num_classes)

    print()
    print("class support, train prior vs val vs clean val (train share is the prior the model fits):")
    print(f"{'cls':>4}{'label':<24}{'train':>9}{'train%':>8}{'val':>9}{'val%':>8}"
          f"{'cleanVal':>9}{'clean%':>8}")
    for cid in range(num_classes):
        print(
            f"{cid:>4}{labels[cid]:<24}{train_vec[cid]:>9}"
            f"{100 * train_vec[cid] / max(sum(train_vec), 1):>7.2f}%"
            f"{val_vec[cid]:>9}"
            f"{100 * val_vec[cid] / max(sum(val_vec), 1):>7.2f}%"
            f"{clean_val_vec[cid]:>9}"
            f"{100 * clean_val_vec[cid] / max(sum(clean_val_vec), 1):>7.2f}%"
        )

    index = {
        "schema": 1,
        "generated_by": "scripts/paper/leak_diag_scan.py",
        "data_dir": str(data_dir),
        "class_labels": labels,
        "protocol": {
            "max_blocks_per_sample": args.max_blocks_per_sample,
            "max_pages_per_sample": args.max_pages_per_sample,
            "seed": args.seed,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "ft3_cutoff_date": FT3_CUTOFF,
        },
        "corpus": {
            "exports": len(zip_paths),
            "segments": len(all_segments),
            "blocks": total_blocks,
            "labeled": total_labeled,
            "zips": per_zip,
        },
        "verification": {"expected": EXPECTED, "checks": checks, "verified": True},
        "split_audit": split_audit,
        "ft3_corpus_candidates": candidates,
        "ft3_reference": {"train_labeled": ft3_train, "val_labeled": ft3_val},
        "class_support": {
            "train": train_vec,
            "val": val_vec,
            "clean_val": clean_val_vec,
            "clean_val_test": class_vector(eval_sets["clean_val_test"], num_classes),
            "set_sizes": {
                name: {"segments": len(segments), "labeled": labelled(segments)}
                for name, segments in eval_sets.items()
            },
        },
        "splits": {
            name: [
                seg.record(index, len(siblings), name)
                for siblings, seg, index in _enumerate(segments, doc_segments)
            ]
            for name, segments in (("train", train), ("val", val), ("test", test))
        },
        "eval_sets": {
            name: [
                _locator(seg, index)
                for _, seg, index in _enumerate(segments, doc_segments)
            ]
            for name, segments in eval_sets.items()
        },
        "straddling_documents": sorted(straddling),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    print(f"wrote {out_path}")
    for name, segments in eval_sets.items():
        print(
            f"  {name:<15}: {len(segments):>3} segments / "
            f"{labelled(segments):>6} labeled blocks / "
            f"{len({s.pdf_name for s in segments}):>3} documents"
        )
    return 0


def _enumerate(subset: list, doc_segments: "OrderedDict[tuple, list]"):
    """Yield (siblings, segment, document-relative index) in corpus order.

    ``index`` is the position of the segment inside its source document, which
    is exactly the index ``load_documents_from_zips`` produces, so phase 2 can
    relocate the very same segment by (zip, pdf_name, index).
    """
    wanted = {id(seg) for seg in subset}
    for siblings in doc_segments.values():
        for index, seg in enumerate(siblings):
            if id(seg) in wanted:
                yield siblings, seg, index


def _locator(seg: SegAgg, seg_index: int) -> dict:
    return {
        "zip": Path(seg.source_zip).name,
        "pdf_name": seg.pdf_name,
        "seg_index": seg_index,
        "blocks": seg.blocks,
        "labeled": seg.labeled,
        "page_first": seg.page_nums[0],
        "page_last": seg.page_nums[-1],
    }


if __name__ == "__main__":
    sys.exit(main())