"""Repair malformed rows in the annotation tool CSV export.

Some text content contains unescaped quotes/commas, which makes standard
csv parsing shift columns after `content`.  This script repairs those rows by
parsing the last nine fields from right to left and rejoining the content.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

FIELDNAMES = [
    "pdf_name", "page_num", "block_type", "bbox", "content", "font",
    "font_size", "is_bold", "is_italic", "font_color", "label",
    "boundary_label", "ocr_text", "block_id",
]

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def is_float(value: str) -> bool:
    if value == "":
        return True
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_binary_attr(value: str) -> bool:
    return value.strip() in ("", "0", "1")


def tail_is_clean(row: list[str]) -> bool:
    if len(row) != len(FIELDNAMES):
        return False
    try:
        int(row[1])
    except (ValueError, TypeError):
        return False
    if row[2].strip().lower() not in ("text", "image", "mixed"):
        return False
    if not is_float(row[5]):
        return False
    if not is_binary_attr(row[6]):
        return False
    if not is_binary_attr(row[7]):
        return False
    color = row[8].strip()
    if color and not HEX_RE.match(color):
        return False
    return True


def parse_right(line: str) -> list[str] | None:
    s = line.rstrip("\r\n")
    if not s.endswith('"'):
        return None
    pos = len(s) - 1
    tail: list[str] = []
    font_start = None
    for _ in range(9):
        if pos < 0 or s[pos] != '"':
            return None
        if pos >= 1 and s[pos - 1] == '"':
            start = pos - 1
        else:
            start = s.rfind('"', 0, pos)
        if start < 0:
            return None
        tail.append(s[start + 1 : pos].replace('""', '"'))
        font_start = start
        pos = start - 2

    prefix = s[: font_start - 1]
    try:
        prefix_fields = next(csv.reader([prefix]))
    except StopIteration:
        prefix_fields = []
    if len(prefix_fields) < 4:
        return None
    content = ",".join(prefix_fields[4:])
    return prefix_fields[:4] + [content] + list(reversed(tail))


def repair(input_path: Path, output_path: Path) -> dict:
    stats = {"total": 0, "standard": 0, "repaired": 0, "unrepairable": 0}
    with input_path.open("r", encoding="utf-8-sig", newline="") as fin:
        lines = fin.read().splitlines()

    if not lines:
        raise SystemExit("input CSV is empty")

    with output_path.open("w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_ALL)
        writer.writerow(FIELDNAMES)

        for line_no, line in enumerate(lines[1:], start=2):
            if not line.strip():
                continue
            stats["total"] += 1

            standard = next(csv.reader([line]), None)
            row = None
            if standard is not None and tail_is_clean(standard):
                row = standard
                stats["standard"] += 1
            else:
                repaired = parse_right(line)
                if repaired is not None and len(repaired) == len(FIELDNAMES):
                    row = repaired
                    stats["repaired"] += 1
                else:
                    stats["unrepairable"] += 1
                    print(f"  [unrepairable] line {line_no}: {line[:160]}", file=sys.stderr)

            if row is not None:
                writer.writerow(row)

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="C:/Users/lizh2/Downloads/annotations_2026-08-19.csv")
    ap.add_argument("--output", default="C:/Users/lizh2/Downloads/annotations_2026-08-19.clean.csv")
    args = ap.parse_args()
    stats = repair(Path(args.input), Path(args.output))
    print(stats)
    print("output:", args.output)


if __name__ == "__main__":
    main()
