"""
标注导出转换器：将 annotation_tool 导出的 JSON/CSV 合并到 all_blocks.xlsx 格式。

用法:
  python convert_annotations.py annotations_2026-07-31.json --excel all_blocks.xlsx --out merged.xlsx
  python convert_annotations.py annotations_2026-07-31.csv --excel all_blocks.xlsx --out merged.xlsx
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd


def load_annotations(path: Path) -> list[dict]:
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif path.suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig").to_dict("records")
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def merge(excel_path: Path, annotations: list[dict], output_path: Path) -> pd.DataFrame:
    df_existing = pd.read_excel(excel_path, engine="openpyxl") if excel_path.exists() else pd.DataFrame()

    new_rows = []
    for ann in annotations:
        new_rows.append({
            "pdf_name": ann.get("pdf_name", ""),
            "page_num": int(ann.get("page_num", 0)),
            "block_type": ann.get("block_type", "text"),
            "bbox": ann.get("bbox", ""),
            "content": ann.get("content", ""),
            "font": ann.get("font", ""),
            "font_size": str(ann.get("font_size", "")),
            "is_bold": int(ann.get("is_bold", 0)),
            "label": ann.get("label", ""),
        })
    df_new = pd.DataFrame(new_rows)

    if not df_existing.empty:
        # Remove existing rows for the same PDFs to avoid duplicates
        new_pdfs = df_new["pdf_name"].unique()
        df_existing = df_existing[~df_existing["pdf_name"].isin(new_pdfs)]
        df_merged = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_merged = df_new

    df_merged.to_excel(output_path, index=False, engine="openpyxl")
    print(f"Merged {len(df_new)} new annotations into {output_path}")
    print(f"Total rows: {len(df_merged)}")
    return df_merged


def main():
    parser = argparse.ArgumentParser(description="合并标注到 Excel")
    parser.add_argument("annotations", type=Path, help="标注 JSON/CSV 文件")
    parser.add_argument("--excel", type=Path, help="现有 all_blocks.xlsx 路径")
    parser.add_argument("--out", type=Path, default=Path("merged_annotations.xlsx"), help="输出路径")
    args = parser.parse_args()

    anns = load_annotations(args.annotations)
    print(f"Loaded {len(anns)} annotations from {args.annotations}")

    excel = args.excel if args.excel else None
    merge(excel, anns, args.out)


if __name__ == "__main__":
    main()