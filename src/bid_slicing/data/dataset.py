"""Bid Document Dataset - loads annotated training data from ZIP exports."""
from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image


# Valid 19-class labels from the annotation schema
VALID_LABELS = [
    "封面页", "目录", "商务偏差表", "投标保证金", "关系说明",
    "基本情况表", "营业执照", "税务证明", "资格证明文件", "财务状况",
    "财务凭证单", "资质业绩凭证单", "评分支撑材料", "名称变更",
    "一致性承诺函", "十不准", "公章授权书", "其他", "法定代表人授权委托书",
]
LABEL_TO_ID = {l: i for i, l in enumerate(VALID_LABELS)}
IGNORE_INDEX = -100


# Open ZIP handles are reused across block accesses.  DataLoader uses
# num_workers=0 in train.py, so no thread-safety issue in this project.
_ZIP_CACHE: dict[str, zipfile.ZipFile] = {}


def _get_zip(zip_path: str) -> zipfile.ZipFile:
    if zip_path not in _ZIP_CACHE:
        _ZIP_CACHE[zip_path] = zipfile.ZipFile(zip_path, "r")
    return _ZIP_CACHE[zip_path]


def _read_image(zip_path: str, member: str) -> Optional[Image.Image]:
    """Read one image member from a ZIP and decode it to RGB PIL Image."""
    if not zip_path or not member:
        return None
    try:
        z = _get_zip(zip_path)
        data = z.read(member)
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return im.convert("RGB")
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        return None


class Block:
    """A single block in a page."""
    __slots__ = (
        "block_id", "block_type", "bbox", "text", "font", "font_size",
        "is_bold", "is_italic", "font_color", "label", "boundary", "ocr_text",
        "block_file", "page_num", "source_zip",
    )

    def __init__(self, row: dict, source_zip: str = ""):
        self.block_id = _safe_int(row.get("block_id", ""), -1)
        self.block_type = row.get("block_type", "text") or "text"
        self.bbox = _parse_bbox(row.get("bbox", "(0,0,0,0)"))
        self.text = row.get("content", "") or ""
        self.font = row.get("font", "") or ""
        self.font_size = _safe_float(row.get("font_size", 0))
        self.is_bold = int(_safe_float(row.get("is_bold", 0)))
        self.is_italic = int(_safe_float(row.get("is_italic", 0)))
        self.font_color = _parse_color(row.get("font_color", ""))
        self.label = row.get("label", "") or ""
        self.boundary = row.get("boundary_label", "O") or "O"
        self.ocr_text = row.get("ocr_text", "") or ""
        self.block_file = row.get("block_file", "") or ""
        self.page_num = _safe_int(row.get("page_num", "1"), 1)
        self.source_zip = source_zip

    @property
    def label_id(self) -> int:
        lbl = self.label.strip()
        if not lbl or lbl not in LABEL_TO_ID:
            return IGNORE_INDEX
        return LABEL_TO_ID[lbl]

    @property
    def block_type_id(self) -> int:
        t = self.block_type.lower()
        if t == "text":
            return 0
        elif t == "image":
            return 1
        return 2

    def load_image(self) -> Optional[Image.Image]:
        """Load the block image from its source ZIP on demand."""
        if self.block_type_id not in (1, 2):
            return None
        member = self.block_file or ""
        if not member:
            member = self.text if self.text and not self.text.startswith("[") else ""
        return _read_image(self.source_zip, member)


class Page:
    """A single page containing multiple blocks."""
    def __init__(self, page_num: int, blocks: list[Block]):
        self.page_num = page_num
        self.blocks = blocks


class Document:
    """A single PDF document with pages."""
    def __init__(self, pdf_name: str, pages: list[Page], source_zip: str = ""):
        self.pdf_name = pdf_name
        self.pages = sorted(pages, key=lambda p: p.page_num)
        self.source_zip = source_zip


def _parse_bbox(bbox_str: str) -> tuple[float, float, float, float]:
    s = (bbox_str or "(0,0,0,0)").strip("()")
    parts = s.split(",")
    return (
        _safe_float(parts[0]) if len(parts) > 0 else 0,
        _safe_float(parts[1]) if len(parts) > 1 else 0,
        _safe_float(parts[2]) if len(parts) > 2 else 0,
        _safe_float(parts[3]) if len(parts) > 3 else 0,
    )


def _parse_color(color_str: str) -> tuple[int, int, int]:
    s = (color_str or "#000000").strip().lstrip("#")
    try:
        if len(s) >= 6:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        pass
    return (0, 0, 0)


def _safe_float(val) -> float:
    try:
        return float(val) if val != "" else 0.0
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val) if val != "" else default
    except (ValueError, TypeError):
        return default


def load_documents_from_zips(
    zip_paths: list[str],
) -> list[Document]:
    """Load annotated documents from ZIP files.

    Rows are associated with their source ZIP so image blocks can be lazily
    read later by `Block.load_image`.
    """
    all_rows: list[tuple[str, dict]] = []

    for zp in zip_paths:
        zip_path = str(zp)
        with zipfile.ZipFile(zip_path, "r") as z:
            csv_bytes = z.read("annotations.csv")
            # Try utf-8-sig first, then utf-8
            try:
                csv_content = csv_bytes.decode("utf-8-sig")
            except UnicodeDecodeError:
                csv_content = csv_bytes.decode("utf-8")
            reader = csv.DictReader(io.StringIO(csv_content))
            for row in reader:
                all_rows.append((zip_path, row))

    # Group by source ZIP as well as PDF name, so identical PDF filenames from
    # different exports are not accidentally merged.
    pdfs: dict[tuple[str, str], dict[int, list[Block]]] = {}
    for zip_path, row in all_rows:
        pdf_name = row.get("pdf_name", "unknown")
        page_num = _safe_int(row.get("page_num", "1"), 1)
        block = Block(row, source_zip=zip_path)

        key = (zip_path, pdf_name)
        if key not in pdfs:
            pdfs[key] = {}
        if page_num not in pdfs[key]:
            pdfs[key][page_num] = []
        pdfs[key][page_num].append(block)

    documents = []
    for (source_zip, pdf_name), pages_dict in pdfs.items():
        pages = [Page(pn, blks) for pn, blks in sorted(pages_dict.items())]
        documents.append(Document(pdf_name, pages, source_zip=source_zip))

    return documents


class BidDocumentDataset(Dataset):
    """PyTorch Dataset for bid document slice classification."""

    def __init__(self, documents: list[Document]):
        self.documents = documents

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> Document:
        return self.documents[idx]

    def split_by_documents(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> tuple["BidDocumentDataset", "BidDocumentDataset", "BidDocumentDataset"]:
        import random
        random.seed(seed)
        docs = list(self.documents)
        random.shuffle(docs)
        n = len(docs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            BidDocumentDataset(docs[:n_train]),
            BidDocumentDataset(docs[n_train:n_train + n_val]),
            BidDocumentDataset(docs[n_train + n_val:]),
        )


def collate_document(batch: list[Document]) -> dict:
    """Collate one document into flat tensors with page_splits."""
    doc = batch[0]
    all_texts, all_ocr, all_image_blocks = [], [], []
    all_bt, all_bbox, all_fs, all_bold, all_italic, all_color = [], [], [], [], [], []
    all_labels, all_boundaries = [], []
    page_splits = []

    for page in doc.pages:
        for b in page.blocks:
            all_texts.append(b.text)
            all_ocr.append(b.ocr_text)
            all_image_blocks.append(b if b.block_type_id in (1, 2) else None)
            all_bt.append(b.block_type_id)
            all_bbox.append(list(b.bbox) + [0.0] * 4)
            all_fs.append(b.font_size)
            all_bold.append(b.is_bold)
            all_italic.append(b.is_italic)
            all_color.append(list(b.font_color))
            all_labels.append(b.label_id)
            all_boundaries.append(0)
        page_splits.append(len(all_texts))

    return {
        "texts": all_texts,
        "ocr_texts": all_ocr,
        "image_blocks": all_image_blocks,
        "block_type_ids": torch.tensor(all_bt, dtype=torch.long),
        "bbox_norm": torch.tensor(all_bbox, dtype=torch.float),
        "font_size": torch.tensor(all_fs, dtype=torch.float).unsqueeze(-1),
        "is_bold": torch.tensor(all_bold, dtype=torch.float).unsqueeze(-1),
        "is_italic": torch.tensor(all_italic, dtype=torch.float).unsqueeze(-1),
        "font_color": torch.tensor(all_color, dtype=torch.float),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "boundaries": torch.tensor(all_boundaries, dtype=torch.long),
        "page_splits": page_splits,
        "pdf_name": doc.pdf_name,
    }
