# HMSAN-BSA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立一个可运行、可测试、可扩展的 HMSAN-BSA 投标文件切片分类工程骨架，支持当前 Excel 标注数据，并为后续补充 PDF 标注后的真实训练预留完整接口。

**Architecture:** 系统采用 `Document -> Page -> Block` 数据层级，先完成 Excel 加载、标签编码、边界标签派生和 PyTorch Dataset，再实现 lite 版 HMSAN-BSA 模型。lite 模型保持与 full 模型一致的接口：块编码、页内聚合、页间记忆、章节记忆、边界头、分类头、多任务 loss、评估和导出。

**Tech Stack:** Python 3.11、PyTorch、pandas、openpyxl、PyYAML、pytest、scikit-learn。第一版不接入大型预训练模型，避免当前小标注集和离线环境阻塞工程闭环。

---

## 0. 文件结构与职责

本计划从空仓库开始创建以下文件：

```text
configs/
  labels.yaml                  # 标签与边界标签配置
  model_hmsan_bsa.yaml          # lite/full 模型配置
  train.yaml                    # 训练与路径配置

src/
  bid_slicing/
    __init__.py
    data/
      __init__.py
      label_schema.py           # 标签编码、ignore_index、配置加载
      excel_loader.py           # 读取 all_blocks.xlsx 并标准化字段
      boundary_builder.py       # 从页/块标签派生 B/I/E/O 边界标签与 section 标签
      dataset.py                # Document/Page/Block 数据对象与 Dataset
      collate.py                # 变长文档/页/块 batch padding
    features/
      __init__.py
      layout_features.py        # bbox/font/block_type 数值特征
      text_features.py          # lite 文本 tokenizer 与词表
      image_features.py         # lite 图像替代特征
    models/
      __init__.py
      block_encoder.py          # BlockMultiModalEncoder lite 版
      page_encoder.py           # 页内 Transformer 聚合
      memory.py                 # page memory 与 section memory 更新
      boundary_head.py          # 边界预测头
      hmsan_bsa.py              # 总模型 forward
    training/
      __init__.py
      losses.py                 # 多任务 loss
      metrics.py                # block/boundary/section 指标
      train.py                  # 训练循环
      evaluate.py               # 评估入口
    inference/
      __init__.py
      predict.py                # 推理入口
      sequence_decoder.py       # Viterbi/规则平滑接口
      export_results.py         # 导出预测 Excel/CSV
    utils/
      __init__.py
      io.py                     # YAML/JSON/目录工具
      seed.py                   # 随机种子
      logging.py                # 简单日志

scripts/
  analyze_labels.py             # 标签统计命令
  prepare_data.py               # 读取 Excel 生成 processed JSONL
  train_hmsan_bsa.py            # 训练命令
  evaluate_hmsan_bsa.py         # 评估命令
  predict_excel.py              # 对 Excel 推理并导出

tests/
  test_label_schema.py
  test_excel_loader.py
  test_boundary_builder.py
  test_dataset_collate.py
  test_model_forward.py
  test_losses_metrics.py
```

每个文件只负责一个清晰边界：数据读取不做模型逻辑，特征构造不做训练逻辑，模型 forward 不做指标计算，脚本只组装已有模块。

---

## Task 1: 初始化 Python 包、配置文件和基础工具

**Files:**
- Create: `pyproject.toml`
- Create: `configs/labels.yaml`
- Create: `configs/model_hmsan_bsa.yaml`
- Create: `configs/train.yaml`
- Create: `src/bid_slicing/__init__.py`
- Create: `src/bid_slicing/utils/io.py`
- Create: `src/bid_slicing/utils/seed.py`
- Create: `src/bid_slicing/utils/logging.py`
- Test: `tests/test_label_schema.py`

- [ ] **Step 1: 写最小项目配置**

Create `pyproject.toml`:

```toml
[project]
name = "bid-slicing-hmsan"
version = "0.1.0"
description = "HMSAN-BSA tender document slicing classifier"
requires-python = ">=3.10"
dependencies = [
  "pandas",
  "openpyxl",
  "pyyaml",
  "torch",
  "scikit-learn",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: 写标签配置**

Create `configs/labels.yaml`:

```yaml
ignore_index: -100
class_labels:
  - 封面页
  - 目录
  - 商务偏差表
  - 投标保证金
  - 关系说明
  - 基本情况表
  - 营业执照
  - 税务证明
  - 资格证明文件
  - 财务状况
  - 财务凭证单
  - 资质业绩凭证单
  - 评分支撑材料
  - 名称变更
  - 一致性承诺函
  - 十不准
  - 公章授权书
  - 其他
  - 法定代表人授权委托书
boundary_labels:
  - O
  - B-SECTION
  - I-SECTION
  - E-SECTION
mixed_section_label: MIXED
```

- [ ] **Step 3: 写模型配置**

Create `configs/model_hmsan_bsa.yaml`:

```yaml
mode: lite
hidden_size: 128
text_vocab_size: 8000
text_max_length: 96
layout_feature_size: 14
image_feature_size: 8
block_type_count: 4
page_encoder_layers: 2
page_encoder_heads: 4
dropout: 0.1
use_sequence_decoder: true
```

- [ ] **Step 4: 写训练配置**

Create `configs/train.yaml`:

```yaml
seed: 42
excel_path: C:\Users\lizh2\Desktop\桌面\毕业论文\try\out_excel\all_blocks.xlsx
processed_dir: data/processed
output_dir: outputs/hmsan_bsa_lite
batch_size: 1
epochs: 5
learning_rate: 0.001
weight_decay: 0.0001
alpha_boundary: 0.5
beta_section: 0.5
gamma_sequence: 0.0
```

- [ ] **Step 5: 写基础工具**

Create `src/bid_slicing/__init__.py`:

```python
"""Bid slicing HMSAN-BSA research prototype."""

__version__ = "0.1.0"
```

Create `src/bid_slicing/utils/io.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
```

Create `src/bid_slicing/utils/seed.py`:

```python
from __future__ import annotations

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

Create `src/bid_slicing/utils/logging.py`:

```python
from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
```

- [ ] **Step 6: 运行基础测试收集**

Run:

```bash
pytest -q
```

Expected:

```text
no tests ran
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml configs src tests
git commit -m "chore: initialize hmsan bsa project"
```

---

## Task 2: 实现标签编码 LabelSchema

**Files:**
- Create: `src/bid_slicing/data/__init__.py`
- Create: `src/bid_slicing/data/label_schema.py`
- Test: `tests/test_label_schema.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_label_schema.py`:

```python
from bid_slicing.data.label_schema import LabelSchema


def test_label_schema_encodes_known_and_unknown_labels():
    schema = LabelSchema(
        class_labels=["封面页", "目录"],
        boundary_labels=["O", "B-SECTION"],
        ignore_index=-100,
        mixed_section_label="MIXED",
    )

    assert schema.encode_class("封面页") == 0
    assert schema.encode_class("目录") == 1
    assert schema.encode_class("") == -100
    assert schema.encode_class(None) == -100
    assert schema.encode_class("不存在") == -100


def test_boundary_and_section_encoding():
    schema = LabelSchema(
        class_labels=["封面页", "目录"],
        boundary_labels=["O", "B-SECTION", "I-SECTION", "E-SECTION"],
        ignore_index=-100,
        mixed_section_label="MIXED",
    )

    assert schema.encode_boundary("O") == 0
    assert schema.encode_boundary("B-SECTION") == 1
    assert schema.encode_boundary("BAD") == 0
    assert schema.num_classes == 2
    assert schema.num_boundaries == 4
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_label_schema.py -q
```

Expected: FAIL，错误包含 `ModuleNotFoundError` 或 `ImportError`。

- [ ] **Step 3: 实现 LabelSchema**

Create `src/bid_slicing/data/__init__.py`:

```python
"""Data loading and schema utilities."""
```

Create `src/bid_slicing/data/label_schema.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bid_slicing.utils.io import read_yaml


@dataclass(frozen=True)
class LabelSchema:
    class_labels: list[str]
    boundary_labels: list[str]
    ignore_index: int = -100
    mixed_section_label: str = "MIXED"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LabelSchema":
        cfg: dict[str, Any] = read_yaml(path)
        return cls(
            class_labels=list(cfg["class_labels"]),
            boundary_labels=list(cfg["boundary_labels"]),
            ignore_index=int(cfg.get("ignore_index", -100)),
            mixed_section_label=str(cfg.get("mixed_section_label", "MIXED")),
        )

    @property
    def class_to_id(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.class_labels)}

    @property
    def boundary_to_id(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.boundary_labels)}

    @property
    def num_classes(self) -> int:
        return len(self.class_labels)

    @property
    def num_boundaries(self) -> int:
        return len(self.boundary_labels)

    def encode_class(self, label: str | None) -> int:
        if label is None:
            return self.ignore_index
        text = str(label).strip()
        if not text:
            return self.ignore_index
        return self.class_to_id.get(text, self.ignore_index)

    def encode_boundary(self, label: str | None) -> int:
        if label is None:
            return self.boundary_to_id["O"]
        return self.boundary_to_id.get(str(label).strip(), self.boundary_to_id["O"])

    def decode_class(self, label_id: int) -> str:
        if label_id == self.ignore_index:
            return ""
        return self.class_labels[label_id]

    def decode_boundary(self, boundary_id: int) -> str:
        return self.boundary_labels[boundary_id]
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
pytest tests/test_label_schema.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/bid_slicing/data tests/test_label_schema.py
git commit -m "feat: add label schema"
```

---

## Task 3: 实现 Excel Loader 与字段标准化

**Files:**
- Create: `src/bid_slicing/data/excel_loader.py`
- Test: `tests/test_excel_loader.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_excel_loader.py`:

```python
import pandas as pd

from bid_slicing.data.excel_loader import load_blocks_excel, parse_bbox


def test_parse_bbox_tuple_string():
    bbox = parse_bbox("(1.0, 2.0, 5.0, 8.0)")
    assert bbox == [1.0, 2.0, 5.0, 8.0]


def test_load_blocks_excel_normalizes_rows(tmp_path):
    path = tmp_path / "blocks.xlsx"
    pd.DataFrame(
        [
            {
                "pdf_name": "a.pdf",
                "page_num": 1,
                "block_type": "text",
                "bbox": "(0, 0, 100, 50)",
                "content": "目录",
                "font": "SimSun",
                "font_size": "12.0",
                "is_bold": True,
                "label": "目录",
            },
            {
                "pdf_name": "a.pdf",
                "page_num": 1,
                "block_type": "image",
                "bbox": "(10, 10, 20, 20)",
                "content": None,
                "font": None,
                "font_size": None,
                "is_bold": None,
                "label": None,
            },
        ]
    ).to_excel(path, index=False)

    rows = load_blocks_excel(path)

    assert len(rows) == 2
    assert rows[0]["document_id"] == "a.pdf"
    assert rows[0]["block_id"] == "a.pdf:1:0"
    assert rows[0]["bbox"] == [0.0, 0.0, 100.0, 50.0]
    assert rows[1]["content"] == ""
    assert rows[1]["font"] == ""
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_excel_loader.py -q
```

Expected: FAIL，错误包含 `No module named` 或 `cannot import name`。

- [ ] **Step 3: 实现 Excel Loader**

Create `src/bid_slicing/data/excel_loader.py`:

```python
from __future__ import annotations

from pathlib import Path
import ast
import math
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "pdf_name",
    "page_num",
    "block_type",
    "bbox",
    "content",
    "font",
    "font_size",
    "is_bold",
    "label",
]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _clean_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _clean_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    text = str(value).split(",")[0].strip()
    try:
        return float(text)
    except ValueError:
        return default


def parse_bbox(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [float(x) for x in value]
    text = _clean_text(value)
    parsed = ast.literal_eval(text)
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        raise ValueError(f"bbox must contain four numbers: {value!r}")
    return [float(x) for x in parsed]


def load_blocks_excel(path: str | Path) -> list[dict[str, Any]]:
    df = pd.read_excel(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for row_index, row in df.iterrows():
        pdf_name = _clean_text(row["pdf_name"])
        page_num = int(row["page_num"])
        page_rows_before = len([r for r in rows if r["pdf_name"] == pdf_name and r["page_num"] == page_num])
        block_id = f"{pdf_name}:{page_num}:{page_rows_before}"
        rows.append(
            {
                "document_id": pdf_name,
                "pdf_name": pdf_name,
                "page_num": page_num,
                "block_id": block_id,
                "block_type": _clean_text(row["block_type"]),
                "bbox": parse_bbox(row["bbox"]),
                "content": _clean_text(row["content"]),
                "font": _clean_text(row["font"]),
                "font_size": _clean_float(row["font_size"]),
                "is_bold": _clean_bool(row["is_bold"]),
                "label": _clean_text(row["label"]),
                "source_row": int(row_index),
            }
        )
    return rows
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
pytest tests/test_excel_loader.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/bid_slicing/data/excel_loader.py tests/test_excel_loader.py
git commit -m "feat: load normalized excel blocks"
```

---

## Task 4: 实现布局特征、文本特征和图像替代特征

**Files:**
- Create: `src/bid_slicing/features/__init__.py`
- Create: `src/bid_slicing/features/layout_features.py`
- Create: `src/bid_slicing/features/text_features.py`
- Create: `src/bid_slicing/features/image_features.py`
- Test: `tests/test_dataset_collate.py`

- [ ] **Step 1: 写失败测试**

Create initial `tests/test_dataset_collate.py`:

```python
from bid_slicing.features.layout_features import build_layout_features
from bid_slicing.features.text_features import CharTokenizer
from bid_slicing.features.image_features import build_image_features


def test_layout_features_have_fixed_size():
    row = {
        "bbox": [0.0, 0.0, 100.0, 50.0],
        "font_size": 12.0,
        "is_bold": True,
        "block_type": "text",
    }
    feats = build_layout_features(row, page_width=200.0, page_height=100.0)
    assert len(feats) == 14
    assert feats[0] == 0.0
    assert feats[2] == 0.5
    assert feats[8] == 1.0


def test_char_tokenizer_encodes_fixed_length():
    tokenizer = CharTokenizer(vocab_size=20, max_length=6)
    ids, mask = tokenizer.encode("目录A")
    assert len(ids) == 6
    assert len(mask) == 6
    assert mask[:3] == [1, 1, 1]
    assert mask[3:] == [0, 0, 0]


def test_image_features_fixed_size():
    feats = build_image_features({"block_type": "image"}, image_feature_size=8)
    assert len(feats) == 8
    assert feats[0] == 1.0
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_dataset_collate.py -q
```

Expected: FAIL，错误包含缺少 features 模块。

- [ ] **Step 3: 实现特征模块**

Create `src/bid_slicing/features/__init__.py`:

```python
"""Feature builders for lite HMSAN-BSA."""
```

Create `src/bid_slicing/features/layout_features.py`:

```python
from __future__ import annotations


BLOCK_TYPES = ["text", "image", "table", "stamp"]


def build_layout_features(
    row: dict,
    page_width: float = 612.0,
    page_height: float = 792.0,
) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in row["bbox"]]
    width = max(x1 - x0, 0.0)
    height = max(y1 - y0, 0.0)
    cx = x0 + width / 2.0
    cy = y0 + height / 2.0
    font_size = float(row.get("font_size", 0.0) or 0.0)
    is_bold = 1.0 if bool(row.get("is_bold", False)) else 0.0
    block_type = str(row.get("block_type", "text"))
    one_hot = [1.0 if block_type == item else 0.0 for item in BLOCK_TYPES]
    return [
        x0 / page_width,
        y0 / page_height,
        x1 / page_width,
        y1 / page_height,
        width / page_width,
        height / page_height,
        cx / page_width,
        cy / page_height,
        is_bold,
        min(font_size / 72.0, 1.0),
        *one_hot,
    ]
```

Create `src/bid_slicing/features/text_features.py`:

```python
from __future__ import annotations


class CharTokenizer:
    def __init__(self, vocab_size: int = 8000, max_length: int = 96) -> None:
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_id = 0
        self.unk_offset = 1

    def _char_to_id(self, ch: str) -> int:
        return (ord(ch) % (self.vocab_size - self.unk_offset)) + self.unk_offset

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        raw_ids = [self._char_to_id(ch) for ch in str(text)[: self.max_length]]
        mask = [1] * len(raw_ids)
        pad_count = self.max_length - len(raw_ids)
        if pad_count > 0:
            raw_ids.extend([self.pad_id] * pad_count)
            mask.extend([0] * pad_count)
        return raw_ids, mask
```

Create `src/bid_slicing/features/image_features.py`:

```python
from __future__ import annotations


def build_image_features(row: dict, image_feature_size: int = 8) -> list[float]:
    feats = [0.0] * image_feature_size
    if str(row.get("block_type", "")) == "image":
        feats[0] = 1.0
    if str(row.get("content", "")).strip():
        feats[1] = 1.0
    return feats
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
pytest tests/test_dataset_collate.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/bid_slicing/features tests/test_dataset_collate.py
git commit -m "feat: add lite feature builders"
```

---

## Task 5: 实现边界与章节标签生成

**Files:**
- Create: `src/bid_slicing/data/boundary_builder.py`
- Test: `tests/test_boundary_builder.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_boundary_builder.py`:

```python
from bid_slicing.data.boundary_builder import add_boundary_labels


def test_add_boundary_labels_for_page_spans():
    rows = [
        {"pdf_name": "a.pdf", "page_num": 1, "block_id": "1", "label": "封面页"},
        {"pdf_name": "a.pdf", "page_num": 2, "block_id": "2", "label": "目录"},
        {"pdf_name": "a.pdf", "page_num": 3, "block_id": "3", "label": "目录"},
        {"pdf_name": "a.pdf", "page_num": 4, "block_id": "4", "label": "投标保证金"},
        {"pdf_name": "a.pdf", "page_num": 5, "block_id": "5", "label": ""},
    ]

    out = add_boundary_labels(rows)

    assert out[0]["boundary_label"] == "B-SECTION"
    assert out[1]["boundary_label"] == "B-SECTION"
    assert out[2]["boundary_label"] == "E-SECTION"
    assert out[3]["boundary_label"] == "B-SECTION"
    assert out[4]["boundary_label"] == "O"
    assert out[1]["section_label"] == "目录"


def test_mixed_page_prefers_continuing_label():
    rows = [
        {"pdf_name": "a.pdf", "page_num": 1, "block_id": "1", "label": "基本情况表"},
        {"pdf_name": "a.pdf", "page_num": 2, "block_id": "2", "label": "基本情况表"},
        {"pdf_name": "a.pdf", "page_num": 2, "block_id": "3", "label": "营业执照"},
    ]

    out = add_boundary_labels(rows)

    assert out[1]["section_label"] == "基本情况表"
    assert out[2]["section_label"] == "基本情况表"
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_boundary_builder.py -q
```

Expected: FAIL，错误包含缺少 `boundary_builder`。

- [ ] **Step 3: 实现边界构造**

Create `src/bid_slicing/data/boundary_builder.py`:

```python
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy


def _page_main_labels(rows: list[dict]) -> dict[tuple[str, int], str]:
    grouped: dict[tuple[str, int], list[str]] = defaultdict(list)
    for row in rows:
        label = str(row.get("label", "")).strip()
        if label:
            grouped[(row["pdf_name"], int(row["page_num"]))].append(label)

    page_labels: dict[tuple[str, int], str] = {}
    ordered_keys = sorted(grouped.keys(), key=lambda x: (x[0], x[1]))
    for key in ordered_keys:
        labels = grouped[key]
        prev_key = (key[0], key[1] - 1)
        next_key = (key[0], key[1] + 1)
        prev_label = page_labels.get(prev_key)
        if prev_label in labels:
            page_labels[key] = prev_label
            continue
        if next_key in grouped:
            next_counts = Counter(grouped[next_key])
            continuing = [label for label in labels if next_counts[label] > 0]
            if continuing:
                page_labels[key] = continuing[0]
                continue
        page_labels[key] = Counter(labels).most_common(1)[0][0]
    return page_labels


def add_boundary_labels(rows: list[dict]) -> list[dict]:
    out = [deepcopy(row) for row in rows]
    page_labels = _page_main_labels(out)
    ordered_pages = sorted(page_labels.keys(), key=lambda x: (x[0], x[1]))

    page_boundary: dict[tuple[str, int], str] = {}
    for index, key in enumerate(ordered_pages):
        label = page_labels[key]
        prev_key = ordered_pages[index - 1] if index > 0 else None
        next_key = ordered_pages[index + 1] if index + 1 < len(ordered_pages) else None
        prev_same = prev_key is not None and prev_key[0] == key[0] and prev_key[1] == key[1] - 1 and page_labels[prev_key] == label
        next_same = next_key is not None and next_key[0] == key[0] and next_key[1] == key[1] + 1 and page_labels[next_key] == label
        if not prev_same:
            page_boundary[key] = "B-SECTION"
        elif next_same:
            page_boundary[key] = "I-SECTION"
        else:
            page_boundary[key] = "E-SECTION"

    for row in out:
        key = (row["pdf_name"], int(row["page_num"]))
        if key in page_labels and str(row.get("label", "")).strip():
            row["section_label"] = page_labels[key]
            row["boundary_label"] = page_boundary[key]
        else:
            row["section_label"] = ""
            row["boundary_label"] = "O"
    return out
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
pytest tests/test_boundary_builder.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/bid_slicing/data/boundary_builder.py tests/test_boundary_builder.py
git commit -m "feat: derive boundary labels"
```

---

## Task 6: 实现 Dataset 与 Collate

**Files:**
- Create: `src/bid_slicing/data/dataset.py`
- Create: `src/bid_slicing/data/collate.py`
- Modify: `tests/test_dataset_collate.py`

- [ ] **Step 1: 追加失败测试**

Append to `tests/test_dataset_collate.py`:

```python
import torch

from bid_slicing.data.dataset import BidBlockDataset
from bid_slicing.data.collate import collate_documents
from bid_slicing.data.label_schema import LabelSchema


def test_dataset_groups_rows_by_document_and_page():
    schema = LabelSchema(["目录"], ["O", "B-SECTION"], -100, "MIXED")
    rows = [
        {"pdf_name": "a.pdf", "page_num": 1, "block_id": "a:1:0", "block_type": "text", "bbox": [0, 0, 10, 10], "content": "目录", "font_size": 12, "is_bold": True, "label": "目录", "boundary_label": "B-SECTION", "section_label": "目录"},
        {"pdf_name": "a.pdf", "page_num": 2, "block_id": "a:2:0", "block_type": "image", "bbox": [0, 0, 10, 10], "content": "", "font_size": 0, "is_bold": False, "label": "", "boundary_label": "O", "section_label": ""},
    ]
    dataset = BidBlockDataset(rows, schema, text_max_length=8)

    item = dataset[0]

    assert item["document_id"] == "a.pdf"
    assert len(item["pages"]) == 2
    assert item["pages"][0]["labels"].tolist() == [0]
    assert item["pages"][1]["labels"].tolist() == [-100]


def test_collate_documents_pads_pages_and_blocks():
    schema = LabelSchema(["目录"], ["O", "B-SECTION"], -100, "MIXED")
    rows = [
        {"pdf_name": "a.pdf", "page_num": 1, "block_id": "a:1:0", "block_type": "text", "bbox": [0, 0, 10, 10], "content": "目录", "font_size": 12, "is_bold": True, "label": "目录", "boundary_label": "B-SECTION", "section_label": "目录"},
    ]
    dataset = BidBlockDataset(rows, schema, text_max_length=8)
    batch = collate_documents([dataset[0]], ignore_index=-100)

    assert batch["text_ids"].shape == torch.Size([1, 1, 1, 8])
    assert batch["layout_features"].shape[-1] == 14
    assert batch["labels"].item() == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_dataset_collate.py -q
```

Expected: FAIL，错误包含缺少 `BidBlockDataset` 或 `collate_documents`。

- [ ] **Step 3: 实现 Dataset**

Create `src/bid_slicing/data/dataset.py`:

```python
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import Dataset

from bid_slicing.data.label_schema import LabelSchema
from bid_slicing.features.image_features import build_image_features
from bid_slicing.features.layout_features import build_layout_features
from bid_slicing.features.text_features import CharTokenizer


class BidBlockDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        schema: LabelSchema,
        text_max_length: int = 96,
        text_vocab_size: int = 8000,
        image_feature_size: int = 8,
    ) -> None:
        self.schema = schema
        self.tokenizer = CharTokenizer(vocab_size=text_vocab_size, max_length=text_max_length)
        self.image_feature_size = image_feature_size
        grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            grouped[row["pdf_name"]][int(row["page_num"])].append(row)
        self.documents = []
        for doc_id in sorted(grouped):
            pages = []
            for page_num in sorted(grouped[doc_id]):
                pages.append((page_num, grouped[doc_id][page_num]))
            self.documents.append((doc_id, pages))

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> dict[str, Any]:
        doc_id, pages_raw = self.documents[index]
        pages = []
        for page_num, rows in pages_raw:
            text_ids = []
            text_mask = []
            layout_features = []
            image_features = []
            labels = []
            boundary_labels = []
            section_labels = []
            block_ids = []
            for row in rows:
                ids, mask = self.tokenizer.encode(row.get("content", ""))
                text_ids.append(ids)
                text_mask.append(mask)
                layout_features.append(build_layout_features(row))
                image_features.append(build_image_features(row, self.image_feature_size))
                labels.append(self.schema.encode_class(row.get("label")))
                boundary_labels.append(self.schema.encode_boundary(row.get("boundary_label")))
                section_labels.append(self.schema.encode_class(row.get("section_label")))
                block_ids.append(row["block_id"])
            pages.append(
                {
                    "page_num": page_num,
                    "block_ids": block_ids,
                    "text_ids": torch.tensor(text_ids, dtype=torch.long),
                    "text_mask": torch.tensor(text_mask, dtype=torch.float32),
                    "layout_features": torch.tensor(layout_features, dtype=torch.float32),
                    "image_features": torch.tensor(image_features, dtype=torch.float32),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "boundary_labels": torch.tensor(boundary_labels, dtype=torch.long),
                    "section_labels": torch.tensor(section_labels, dtype=torch.long),
                }
            )
        return {"document_id": doc_id, "pages": pages}
```

- [ ] **Step 4: 实现 Collate**

Create `src/bid_slicing/data/collate.py`:

```python
from __future__ import annotations

from typing import Any

import torch


def collate_documents(items: list[dict[str, Any]], ignore_index: int = -100) -> dict[str, Any]:
    batch_size = len(items)
    max_pages = max(len(item["pages"]) for item in items)
    max_blocks = max(max(page["text_ids"].shape[0] for page in item["pages"]) for item in items)
    text_len = items[0]["pages"][0]["text_ids"].shape[1]
    layout_size = items[0]["pages"][0]["layout_features"].shape[1]
    image_size = items[0]["pages"][0]["image_features"].shape[1]

    text_ids = torch.zeros(batch_size, max_pages, max_blocks, text_len, dtype=torch.long)
    text_mask = torch.zeros(batch_size, max_pages, max_blocks, text_len, dtype=torch.float32)
    layout_features = torch.zeros(batch_size, max_pages, max_blocks, layout_size, dtype=torch.float32)
    image_features = torch.zeros(batch_size, max_pages, max_blocks, image_size, dtype=torch.float32)
    block_mask = torch.zeros(batch_size, max_pages, max_blocks, dtype=torch.bool)
    labels = torch.full((batch_size, max_pages, max_blocks), ignore_index, dtype=torch.long)
    boundary_labels = torch.zeros(batch_size, max_pages, max_blocks, dtype=torch.long)
    section_labels = torch.full((batch_size, max_pages, max_blocks), ignore_index, dtype=torch.long)
    page_nums: list[list[int]] = []
    block_ids: list[list[list[str]]] = []

    for b, item in enumerate(items):
        doc_page_nums = []
        doc_block_ids = []
        for p, page in enumerate(item["pages"]):
            n = page["text_ids"].shape[0]
            text_ids[b, p, :n] = page["text_ids"]
            text_mask[b, p, :n] = page["text_mask"]
            layout_features[b, p, :n] = page["layout_features"]
            image_features[b, p, :n] = page["image_features"]
            labels[b, p, :n] = page["labels"]
            boundary_labels[b, p, :n] = page["boundary_labels"]
            section_labels[b, p, :n] = page["section_labels"]
            block_mask[b, p, :n] = True
            doc_page_nums.append(page["page_num"])
            doc_block_ids.append(page["block_ids"])
        page_nums.append(doc_page_nums)
        block_ids.append(doc_block_ids)

    return {
        "document_ids": [item["document_id"] for item in items],
        "page_nums": page_nums,
        "block_ids": block_ids,
        "text_ids": text_ids,
        "text_mask": text_mask,
        "layout_features": layout_features,
        "image_features": image_features,
        "block_mask": block_mask,
        "labels": labels,
        "boundary_labels": boundary_labels,
        "section_labels": section_labels,
    }
```

- [ ] **Step 5: 运行测试确认通过**

Run:

```bash
pytest tests/test_dataset_collate.py -q
```

Expected:

```text
5 passed
```

- [ ] **Step 6: Commit**

```bash
git add src/bid_slicing/data/dataset.py src/bid_slicing/data/collate.py tests/test_dataset_collate.py
git commit -m "feat: add document dataset and collate"
```

---

## Task 7: 实现 lite HMSAN-BSA 模型 forward

**Files:**
- Create: `src/bid_slicing/models/__init__.py`
- Create: `src/bid_slicing/models/block_encoder.py`
- Create: `src/bid_slicing/models/page_encoder.py`
- Create: `src/bid_slicing/models/memory.py`
- Create: `src/bid_slicing/models/boundary_head.py`
- Create: `src/bid_slicing/models/hmsan_bsa.py`
- Test: `tests/test_model_forward.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_model_forward.py`:

```python
import torch

from bid_slicing.models.hmsan_bsa import HMSANBSA


def test_hmsan_bsa_forward_shapes():
    model = HMSANBSA(
        vocab_size=100,
        hidden_size=32,
        layout_feature_size=14,
        image_feature_size=8,
        num_classes=3,
        num_boundaries=4,
        page_encoder_layers=1,
        page_encoder_heads=4,
        dropout=0.0,
    )
    batch = {
        "text_ids": torch.randint(0, 100, (1, 2, 3, 8)),
        "text_mask": torch.ones(1, 2, 3, 8),
        "layout_features": torch.randn(1, 2, 3, 14),
        "image_features": torch.randn(1, 2, 3, 8),
        "block_mask": torch.tensor([[[True, True, False], [True, False, False]]]),
    }

    out = model(batch)

    assert out["block_label_logits"].shape == torch.Size([1, 2, 3, 3])
    assert out["boundary_logits"].shape == torch.Size([1, 2, 3, 4])
    assert out["section_label_logits"].shape == torch.Size([1, 2, 3, 3])
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_model_forward.py -q
```

Expected: FAIL，错误包含缺少 `HMSANBSA`。

- [ ] **Step 3: 实现模型模块**

Create `src/bid_slicing/models/__init__.py`:

```python
"""Model components for HMSAN-BSA."""
```

Create `src/bid_slicing/models/block_encoder.py`:

```python
from __future__ import annotations

import torch
from torch import nn


class BlockMultiModalEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        layout_feature_size: int,
        image_feature_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layout_proj = nn.Sequential(
            nn.Linear(layout_feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        layout_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        token_embeddings = self.text_embedding(text_ids)
        mask = text_mask.unsqueeze(-1).clamp(min=0.0, max=1.0)
        denom = mask.sum(dim=-2).clamp(min=1.0)
        text_vec = (token_embeddings * mask).sum(dim=-2) / denom
        layout_vec = self.layout_proj(layout_features)
        image_vec = self.image_proj(image_features)
        return self.fusion(torch.cat([text_vec, layout_vec, image_vec], dim=-1))
```

Create `src/bid_slicing/models/page_encoder.py`:

```python
from __future__ import annotations

import torch
from torch import nn


class PageEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.mem_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(
        self,
        block_embeddings: torch.Tensor,
        block_mask: torch.Tensor,
        prev_page_memory: torch.Tensor,
        prev_section_memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = block_embeddings.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        mem = self.mem_token.expand(batch_size, -1, -1)
        memory_prefix = torch.stack([prev_page_memory, prev_section_memory], dim=1)
        sequence = torch.cat([memory_prefix, cls, block_embeddings, mem], dim=1)
        prefix_mask = torch.ones(batch_size, 4, dtype=torch.bool, device=block_mask.device)
        full_mask = torch.cat([prefix_mask, block_mask], dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=~full_mask)
        page_vec = encoded[:, 2]
        block_ctx = encoded[:, 3 : 3 + block_embeddings.shape[1]]
        page_memory = encoded[:, -1]
        return block_ctx, page_vec, page_memory
```

Create `src/bid_slicing/models/memory.py`:

```python
from __future__ import annotations

import torch
from torch import nn


class SectionMemoryUpdater(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.z = nn.Linear(hidden_size * 2, hidden_size)
        self.r = nn.Linear(hidden_size * 2, hidden_size)
        self.candidate = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, prev_section_memory: torch.Tensor, page_vec: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([prev_section_memory, page_vec], dim=-1)
        z = torch.sigmoid(self.z(joined))
        r = torch.sigmoid(self.r(joined))
        candidate = torch.tanh(self.candidate(torch.cat([r * prev_section_memory, page_vec], dim=-1)))
        return (1.0 - z) * prev_section_memory + z * candidate
```

Create `src/bid_slicing/models/boundary_head.py`:

```python
from __future__ import annotations

from torch import nn


class PredictionHead(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)
```

Create `src/bid_slicing/models/hmsan_bsa.py`:

```python
from __future__ import annotations

import torch
from torch import nn

from bid_slicing.models.block_encoder import BlockMultiModalEncoder
from bid_slicing.models.boundary_head import PredictionHead
from bid_slicing.models.memory import SectionMemoryUpdater
from bid_slicing.models.page_encoder import PageEncoder


class HMSANBSA(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        layout_feature_size: int,
        image_feature_size: int,
        num_classes: int,
        num_boundaries: int,
        page_encoder_layers: int = 2,
        page_encoder_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.block_encoder = BlockMultiModalEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            layout_feature_size=layout_feature_size,
            image_feature_size=image_feature_size,
            dropout=dropout,
        )
        self.page_encoder = PageEncoder(
            hidden_size=hidden_size,
            num_layers=page_encoder_layers,
            num_heads=page_encoder_heads,
            dropout=dropout,
        )
        self.section_memory = SectionMemoryUpdater(hidden_size)
        self.block_head = PredictionHead(hidden_size, num_classes, dropout)
        self.boundary_head = PredictionHead(hidden_size, num_boundaries, dropout)
        self.section_head = PredictionHead(hidden_size, num_classes, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        text_ids = batch["text_ids"]
        text_mask = batch["text_mask"]
        layout_features = batch["layout_features"]
        image_features = batch["image_features"]
        block_mask = batch["block_mask"]
        batch_size, num_pages, max_blocks, text_len = text_ids.shape
        device = text_ids.device

        flat_blocks = batch_size * num_pages * max_blocks
        block_embeddings = self.block_encoder(
            text_ids.reshape(flat_blocks, text_len),
            text_mask.reshape(flat_blocks, text_len),
            layout_features.reshape(flat_blocks, layout_features.shape[-1]),
            image_features.reshape(flat_blocks, image_features.shape[-1]),
        ).reshape(batch_size, num_pages, max_blocks, self.hidden_size)

        page_memory = torch.zeros(batch_size, self.hidden_size, device=device)
        section_memory = torch.zeros(batch_size, self.hidden_size, device=device)
        block_outputs = []
        boundary_outputs = []
        section_outputs = []

        for page_index in range(num_pages):
            block_ctx, page_vec, page_memory = self.page_encoder(
                block_embeddings[:, page_index],
                block_mask[:, page_index],
                page_memory,
                section_memory,
            )
            section_memory = self.section_memory(section_memory, page_vec)
            block_outputs.append(self.block_head(block_ctx))
            boundary_outputs.append(self.boundary_head(block_ctx))
            section_outputs.append(self.section_head(block_ctx))

        return {
            "block_label_logits": torch.stack(block_outputs, dim=1),
            "boundary_logits": torch.stack(boundary_outputs, dim=1),
            "section_label_logits": torch.stack(section_outputs, dim=1),
        }
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
pytest tests/test_model_forward.py -q
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/bid_slicing/models tests/test_model_forward.py
git commit -m "feat: add lite hmsan bsa forward pass"
```

---

## Task 8: 实现多任务 Loss 和指标

**Files:**
- Create: `src/bid_slicing/training/__init__.py`
- Create: `src/bid_slicing/training/losses.py`
- Create: `src/bid_slicing/training/metrics.py`
- Test: `tests/test_losses_metrics.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_losses_metrics.py`:

```python
import torch

from bid_slicing.training.losses import compute_multitask_loss
from bid_slicing.training.metrics import classification_report_dict


def test_compute_multitask_loss_returns_scalar():
    outputs = {
        "block_label_logits": torch.randn(1, 1, 2, 3),
        "boundary_logits": torch.randn(1, 1, 2, 4),
        "section_label_logits": torch.randn(1, 1, 2, 3),
    }
    batch = {
        "labels": torch.tensor([[[1, -100]]]),
        "boundary_labels": torch.tensor([[[1, 0]]]),
        "section_labels": torch.tensor([[[1, -100]]]),
    }
    loss, parts = compute_multitask_loss(outputs, batch, ignore_index=-100)

    assert loss.ndim == 0
    assert set(parts) == {"classification_loss", "boundary_loss", "section_loss"}


def test_classification_report_dict_ignores_masked_labels():
    result = classification_report_dict(
        y_true=[0, 1, -100],
        y_pred=[0, 0, 1],
        labels=["A", "B"],
        ignore_index=-100,
    )

    assert result["support"] == 2
    assert "macro_f1" in result
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_losses_metrics.py -q
```

Expected: FAIL，错误包含缺少 training 模块。

- [ ] **Step 3: 实现 Loss 和 Metrics**

Create `src/bid_slicing/training/__init__.py`:

```python
"""Training utilities for HMSAN-BSA."""
```

Create `src/bid_slicing/training/losses.py`:

```python
from __future__ import annotations

import torch
from torch import nn


def _ce(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    ignore_index: int,
    alpha_boundary: float = 0.5,
    beta_section: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    cls_loss = _ce(outputs["block_label_logits"], batch["labels"], ignore_index)
    boundary_loss = _ce(outputs["boundary_logits"], batch["boundary_labels"], ignore_index=-100)
    section_loss = _ce(outputs["section_label_logits"], batch["section_labels"], ignore_index)
    total = cls_loss + alpha_boundary * boundary_loss + beta_section * section_loss
    return total, {
        "classification_loss": float(cls_loss.detach().cpu()),
        "boundary_loss": float(boundary_loss.detach().cpu()),
        "section_loss": float(section_loss.detach().cpu()),
    }
```

Create `src/bid_slicing/training/metrics.py`:

```python
from __future__ import annotations

from sklearn.metrics import f1_score, accuracy_score


def classification_report_dict(
    y_true: list[int],
    y_pred: list[int],
    labels: list[str],
    ignore_index: int = -100,
) -> dict[str, float | int]:
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != ignore_index]
    if not pairs:
        return {"support": 0, "accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
    true_clean = [t for t, _ in pairs]
    pred_clean = [p for _, p in pairs]
    label_ids = list(range(len(labels)))
    return {
        "support": len(true_clean),
        "accuracy": float(accuracy_score(true_clean, pred_clean)),
        "macro_f1": float(f1_score(true_clean, pred_clean, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_clean, pred_clean, labels=label_ids, average="weighted", zero_division=0)),
    }
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
pytest tests/test_losses_metrics.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/bid_slicing/training tests/test_losses_metrics.py
git commit -m "feat: add multitask loss and metrics"
```

---

## Task 9: 实现数据准备与标签分析脚本

**Files:**
- Create: `scripts/analyze_labels.py`
- Create: `scripts/prepare_data.py`

- [ ] **Step 1: 实现标签分析脚本**

Create `scripts/analyze_labels.py`:

```python
from __future__ import annotations

import argparse
from collections import Counter

from bid_slicing.data.boundary_builder import add_boundary_labels
from bid_slicing.data.excel_loader import load_blocks_excel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True)
    args = parser.parse_args()

    rows = add_boundary_labels(load_blocks_excel(args.excel))
    label_counts = Counter(row["label"] for row in rows if row["label"])
    boundary_counts = Counter(row["boundary_label"] for row in rows)
    pdfs = {row["pdf_name"] for row in rows}
    pages = {(row["pdf_name"], row["page_num"]) for row in rows}
    labeled_pages = {(row["pdf_name"], row["page_num"]) for row in rows if row["label"]}

    print(f"blocks={len(rows)}")
    print(f"pdfs={len(pdfs)}")
    print(f"pages={len(pages)}")
    print(f"labeled_pages={len(labeled_pages)}")
    print("label_counts:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")
    print("boundary_counts:")
    for label, count in boundary_counts.most_common():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 实现数据准备脚本**

Create `scripts/prepare_data.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from bid_slicing.data.boundary_builder import add_boundary_labels
from bid_slicing.data.excel_loader import load_blocks_excel
from bid_slicing.utils.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = add_boundary_labels(load_blocks_excel(args.excel))
    out = Path(args.out)
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行脚本检查当前 Excel**

Run:

```bash
python scripts/analyze_labels.py --excel "C:\Users\lizh2\Desktop\桌面\毕业论文\try\out_excel\all_blocks.xlsx"
```

Expected output includes:

```text
blocks=85304
pdfs=24
pages=8750
labeled_pages=151
```

- [ ] **Step 4: 生成 processed JSONL**

Run:

```bash
python scripts/prepare_data.py --excel "C:\Users\lizh2\Desktop\桌面\毕业论文\try\out_excel\all_blocks.xlsx" --out data/processed/all_blocks.jsonl
```

Expected output includes:

```text
wrote 85304 rows to data\processed\all_blocks.jsonl
```

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_labels.py scripts/prepare_data.py data/processed/.gitkeep
git commit -m "feat: add data preparation scripts"
```

Note for execution: before Step 5, create `data/processed/.gitkeep` if the processed dataset should not be committed. Commit scripts and `.gitkeep`; do not commit `all_blocks.jsonl` unless the user explicitly wants generated data tracked.

---

## Task 10: 实现训练和评估入口

**Files:**
- Create: `src/bid_slicing/training/train.py`
- Create: `src/bid_slicing/training/evaluate.py`
- Create: `scripts/train_hmsan_bsa.py`
- Create: `scripts/evaluate_hmsan_bsa.py`

- [ ] **Step 1: 实现训练模块**

Create `src/bid_slicing/training/train.py`:

```python
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from bid_slicing.data.collate import collate_documents
from bid_slicing.training.losses import compute_multitask_loss


def train_one_epoch(
    model: torch.nn.Module,
    dataset,
    optimizer: torch.optim.Optimizer,
    ignore_index: int,
    batch_size: int = 1,
    alpha_boundary: float = 0.5,
    beta_section: float = 0.5,
    device: str = "cpu",
) -> dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda items: collate_documents(items, ignore_index=ignore_index),
    )
    model.train()
    total_loss = 0.0
    steps = 0
    for batch in loader:
        tensor_batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(tensor_batch)
        loss, _ = compute_multitask_loss(
            outputs,
            tensor_batch,
            ignore_index=ignore_index,
            alpha_boundary=alpha_boundary,
            beta_section=beta_section,
        )
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        steps += 1
    return {"loss": total_loss / max(steps, 1), "steps": steps}
```

- [ ] **Step 2: 实现评估模块**

Create `src/bid_slicing/training/evaluate.py`:

```python
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from bid_slicing.data.collate import collate_documents
from bid_slicing.training.metrics import classification_report_dict


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset,
    labels: list[str],
    ignore_index: int,
    batch_size: int = 1,
    device: str = "cpu",
) -> dict[str, float | int]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda items: collate_documents(items, ignore_index=ignore_index),
    )
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    for batch in loader:
        tensor_batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        outputs = model(tensor_batch)
        pred = outputs["block_label_logits"].argmax(dim=-1).cpu().reshape(-1).tolist()
        true = batch["labels"].cpu().reshape(-1).tolist()
        all_true.extend(true)
        all_pred.extend(pred)
    return classification_report_dict(all_true, all_pred, labels=labels, ignore_index=ignore_index)
```

- [ ] **Step 3: 实现训练脚本**

Create `scripts/train_hmsan_bsa.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bid_slicing.data.boundary_builder import add_boundary_labels
from bid_slicing.data.dataset import BidBlockDataset
from bid_slicing.data.excel_loader import load_blocks_excel
from bid_slicing.data.label_schema import LabelSchema
from bid_slicing.models.hmsan_bsa import HMSANBSA
from bid_slicing.training.evaluate import evaluate_model
from bid_slicing.training.train import train_one_epoch
from bid_slicing.utils.io import ensure_dir, read_yaml
from bid_slicing.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--model-config", default="configs/model_hmsan_bsa.yaml")
    parser.add_argument("--labels-config", default="configs/labels.yaml")
    args = parser.parse_args()

    train_cfg = read_yaml(args.train_config)
    model_cfg = read_yaml(args.model_config)
    schema = LabelSchema.from_yaml(args.labels_config)
    set_seed(int(train_cfg["seed"]))

    rows = add_boundary_labels(load_blocks_excel(train_cfg["excel_path"]))
    labeled_rows = [row for row in rows if row["label"]]
    dataset = BidBlockDataset(
        labeled_rows,
        schema,
        text_max_length=int(model_cfg["text_max_length"]),
        text_vocab_size=int(model_cfg["text_vocab_size"]),
        image_feature_size=int(model_cfg["image_feature_size"]),
    )
    model = HMSANBSA(
        vocab_size=int(model_cfg["text_vocab_size"]),
        hidden_size=int(model_cfg["hidden_size"]),
        layout_feature_size=int(model_cfg["layout_feature_size"]),
        image_feature_size=int(model_cfg["image_feature_size"]),
        num_classes=schema.num_classes,
        num_boundaries=schema.num_boundaries,
        page_encoder_layers=int(model_cfg["page_encoder_layers"]),
        page_encoder_heads=int(model_cfg["page_encoder_heads"]),
        dropout=float(model_cfg["dropout"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    for epoch in range(int(train_cfg["epochs"])):
        train_stats = train_one_epoch(
            model,
            dataset,
            optimizer,
            ignore_index=schema.ignore_index,
            batch_size=int(train_cfg["batch_size"]),
            alpha_boundary=float(train_cfg["alpha_boundary"]),
            beta_section=float(train_cfg["beta_section"]),
        )
        eval_stats = evaluate_model(model, dataset, schema.class_labels, schema.ignore_index)
        print(f"epoch={epoch + 1} train={train_stats} eval={eval_stats}")

    output_dir = ensure_dir(train_cfg["output_dir"])
    ckpt_path = Path(output_dir) / "hmsan_bsa_lite.pt"
    torch.save({"model_state_dict": model.state_dict(), "schema": schema.class_labels}, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 实现评估脚本**

Create `scripts/evaluate_hmsan_bsa.py`:

```python
from __future__ import annotations

import argparse

import torch

from bid_slicing.data.boundary_builder import add_boundary_labels
from bid_slicing.data.dataset import BidBlockDataset
from bid_slicing.data.excel_loader import load_blocks_excel
from bid_slicing.data.label_schema import LabelSchema
from bid_slicing.models.hmsan_bsa import HMSANBSA
from bid_slicing.training.evaluate import evaluate_model
from bid_slicing.utils.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--model-config", default="configs/model_hmsan_bsa.yaml")
    parser.add_argument("--labels-config", default="configs/labels.yaml")
    args = parser.parse_args()

    train_cfg = read_yaml(args.train_config)
    model_cfg = read_yaml(args.model_config)
    schema = LabelSchema.from_yaml(args.labels_config)
    rows = add_boundary_labels(load_blocks_excel(train_cfg["excel_path"]))
    labeled_rows = [row for row in rows if row["label"]]
    dataset = BidBlockDataset(labeled_rows, schema, int(model_cfg["text_max_length"]), int(model_cfg["text_vocab_size"]))
    model = HMSANBSA(
        vocab_size=int(model_cfg["text_vocab_size"]),
        hidden_size=int(model_cfg["hidden_size"]),
        layout_feature_size=int(model_cfg["layout_feature_size"]),
        image_feature_size=int(model_cfg["image_feature_size"]),
        num_classes=schema.num_classes,
        num_boundaries=schema.num_boundaries,
        page_encoder_layers=int(model_cfg["page_encoder_layers"]),
        page_encoder_heads=int(model_cfg["page_encoder_heads"]),
        dropout=float(model_cfg["dropout"]),
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    print(evaluate_model(model, dataset, schema.class_labels, schema.ignore_index))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行训练烟测**

Run:

```bash
python scripts/train_hmsan_bsa.py --train-config configs/train.yaml --model-config configs/model_hmsan_bsa.yaml --labels-config configs/labels.yaml
```

Expected output includes:

```text
epoch=1
saved checkpoint to outputs
```

- [ ] **Step 6: Commit**

```bash
git add src/bid_slicing/training scripts/train_hmsan_bsa.py scripts/evaluate_hmsan_bsa.py
git commit -m "feat: add training and evaluation entrypoints"
```

---

## Task 11: 实现预测、序列平滑和导出

**Files:**
- Create: `src/bid_slicing/inference/__init__.py`
- Create: `src/bid_slicing/inference/sequence_decoder.py`
- Create: `src/bid_slicing/inference/export_results.py`
- Create: `src/bid_slicing/inference/predict.py`
- Create: `scripts/predict_excel.py`

- [ ] **Step 1: 实现序列平滑接口**

Create `src/bid_slicing/inference/__init__.py`:

```python
"""Inference utilities for HMSAN-BSA."""
```

Create `src/bid_slicing/inference/sequence_decoder.py`:

```python
from __future__ import annotations


def smooth_section_predictions(label_ids: list[int], boundary_ids: list[int]) -> list[int]:
    if not label_ids:
        return []
    smoothed = list(label_ids)
    for i in range(1, len(smoothed) - 1):
        if smoothed[i - 1] == smoothed[i + 1] and smoothed[i] != smoothed[i - 1]:
            smoothed[i] = smoothed[i - 1]
    return smoothed
```

- [ ] **Step 2: 实现导出模块**

Create `src/bid_slicing/inference/export_results.py`:

```python
from __future__ import annotations

from pathlib import Path
import pandas as pd


def export_predictions(rows: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(output_path, index=False)
```

- [ ] **Step 3: 实现推理函数**

Create `src/bid_slicing/inference/predict.py`:

```python
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from bid_slicing.data.collate import collate_documents
from bid_slicing.inference.sequence_decoder import smooth_section_predictions


@torch.no_grad()
def predict_dataset(model, dataset, schema, batch_size: int = 1, device: str = "cpu") -> list[dict]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda items: collate_documents(items, ignore_index=schema.ignore_index),
    )
    model.eval()
    results: list[dict] = []
    for batch in loader:
        tensor_batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        outputs = model(tensor_batch)
        pred_labels = outputs["block_label_logits"].argmax(dim=-1).cpu()
        pred_boundaries = outputs["boundary_logits"].argmax(dim=-1).cpu()
        for b, doc_id in enumerate(batch["document_ids"]):
            flat_labels = pred_labels[b].reshape(-1).tolist()
            flat_boundaries = pred_boundaries[b].reshape(-1).tolist()
            flat_labels = smooth_section_predictions(flat_labels, flat_boundaries)
            cursor = 0
            for p, page_num in enumerate(batch["page_nums"][b]):
                for block_id in batch["block_ids"][b][p]:
                    label_id = flat_labels[cursor]
                    boundary_id = flat_boundaries[cursor]
                    results.append(
                        {
                            "pdf_name": doc_id,
                            "page_num": page_num,
                            "block_id": block_id,
                            "pred_label": schema.decode_class(label_id),
                            "pred_boundary": schema.decode_boundary(boundary_id),
                        }
                    )
                    cursor += 1
    return results
```

- [ ] **Step 4: 实现预测脚本**

Create `scripts/predict_excel.py`:

```python
from __future__ import annotations

import argparse

import torch

from bid_slicing.data.boundary_builder import add_boundary_labels
from bid_slicing.data.dataset import BidBlockDataset
from bid_slicing.data.excel_loader import load_blocks_excel
from bid_slicing.data.label_schema import LabelSchema
from bid_slicing.inference.export_results import export_predictions
from bid_slicing.inference.predict import predict_dataset
from bid_slicing.models.hmsan_bsa import HMSANBSA
from bid_slicing.utils.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-config", default="configs/model_hmsan_bsa.yaml")
    parser.add_argument("--labels-config", default="configs/labels.yaml")
    args = parser.parse_args()

    model_cfg = read_yaml(args.model_config)
    schema = LabelSchema.from_yaml(args.labels_config)
    rows = add_boundary_labels(load_blocks_excel(args.excel))
    dataset = BidBlockDataset(rows, schema, int(model_cfg["text_max_length"]), int(model_cfg["text_vocab_size"]))
    model = HMSANBSA(
        vocab_size=int(model_cfg["text_vocab_size"]),
        hidden_size=int(model_cfg["hidden_size"]),
        layout_feature_size=int(model_cfg["layout_feature_size"]),
        image_feature_size=int(model_cfg["image_feature_size"]),
        num_classes=schema.num_classes,
        num_boundaries=schema.num_boundaries,
        page_encoder_layers=int(model_cfg["page_encoder_layers"]),
        page_encoder_heads=int(model_cfg["page_encoder_heads"]),
        dropout=float(model_cfg["dropout"]),
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    predictions = predict_dataset(model, dataset, schema)
    export_predictions(predictions, args.out)
    print(f"wrote predictions to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 运行预测烟测**

Run:

```bash
python scripts/predict_excel.py --excel "C:\Users\lizh2\Desktop\桌面\毕业论文\try\out_excel\all_blocks.xlsx" --checkpoint outputs/hmsan_bsa_lite/hmsan_bsa_lite.pt --out outputs/hmsan_bsa_lite/predictions.xlsx
```

Expected output includes:

```text
wrote predictions to outputs
```

- [ ] **Step 6: Commit**

```bash
git add src/bid_slicing/inference scripts/predict_excel.py
git commit -m "feat: add prediction export pipeline"
```

---

## Task 12: 全量验证与 README 说明

**Files:**
- Create: `README.md`

- [ ] **Step 1: 写 README**

Create `README.md`:

```markdown
# HMSAN-BSA 投标文件切片分类原型

本项目实现 HMSAN-BSA：边界感知的层次化记忆稀疏注意力网络工程骨架。

## 当前能力

- 读取 `all_blocks.xlsx`
- 标准化为 Document/Page/Block 层级
- 生成块标签、章节边界标签、章节标签
- 运行 lite 版 HMSAN-BSA forward
- 计算多任务 loss
- 输出块级分类、边界和章节指标
- 导出预测 Excel

## 数据约束

当前监督标签只覆盖 1 个 PDF 的 151 页和 907 个块。未标注块使用 `ignore_index`，不作为负样本参与监督训练。

## 常用命令

```bash
python scripts/analyze_labels.py --excel "C:\Users\lizh2\Desktop\桌面\毕业论文\try\out_excel\all_blocks.xlsx"
python scripts/prepare_data.py --excel "C:\Users\lizh2\Desktop\桌面\毕业论文\try\out_excel\all_blocks.xlsx" --out data/processed/all_blocks.jsonl
python scripts/train_hmsan_bsa.py
python scripts/evaluate_hmsan_bsa.py --checkpoint outputs/hmsan_bsa_lite/hmsan_bsa_lite.pt
python scripts/predict_excel.py --excel "C:\Users\lizh2\Desktop\桌面\毕业论文\try\out_excel\all_blocks.xlsx" --checkpoint outputs/hmsan_bsa_lite/hmsan_bsa_lite.pt --out outputs/hmsan_bsa_lite/predictions.xlsx
```

## 后续升级方向

- 接入 RoBERTa-wwm-ext 文本编码器
- 接入块截图和 ResNet/ViT 图像编码器
- 接入 Longformer 稀疏注意力
- 增加 CRF/Viterbi 约束解码
- 使用新增 PDF 标注构建 train/validation/test 划分
```

- [ ] **Step 2: 运行全部测试**

Run:

```bash
pytest -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 3: 运行端到端烟测**

Run:

```bash
python scripts/analyze_labels.py --excel "C:\Users\lizh2\Desktop\桌面\毕业论文\try\out_excel\all_blocks.xlsx"
python scripts/train_hmsan_bsa.py
python scripts/predict_excel.py --excel "C:\Users\lizh2\Desktop\桌面\毕业论文\try\out_excel\all_blocks.xlsx" --checkpoint outputs/hmsan_bsa_lite/hmsan_bsa_lite.pt --out outputs/hmsan_bsa_lite/predictions.xlsx
```

Expected:

```text
analyze_labels prints dataset statistics
train_hmsan_bsa saves hmsan_bsa_lite.pt
predict_excel writes predictions.xlsx
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: describe hmsan bsa prototype workflow"
```

---

## 实施顺序建议

1. Task 1-3 先完成数据入口。
2. Task 4-6 完成特征、边界和 Dataset。
3. Task 7-8 完成模型 forward、loss、指标。
4. Task 9-11 完成脚本闭环。
5. Task 12 做最终验证与说明。

每个 task 独立提交一次。若某个 task 中测试失败，先修当前 task，不进入下一 task。

---

## 自检记录

Spec 覆盖情况：

- 数据层级 `Document -> Page -> Block`：Task 3、Task 6。
- 标签体系 `y_cls / y_boundary / y_section`：Task 2、Task 5、Task 6。
- HMSAN-BSA lite 模型接口：Task 7。
- 多任务损失：Task 8。
- 标签分析、训练、评估、预测导出：Task 9、Task 10、Task 11。
- 当前小数据限制说明：Task 12。

计划中没有留空的实现任务；full 模式的 RoBERTa、ResNet/ViT、Longformer、CRF 被明确列为 lite 骨架跑通后的升级方向，不属于第一版工程闭环。
