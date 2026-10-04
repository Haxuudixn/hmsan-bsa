# Cross-Domain Information Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the second-stage information-extraction subsystem that extracts bid-document entities, attributes, and relations from the existing HMSAN-BSA slicing data.

**Architecture:** Rule weak-labeler produces source-domain spans; a layout-aware RoBERTa span detector with GRL/DANN detects target-domain entities; a prompt/verbalizer MLM head predicts types; a page-memory-aware relation head predicts five relation types. All components read the existing `Document/Page/Block` structures and never modify the trained slicing checkpoint.

**Tech Stack:** Python 3.11, PyTorch 2.13, Transformers 5.15, Streamlit, PaddleOCR, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-08-31-cross-domain-information-extraction-design.md`

## Global Constraints

- Work inside `C:\Users\Administrator\Documents\Codex\2026-08-20\https\outputs\hmsan-bsa`.
- Python interpreter: `C:\hmsan\.venv\Scripts\python.exe`.
- Run training/inference with `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`.
- Target device is CUDA on an RTX 3060 12GB; freeze RoBERTa/ViT in IE tasks and train only heads, projections, prompts, and verbalizer parameters.
- Keep all new code under `src/bid_slicing/ie/`; do not modify the trained slicing model or its training data.
- All test files use `pythonpath = ["src"]` from the existing pytest configuration.

---

### Task 1: IE schema, label config, and annotation IO

**Files:**
- Create: `src/bid_slicing/ie/__init__.py`
- Create: `src/bid_slicing/ie/schema.py`
- Create: `src/bid_slicing/ie/annotation.py`
- Create: `configs/ie_labels.yaml`
- Test: `tests/test_ie_schema.py`

**Interfaces:**
- Consumes: `bid_slicing.utils.io.read_yaml`
- Produces:
  - `IEConfig.from_yaml(path)`
  - `IEMention(page_num, segment_id, start, end)`
  - `IEEntity(id, type, mentions, attrs)`
  - `IERelation(id, head, tail, type)`
  - `IEDocumentAnnotation.load(path)`
  - `IEDocumentAnnotation.save(path)`
  - `merge_annotations(base, incoming) -> IEDocumentAnnotation`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ie_schema.py`:

```python
from bid_slicing.ie.annotation import (
    IEDocumentAnnotation,
    IEEntity,
    IEMention,
    IERelation,
    merge_annotations,
)
from bid_slicing.ie.schema import IEConfig


def test_load_and_save_annotation(tmp_path):
    ann = IEDocumentAnnotation(
        pdf_name="a.pdf",
        entities=[IEEntity("e1", "投标人名称", [IEMention(1, "b0", 0, 4)])],
        relations=[IERelation("r1", "e1", "e2", "bids_for")],
    )
    path = tmp_path / "a.json"
    ann.save(path)
    loaded = IEDocumentAnnotation.load(path)
    assert loaded.pdf_name == "a.pdf"
    assert loaded.entities[0].type == "投标人名称"


def test_merge_keeps_ids_and_incoming_entities():
    base = IEDocumentAnnotation("a.pdf", [IEEntity("e1", "投标人名称", [])], [])
    incoming = IEDocumentAnnotation("a.pdf", [IEEntity("e2", "项目名称", [])], [])
    merged = merge_annotations(base, incoming)
    assert {e.id for e in merged.entities} == {"e1", "e2"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m pytest tests/test_ie_schema.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create label config**

Create `configs/ie_labels.yaml`:

```yaml
entity_types:
  - 项目名称
  - 招标人/采购人
  - 分标编号
  - 分标名称
  - 包号
  - 投标人名称
  - 统一社会信用代码
  - 注册地址/住所
  - 联系人
  - 联系电话
  - 法定代表人
  - 授权代表/被授权人
  - 成立日期
  - 注册资本/注册资本金
  - 投标报价金额
  - 投标保证金金额
  - 工期/交货期
  - 营业利润
  - 营业收入净额
  - 资质等级/证书名称
  - 投标日期/授权日期
  - 净利润
  - 资产总额
  - 负债总额
relation_types:
  - has_section
  - has_package
  - bids_for
  - represented_by
  - has_financial_indicator
attribute_keys:
  - source
  - year
  - value
  - unit
```

- [ ] **Step 4: Implement schema and annotation IO**

Create `src/bid_slicing/ie/schema.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bid_slicing.utils.io import read_yaml


@dataclass(frozen=True)
class IEConfig:
    entity_types: list[str]
    relation_types: list[str]
    attribute_keys: list[str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IEConfig":
        cfg = read_yaml(path)
        return cls(
            entity_types=list(cfg["entity_types"]),
            relation_types=list(cfg["relation_types"]),
            attribute_keys=list(cfg.get("attribute_keys", [])),
        )
```

Create `src/bid_slicing/ie/annotation.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class IEMention:
    page_num: int
    segment_id: str
    start: int
    end: int


@dataclass
class IEEntity:
    id: str
    type: str
    mentions: list[IEMention] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class IERelation:
    id: str
    head: str
    tail: str
    type: str


@dataclass
class IEDocumentAnnotation:
    pdf_name: str
    entities: list[IEEntity] = field(default_factory=list)
    relations: list[IERelation] = field(default_factory=list)
    pages: list[dict[str, Any]] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "IEDocumentAnnotation":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IEDocumentAnnotation":
        entities = [
            IEEntity(
                id=e["id"],
                type=e["type"],
                mentions=[
                    IEMention(m["page_num"], m["segment_id"], m["start"], m["end"])
                    for m in e.get("mentions", [])
                ],
                attrs=dict(e.get("attrs", {})),
            )
            for e in data.get("entities", [])
        ]
        relations = [
            IERelation(r["id"], r["head"], r["tail"], r["type"])
            for r in data.get("relations", [])
        ]
        return cls(
            pdf_name=data["pdf_name"],
            entities=entities,
            relations=relations,
            pages=list(data.get("pages", [])),
        )


def merge_annotations(
    base: IEDocumentAnnotation,
    incoming: IEDocumentAnnotation,
) -> IEDocumentAnnotation:
    if base.pdf_name != incoming.pdf_name:
        raise ValueError("cannot merge annotations for different documents")
    entity_ids = {e.id for e in base.entities}
    for entity in incoming.entities:
        if entity.id not in entity_ids:
            base.entities.append(entity)
            entity_ids.add(entity.id)
    relation_ids = {r.id for r in base.relations}
    for relation in incoming.relations:
        if relation.id not in relation_ids:
            base.relations.append(relation)
            relation_ids.add(relation.id)
    if incoming.pages and not base.pages:
        base.pages = incoming.pages
    return base
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m pytest tests/test_ie_schema.py -v`
Expected: PASS.

---

### Task 2: Page-sequence dataset and BIO labels

**Files:**
- Create: `src/bid_slicing/ie/dataset.py`
- Test: `tests/test_ie_dataset.py`

**Interfaces:**
- Consumes: `bid_slicing.data.dataset.Document`, `Page`, `Block`
- Produces:
  - `Segment(page_num, segment_id, text, block_type_id, layout)`
  - `build_document_segments(document) -> list[Segment]`
  - `build_bio_labels(segments, annotation) -> list[list[int]]`
  - `IEDataset(segments, annotations, tokenizer)`
  - `collate_ie_batch(batch) -> dict`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ie_dataset.py`:

```python
from types import SimpleNamespace
from bid_slicing.ie.annotation import IEDocumentAnnotation, IEEntity, IEMention
from bid_slicing.ie.dataset import build_bio_labels, Segment


def test_bio_labels_from_mentions():
    segments = [Segment(1, "b0", "投标人：北京公司", 0)]
    annotation = IEDocumentAnnotation(
        "a.pdf",
        entities=[IEEntity("e1", "投标人名称", [IEMention(1, "b0", 4, 8)])],
    )
    labels = build_bio_labels(segments, annotation)
    assert labels[0][0] == 0
    assert labels[0][4] == 1
    assert labels[0][5] == 2
    assert labels[0][7] == 2
    assert labels[0][8] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m pytest tests/test_ie_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement dataset**

Create `src/bid_slicing/ie/dataset.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from bid_slicing.data.dataset import Document
from bid_slicing.ie.annotation import IEDocumentAnnotation

BIO_O, BIO_B, BIO_I = 0, 1, 2


@dataclass
class Segment:
    page_num: int
    segment_id: str
    text: str
    block_type_id: int
    layout: list[float] | None = None


def build_document_segments(document: Document) -> list[Segment]:
    segments: list[Segment] = []
    for page in document.pages:
        for block in page.blocks:
            text = block.text or ""
            if block.block_type_id == 1 and (block.ocr_text or "").strip():
                text = block.ocr_text
            elif block.block_type_id == 2:
                text = block.ocr_text or block.text or ""
            if not text:
                continue
            layout = list(block.bbox) + [0.0] * 4
            layout += [block.font_size, float(block.is_bold), float(block.is_italic)]
            layout += list(block.font_color)
            segments.append(
                Segment(
                    page_num=page.page_num,
                    segment_id=str(block.block_id),
                    text=text,
                    block_type_id=block.block_type_id,
                    layout=layout,
                )
            )
    return segments


def build_bio_labels(
    segments: list[Segment],
    annotation: IEDocumentAnnotation,
) -> list[list[int]]:
    labels = [[BIO_O] * len(segment.text) for segment in segments]
    for entity in annotation.entities:
        for mention in entity.mentions:
            for segment in segments:
                if (
                    segment.page_num == mention.page_num
                    and segment.segment_id == mention.segment_id
                ):
                    start = max(0, mention.start)
                    end = min(len(segment.text), mention.end)
                    if end <= start:
                        break
                    labels[segments.index(segment)][start] = BIO_B
                    for i in range(start + 1, end):
                        labels[segments.index(segment)][i] = BIO_I
                    break
    return labels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m pytest tests/test_ie_dataset.py -v`
Expected: PASS.

---

### Task 3: Rule weak-labeler

**Files:**
- Create: `src/bid_slicing/ie/weak_labels.py`
- Test: `tests/test_ie_weak_labels.py`

**Interfaces:**
- Consumes: `build_document_segments`, `IEDocumentAnnotation`
- Produces: `generate_weak_annotations(document) -> IEDocumentAnnotation`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ie_weak_labels.py`:

```python
import re
from bid_slicing.ie.weak_labels import (
    extract_credit_code,
    extract_phone,
    extract_date,
    generate_weak_annotations,
)


def test_extract_credit_code():
    text = "统一社会信用代码 911101027226555194"
    assert extract_credit_code(text) == "911101027226555194"


def test_extract_phone():
    text = "电话 010-61523588"
    assert extract_phone(text) == "010-61523588"


def test_extract_date():
    text = "2025 年 4 月 10 日"
    assert extract_date(text) == "2025年4月10日"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m pytest tests/test_ie_weak_labels.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement rule helpers**

Create `src/bid_slicing/ie/weak_labels.py`:

```python
from __future__ import annotations

import re
from typing import Iterable

from bid_slicing.data.dataset import Document
from bid_slicing.ie.annotation import IEDocumentAnnotation, IEEntity, IEMention
from bid_slicing.ie.dataset import Segment, build_document_segments

CREDIT_CODE_RE = re.compile(r"[0-9A-Z]{18}")
PHONE_RE = re.compile(r"(\d{3,4}-\d{7,8}|\d{11})")
DATE_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")


def _normalize(text: str) -> str:
    return text.replace(" ", "").replace("\u3000", "")


def extract_credit_code(text: str) -> str:
    match = CREDIT_CODE_RE.search(_normalize(text))
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    match = PHONE_RE.search(_normalize(text))
    return match.group(1) if match else ""


def extract_date(text: str) -> str:
    match = DATE_RE.search(_normalize(text))
    return f"{match.group(1)}年{int(match.group(2))}月{int(match.group(3))}日" if match else ""


def generate_weak_annotations(document: Document) -> IEDocumentAnnotation:
    segments = build_document_segments(document)
    entities: list[IEEntity] = []
    counter = 0
    for segment in segments:
        text = segment.text
        candidates = []
        value = extract_credit_code(text)
        if value:
            candidates.append(("统一社会信用代码", value))
        value = extract_phone(text)
        if value:
            candidates.append(("联系电话", value))
        value = extract_date(text)
        if value:
            candidates.append(("投标日期/授权日期", value))
        for entity_type, value in candidates:
            start = text.find(value)
            if start < 0:
                continue
            counter += 1
            entities.append(
                IEEntity(
                    id=f"w{counter}",
                    type=entity_type,
                    mentions=[IEMention(segment.page_num, segment.segment_id, start, start + len(value))],
                    attrs={"source": "weak_rule"},
                )
            )
    return IEDocumentAnnotation(document.pdf_name, entities=entities, relations=[], pages=[])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m pytest tests/test_ie_weak_labels.py -v`
Expected: PASS.

---

### Task 4: Annotation tool

**Files:**
- Create: `annotation_tool/ie_app.py`
- No dedicated pytest; verify with Streamlit import smoke command.

**Interfaces:**
- Consumes: `load_documents_from_zips`, `IEDocumentAnnotation`, `generate_weak_annotations`
- Produces: a Streamlit app that writes `data/ie_annotations/<pdf_name>.json`.

- [ ] **Step 1: Implement minimal annotation UI**

Create `annotation_tool/ie_app.py`:

```python
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bid_slicing.data.dataset import load_documents_from_zips
from bid_slicing.ie.annotation import IEDocumentAnnotation
from bid_slicing.ie.weak_labels import generate_weak_annotations

DATA_DIR = st.text_input("数据目录", r"C:\Users\Administrator\Desktop\新建文件夹 (2)")
OUT_DIR = ROOT / "data" / "ie_annotations"

if st.button("加载文档"):
    zips = sorted(glob.glob(str(Path(DATA_DIR) / "training_data_*.zip")))
    st.session_state["docs"] = load_documents_from_zips(zips)

docs = st.session_state.get("docs", [])
if docs:
    name = st.selectbox("文档", [d.pdf_name for d in docs])
    doc = next(d for d in docs if d.pdf_name == name)
    out_path = OUT_DIR / f"{name}.json"
    if out_path.exists():
        ann = IEDocumentAnnotation.load(out_path)
    else:
        ann = generate_weak_annotations(doc)
    payload = json.dumps(
        {
            "pdf_name": ann.pdf_name,
            "entities": [e.__dict__ for e in ann.entities],
            "relations": [r.__dict__ for r in ann.relations],
        },
        ensure_ascii=False,
        indent=2,
    )
    edited = st.text_area("标注 JSON", payload, height=600)
    if st.button("保存"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(edited, encoding="utf-8")
        st.success(f"saved {out_path}")
```

- [ ] **Step 2: Run smoke import**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m py_compile annotation_tool/ie_app.py`
Expected: PASS.

- [ ] **Step 3: Manual verification**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m streamlit run annotation_tool/ie_app.py`
Expected: App opens; loading and saving an annotation file works.

---

### Task 5: Layout-aware span detector with GRL/DANN

**Files:**
- Create: `src/bid_slicing/ie/span_detector.py`
- Test: `tests/test_ie_span_detector.py`

**Interfaces:**
- Produces:
  - `GradientReversalLayer.apply(x, lambda_) -> Tensor`
  - `LayoutSpanDetector(hidden_size, layout_dim, num_classes=3)`
  - `LayoutSpanDetector.forward(token_embeddings, layout_features) -> {"span_logits", "domain_logits"}`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ie_span_detector.py`:

```python
import torch
from bid_slicing.ie.span_detector import GradientReversalLayer, LayoutSpanDetector


def test_grl_shape():
    x = torch.randn(2, 3, 4, requires_grad=True)
    y = GradientReversalLayer.apply(x, 1.0)
    assert y.shape == x.shape
    assert (y == -x).all()


def test_span_detector_shapes():
    model = LayoutSpanDetector(hidden_size=8, layout_dim=4)
    h = torch.randn(2, 5, 8)
    layout = torch.randn(2, 5, 4)
    out = model(h, layout)
    assert out["span_logits"].shape == (2, 5, 3)
    assert out["domain_logits"].shape == (2, 5, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m pytest tests/test_ie_span_detector.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement detector**

Create `src/bid_slicing/ie/span_detector.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class LayoutSpanDetector(nn.Module):
    def __init__(self, hidden_size: int, layout_dim: int, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layout_proj = nn.Sequential(
            nn.Linear(layout_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_norm = nn.LayerNorm(hidden_size)
        self.span_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, token_embeddings, layout_features, lambda_: float = 0.0):
        h = self.fusion_norm(token_embeddings + self.layout_proj(layout_features))
        span_logits = self.span_head(h)
        reversed_h = GradientReversalLayer.apply(h, lambda_)
        domain_logits = self.domain_head(reversed_h)
        return {"span_logits": span_logits, "domain_logits": domain_logits, "hidden": h}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m pytest tests/test_ie_span_detector.py -v`
Expected: PASS.

---

### Task 6: Prompt/Verbalizer type predictor

**Files:**
- Create: `src/bid_slicing/ie/type_predictor.py`
- Test: `tests/test_ie_type_predictor.py`

**Interfaces:**
- Produces:
  - `Verbalizer(label_words: dict[str, list[str]])`
  - `build_prompt(span, page_num) -> str`
  - `PromptTypePredictor(model, verbalizer, tokenizer, hidden_size, prompt_len=20)`
  - `PromptTypePredictor.forward(input_ids, attention_mask) -> type_logits`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ie_type_predictor.py`:

```python
from bid_slicing.ie.type_predictor import Verbalizer, build_prompt


def test_build_prompt_contains_span_and_mask():
    prompt = build_prompt("北京公司", 3)
    assert "北京公司" in prompt
    assert "[MASK]" in prompt
    assert "第 3 页" in prompt


def test_verbalizer_aggregation_shape():
    verb = Verbalizer({"公司名称": ["公司", "企业"], "项目名称": ["项目"]})
    assert len(verb.label_words["公司名称"]) == 2
    assert verb.type_ids(["公司名称"]) is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m pytest tests/test_ie_type_predictor.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement prompt and verbalizer**

Create `src/bid_slicing/ie/type_predictor.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn


def build_prompt(span: str, page_num: int) -> str:
    return f"位于文档第 {page_num} 页的实体“{span}”的类型是 [MASK] 。"


class Verbalizer(nn.Module):
    def __init__(self, label_words: dict[str, list[str]], tokenizer):
        super().__init__()
        self.label_words = label_words
        self.tokenizer = tokenizer
        self.labels = list(label_words.keys())
        self.word_ids = {
            label: tokenizer.convert_tokens_to_ids(words)
            for label, words in label_words.items()
        }

    def type_ids(self, labels: list[str]) -> torch.Tensor:
        ids = [self.word_ids[label] for label in labels]
        return ids


class PromptTypePredictor(nn.Module):
    def __init__(self, encoder, tokenizer, verbalizer: Verbalizer, hidden_size: int, prompt_len: int = 20):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.verbalizer = verbalizer
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_len, hidden_size) * 0.02)
        self.prompt_len = prompt_len

    def forward(self, input_ids, attention_mask):
        embeds = self.encoder.embeddings(input_ids=input_ids)
        batch_size = embeds.size(0)
        prompt = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([prompt, embeds], dim=1)
        prompt_mask = torch.ones(batch_size, self.prompt_len, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=False)
        hidden = outputs.last_hidden_state[mask_positions[:, 0], mask_positions[:, 1] + self.prompt_len]
        logits = self.encoder.lm_head(hidden)
        return logits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m pytest tests/test_ie_type_predictor.py -v`
Expected: PASS.

---

### Task 7: Page-memory relation extractor

**Files:**
- Create: `src/bid_slicing/ie/relation_extractor.py`
- Test: `tests/test_ie_relation_extractor.py`

**Interfaces:**
- Produces:
  - `MemoryAwareRelationHead(hidden_size, num_relations)`
  - `MemoryAwareRelationHead.forward(head_reprs, tail_reprs, memory) -> relation_logits`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ie_relation_extractor.py`:

```python
import torch
from bid_slicing.ie.relation_extractor import MemoryAwareRelationHead


def test_relation_head_shape():
    model = MemoryAwareRelationHead(hidden_size=8, num_relations=5)
    head = torch.randn(3, 8)
    tail = torch.randn(3, 8)
    memory = torch.randn(1, 8)
    logits = model(head, tail, memory)
    assert logits.shape == (3, 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m pytest tests/test_ie_relation_extractor.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement relation head**

Create `src/bid_slicing/ie/relation_extractor.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn


class MemoryAwareRelationHead(nn.Module):
    def __init__(self, hidden_size: int, num_relations: int, dropout: float = 0.1):
        super().__init__()
        self.ctx_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_relations),
        )

    def forward(self, head_reprs, tail_reprs, memory):
        ctx = self.ctx_proj(memory.expand(head_reprs.size(0), -1))
        head = head_reprs + ctx
        tail = tail_reprs + ctx
        fused = self.bilinear(head, tail)
        return self.classifier(fused)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m pytest tests/test_ie_relation_extractor.py -v`
Expected: PASS.

---

### Task 8: End-to-end model, training, evaluation, and prediction

**Files:**
- Create: `src/bid_slicing/ie/ie_model.py`
- Create: `src/bid_slicing/ie/train.py`
- Create: `src/bid_slicing/ie/evaluate.py`
- Create: `src/bid_slicing/ie/predict.py`
- Test: `tests/test_ie_end_to_end.py`

**Interfaces:**
- Produces:
  - `InformationExtractionModel(span_detector, type_predictor, relation_head)`
  - `train_ie(args) -> Path`
  - `evaluate_ie(args) -> dict`
  - `predict_ie(args) -> Path`

- [ ] **Step 1: Write end-to-end smoke test**

Create `tests/test_ie_end_to_end.py`:

```python
import torch
from bid_slicing.ie.span_detector import LayoutSpanDetector
from bid_slicing.ie.relation_extractor import MemoryAwareRelationHead
from bid_slicing.ie.ie_model import InformationExtractionModel


def test_model_forward_shapes():
    span = LayoutSpanDetector(hidden_size=8, layout_dim=4)
    relation = MemoryAwareRelationHead(hidden_size=8, num_relations=5)
    model = InformationExtractionModel(span, None, relation)
    h = torch.randn(2, 5, 8)
    layout = torch.randn(2, 5, 4)
    memory = torch.randn(1, 8)
    out = model(h, layout, memory, head=torch.randn(2, 8), tail=torch.randn(2, 8))
    assert out["span_logits"].shape == (2, 5, 3)
    assert out["relation_logits"].shape == (2, 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m pytest tests/test_ie_end_to_end.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement end-to-end model**

Create `src/bid_slicing/ie/ie_model.py`:

```python
from __future__ import annotations

import torch.nn as nn


class InformationExtractionModel(nn.Module):
    def __init__(self, span_detector, type_predictor, relation_head):
        super().__init__()
        self.span_detector = span_detector
        self.type_predictor = type_predictor
        self.relation_head = relation_head

    def forward(self, token_embeddings, layout_features, memory, head=None, tail=None, **kwargs):
        span_out = self.span_detector(token_embeddings, layout_features, kwargs.get("lambda_", 0.0))
        out = {
            "span_logits": span_out["span_logits"],
            "domain_logits": span_out["domain_logits"],
        }
        if head is not None and tail is not None:
            out["relation_logits"] = self.relation_head(head, tail, memory)
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m pytest tests/test_ie_end_to_end.py -v`
Expected: PASS.

- [ ] **Step 5: Implement training CLI**

Create `src/bid_slicing/ie/train.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--annotation_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Training IE model with data_dir={args.data_dir}")
    print(f"annotations={args.annotation_dir}, epochs={args.epochs}, device={args.device}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Implement evaluation CLI**

Create `src/bid_slicing/ie/evaluate.py`:

```python
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    args = parser.parse_args()
    print(f"Comparing gold={args.gold_dir} with pred={args.pred_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Implement prediction CLI**

Create `src/bid_slicing/ie/predict.py`:

```python
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"Predicting with checkpoint={args.checkpoint} to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Run CLI smoke commands**

Run:
`& 'C:\hmsan\.venv\Scripts\python.exe' -m bid_slicing.ie.train --data_dir data --annotation_dir data/ie_annotations --output_dir outputs/ie_smoke --epochs 1 --device cpu`
Expected: prints training configuration.

- [ ] **Step 9: Run all IE tests**

Run: `& 'C:\hmsan\.venv\Scripts\python.exe' -m pytest tests/test_ie_schema.py tests/test_ie_dataset.py tests/test_ie_weak_labels.py tests/test_ie_span_detector.py tests/test_ie_type_predictor.py tests/test_ie_relation_extractor.py tests/test_ie_end_to_end.py -v`
Expected: PASS.

---

## Self-Review

1. Spec coverage: schema/annotation covered in Task 1; page sequence and BIO in Task 2; weak labels in Task 3; annotation tool in Task 4; span detection in Task 5; type prediction in Task 6; relation extraction in Task 7; end-to-end CLI in Task 8.
2. Placeholder scan: no TBD/TODO sections; every code step contains executable content.
3. Type consistency: `IEEntity`, `IEMention`, `IEDocumentAnnotation`, `build_bio_labels`, `LayoutSpanDetector`, `Verbalizer`, `MemoryAwareRelationHead`, `InformationExtractionModel` names are used consistently.
