"""Path-3 quantitative baselines B1-B3 (plan decision D4).

Why these exist
---------------
The ablation matrix only ever removes parts of HMSAN-BSA.  It can say which
component matters, but not whether the assembled architecture beats a plain
model.  Paper table 3 currently carries no numbers at all, only a capability
checklist.  B1-B3 fill that gap:

  B1  RoBERTa (fine-tuned end to end) + per-block MLP.
      No page hierarchy, no layout features, no image.
  B2  B1 plus explicit layout features (page-normalised bbox, font, colour).
  B3  B1 plus a per-page BiGRU over the block sequence: in-page order is
      modelled, page hierarchy and cross-page memory are not.
  B4  free -- it is the existing ``A10_dense`` matrix row, not built here.

Fairness rules (docs/path3_execution_plan_2026-09-21.md, W6-W8)
---------------------------------------------------------------
* The text encoder is fine-tuned end to end.  Reusing frozen cached features
  would compare a frozen baseline against an unfrozen main model and invite a
  straw-man objection.
* Corpus loader, grouped split and metric code are the ones ``train.py`` uses,
  so the numbers are directly comparable.  The split is asserted against the
  frozen manifest (train=306, val=66, test=62) and the run aborts if it
  differs -- a silently different split would invalidate the comparison.

Usage
-----
    python scripts/paper/baselines.py --baseline b1 --output_dir outputs/path3_baselines/b1

Add ``--limit_docs 4 --epochs 1`` for a CPU-sized smoke run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from bid_slicing.data.dataset import (  # noqa: E402
    IGNORE_INDEX,
    BidDocumentDataset,
    load_documents_from_zips,
)
from bid_slicing.data.label_schema import LabelSchema  # noqa: E402
from bid_slicing.features.text_features import RoBERTaTextEncoder  # noqa: E402
from bid_slicing.models.block_encoder import BlockType  # noqa: E402

# train.py owns the metric definitions; importing it keeps B1-B3 on exactly the
# same footing as the main model (accuracy, macro F1, per-class F1/support).
_spec = importlib.util.spec_from_file_location("hmsan_train", REPO / "train.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

EXPECTED_SPLIT = (306, 66, 62)
BASELINES = ("b1", "b2", "b3")


def parse_args():
    p = argparse.ArgumentParser(description="Path-3 baselines B1-B3")
    p.add_argument("--baseline", type=str, required=True, choices=BASELINES)
    p.add_argument("--data_dir", type=str, default=r"C:\Users\Administrator\Desktop\训练数据\final_train")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--label_config", type=str, default="configs/labels.yaml")
    p.add_argument("--encoder_name", type=str, default="hfl/chinese-roberta-wwm-ext")
    p.add_argument("--text_max_length", type=int, default=510)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--encoder_lr", type=float, default=2e-5)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--device", type=str, default="cuda")
    # Mirrors train.py: the caps decide how long documents are segmented, and
    # the split assigns whole segment groups, so a different cap silently
    # produces a different split.
    p.add_argument("--max_blocks_per_sample", type=int, default=3000)
    p.add_argument("--max_pages_per_sample", type=int, default=150)
    p.add_argument("--batch_blocks", type=int, default=32,
                   help="B1/B2: blocks per optimizer step.")
    p.add_argument("--batch_pages", type=int, default=8,
                   help="B3: pages per optimizer step (gradients accumulate).")
    p.add_argument("--max_blocks_per_page", type=int, default=48,
                   help="B3: cap on the in-page sequence length.")
    p.add_argument("--encode_chunk", type=int, default=16,
                   help="B3: blocks encoded per RoBERTa call, bounds activation memory.")
    p.add_argument("--limit_docs", type=int, default=0,
                   help="Smoke runs: keep only the first N documents of each split "
                        "(applied after the split, so the split itself is unchanged).")
    # Mirrors train.py's flag of the same name.  The default stays "none" so
    # every B1-B3 number recorded before 2026-09-30 keeps reproducing, and
    # aligning the baselines with the main model's "inverse" is an explicit
    # choice rather than a silent change.
    p.add_argument("--class_weight_mode", type=str, default="none",
                   choices=("none", "inverse"),
                   help="Class weighting for the training loss (same rule as train.py).")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to this row's last.pt; continues at the epoch after "
                        "the one recorded in it.")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def block_text(block) -> str:
    """The text the main model feeds to RoBERTa for this block.

    Text blocks keep their content in ``text``; image and mixed blocks keep the
    ZIP member path there and their real text in ``ocr_text`` (see
    ``Block.block_type_id``).  Falling back to ``text`` for those hands the
    baseline a filename such as ``blocks/block_0222.png`` instead of a token
    sequence -- and that filename carries the block index, so it is memorisable
    rather than merely useless.
    """
    if block.block_type_id == BlockType.TEXT:
        return (block.text or "").strip()
    return (block.ocr_text or "").strip()


def page_extent(page) -> tuple[float, float]:
    """Page width/height proxy: the furthest block corner on the page."""
    width = max((b.bbox[2] for b in page.blocks), default=1.0)
    height = max((b.bbox[3] for b in page.blocks), default=1.0)
    return max(width, 1.0), max(height, 1.0)


def layout_vector(block, extent) -> list[float]:
    """Layout features for B2.

    Eight bbox terms in the same [x0,y0,x1,y1,w,h,cx,cy] order the main model's
    LayoutEncoder consumes, then font size (scaled), bold/italic flags and the
    RGB colour.  Everything is page-normalised so the vector is scale-free.
    """
    x0, y0, x1, y1 = block.bbox
    width, height = max(x1 - x0, 0.0), max(y1 - y0, 0.0)
    page_w, page_h = extent
    return [
        x0 / page_w, y0 / page_h, x1 / page_w, y1 / page_h,
        width / page_w, height / page_h,
        (x0 + x1) / 2 / page_w, (y0 + y1) / 2 / page_h,
        min(block.font_size / 24.0, 2.0),
        float(block.is_bold),
        float(block.is_italic),
        block.font_color[0] / 255.0,
        block.font_color[1] / 255.0,
        block.font_color[2] / 255.0,
    ]


LAYOUT_DIM = 14


class BlockDataset(Dataset):
    """Labelled blocks of a split, each an independent example (B1/B2)."""

    def __init__(self, documents, use_layout: bool):
        self.use_layout = use_layout
        self.items: list[tuple] = []
        for doc in documents:
            for page in doc.pages:
                extent = page_extent(page) if use_layout else None
                for block in page.blocks:
                    label = block.label_id
                    if label == IGNORE_INDEX:
                        continue
                    layout = layout_vector(block, extent) if use_layout else None
                    self.items.append((block_text(block), label, layout))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple:
        return self.items[index]


def collate_blocks(items):
    texts = [item[0] for item in items]
    labels = torch.tensor([item[1] for item in items], dtype=torch.long)
    if items[0][2] is None:
        return texts, None, labels
    layout = torch.tensor([item[2] for item in items], dtype=torch.float)
    return texts, layout, labels


class PageDataset(Dataset):
    """Pages of a split, keeping the in-page block order (B3)."""

    def __init__(self, documents, max_blocks_per_page: int):
        self.pages: list[tuple[list[str], list[int]]] = []
        for doc in documents:
            for page in doc.pages:
                blocks = page.blocks[:max_blocks_per_page]
                labels = [b.label_id for b in blocks]
                if not blocks or all(l == IGNORE_INDEX for l in labels):
                    continue
                self.pages.append(([block_text(b) for b in blocks], labels))

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> tuple:
        return self.pages[index]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class BlockClassifier(nn.Module):
    """B1 (layout_dim=0) and B2: RoBERTa -> optional layout concat -> MLP."""

    def __init__(self, encoder_name, max_length, num_classes,
                 layout_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.text_encoder = RoBERTaTextEncoder(
            model_name=encoder_name, max_length=max_length, freeze=False
        )
        hidden = 256
        self.head = nn.Sequential(
            nn.Linear(self.text_encoder.output_dim + layout_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, texts, layout=None):
        features = self.text_encoder(texts)
        if layout is not None:
            features = torch.cat([features, layout], dim=-1)
        return self.head(features)


class PageSequenceClassifier(nn.Module):
    """B3: RoBERTa per block, then a BiGRU over the in-page block order."""

    def __init__(self, encoder_name, max_length, num_classes,
                 gru_hidden: int = 128, dropout: float = 0.1,
                 encode_chunk: int = 16):
        super().__init__()
        self.text_encoder = RoBERTaTextEncoder(
            model_name=encoder_name, max_length=max_length, freeze=False
        )
        self.encode_chunk = encode_chunk
        self.gru = nn.GRU(
            self.text_encoder.output_dim, gru_hidden,
            batch_first=True, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * gru_hidden, num_classes),
        )

    def encode_page(self, texts) -> torch.Tensor:
        """Embed a page's blocks, chunked so activation memory stays bounded."""
        chunks = [
            self.text_encoder(texts[i:i + self.encode_chunk])
            for i in range(0, len(texts), self.encode_chunk)
        ]
        return torch.cat(chunks, dim=0)

    def forward_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(embeddings.unsqueeze(0))
        return self.head(sequence.squeeze(0))


# --------------------------------------------------------------------------
# train / eval
# --------------------------------------------------------------------------
def mask_metrics(preds, targets, num_classes):
    """compute_metrics ignores IGNORE_INDEX targets internally."""
    return train.compute_metrics(
        torch.as_tensor(preds), torch.as_tensor(targets), num_classes=num_classes
    )


@torch.no_grad()
def evaluate_blocks(model, loader, device, num_classes, use_amp):
    model.eval()
    preds, targets = [], []
    for texts, layout, labels in loader:
        if layout is not None:
            layout = layout.to(device, non_blocking=True)
        with train._autocast(device, use_amp):
            logits = model(texts, layout)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        targets.extend(labels.tolist())
    return mask_metrics(preds, targets, num_classes)


@torch.no_grad()
def evaluate_pages(model, loader, device, num_classes, use_amp):
    model.eval()
    preds, targets = [], []
    # ``collate_fn`` hands back the raw list of pages, so one loader step is a
    # batch of (texts, labels) pairs, not a single pair.  Unpacking one level
    # shallower passed a tuple straight to the tokenizer.
    for batch in loader:
        for texts, labels in batch:
            with train._autocast(device, use_amp):
                logits = model.forward_embeddings(model.encode_page(texts))
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(labels)
    return mask_metrics(preds, targets, num_classes)


def is_ignored(labels) -> list[bool]:
    if torch.is_tensor(labels):
        return (labels == IGNORE_INDEX).tolist()
    return [l == IGNORE_INDEX for l in labels]


def make_criterion(class_weights, device):
    """CrossEntropyLoss with optional class weights.

    train.py builds the weights in fp32 and casts them to the logit dtype,
    because bf16 autocast plus fp32 weights makes cross_entropy raise.  Here the
    logits are forced to fp32 before the loss, so the dtypes agree by
    construction; the cast is kept anyway so the two paths cannot drift apart.
    """
    if class_weights is None:
        return nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    return nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX,
                               weight=class_weights.to(device))


def train_one_epoch_blocks(model, loader, optimizer, device, num_classes,
                           use_amp, grad_clip, class_weights=None):
    model.train()
    total_loss, steps = 0.0, 0
    criterion = make_criterion(class_weights, device)
    for texts, layout, labels in loader:
        labels = labels.to(device, non_blocking=True)
        if layout is not None:
            layout = layout.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with train._autocast(device, use_amp):
            logits = model(texts, layout)
            loss = criterion(logits.float(), labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.detach())
        steps += 1
    return total_loss / max(steps, 1), steps


def train_one_epoch_pages(model, loader, optimizer, device, num_classes,
                          use_amp, grad_clip, class_weights=None):
    """Gradient accumulation over ``batch_pages`` pages; one step per batch."""
    model.train()
    criterion = make_criterion(class_weights, device)
    total_loss, steps = 0.0, 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for texts, labels in batch:
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)
            with train._autocast(device, use_amp):
                logits = model.forward_embeddings(model.encode_page(texts))
                loss = criterion(logits.float(), labels_t) / len(batch)
            loss.backward()
            batch_loss += float(loss.detach())
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += batch_loss
        steps += 1
    return total_loss / max(steps, 1), steps


def build_optimizer(model, encoder_lr, head_lr, weight_decay):
    encoder_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
    encoder_ids = {id(p) for p in encoder_params}
    head_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def main():
    args = parse_args()
    train.set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp) and device.type == "cuda"
    print(f"Baseline {args.baseline.upper()} | device={device} | amp={use_amp}")

    label_schema = LabelSchema.from_yaml(args.label_config)
    num_classes = label_schema.num_classes

    data_dir = Path(args.data_dir)
    zip_paths = sorted(str(z) for z in data_dir.glob("training_data_*.zip"))
    if not zip_paths:
        raise SystemExit(f"no training_data_*.zip under {data_dir}")
    print(f"Found {len(zip_paths)} ZIP file(s)")

    documents = load_documents_from_zips(
        zip_paths, args.max_blocks_per_sample, args.max_pages_per_sample
    )
    train_docs, val_docs, test_docs = BidDocumentDataset(documents).split_by_documents(
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    split = (len(train_docs), len(val_docs), len(test_docs))
    print(f"Split: train={split[0]}, val={split[1]}, test={split[2]}")
    if args.limit_docs <= 0 and split != EXPECTED_SPLIT:
        raise SystemExit(
            f"split {split} != frozen manifest {EXPECTED_SPLIT}; the baseline would "
            f"not be comparable with the main model. Refusing to train."
        )
    if args.limit_docs > 0:
        train_docs = train_docs.documents[:args.limit_docs]
        val_docs = val_docs.documents[:args.limit_docs]
        print(f"[smoke] limit_docs={args.limit_docs} -> "
              f"train={len(train_docs)} val={len(val_docs)}")

    use_layout = args.baseline == "b2"

    if args.baseline in ("b1", "b2"):
        model = BlockClassifier(
            args.encoder_name, args.text_max_length, num_classes,
            layout_dim=LAYOUT_DIM if use_layout else 0, dropout=args.dropout,
        ).to(device)
        train_loader = DataLoader(
            BlockDataset(train_docs, use_layout), batch_size=args.batch_blocks,
            shuffle=True, collate_fn=collate_blocks, num_workers=0,
        )
        val_loader = DataLoader(
            BlockDataset(val_docs, use_layout), batch_size=args.batch_blocks,
            shuffle=False, collate_fn=collate_blocks, num_workers=0,
        )
        train_epoch, evaluate = train_one_epoch_blocks, evaluate_blocks
    else:
        model = PageSequenceClassifier(
            args.encoder_name, args.text_max_length, num_classes,
            dropout=args.dropout, encode_chunk=args.encode_chunk,
        ).to(device)
        train_loader = DataLoader(
            PageDataset(train_docs, args.max_blocks_per_page),
            batch_size=args.batch_pages, shuffle=True,
            collate_fn=lambda batch: batch, num_workers=0,
        )
        val_loader = DataLoader(
            PageDataset(val_docs, args.max_blocks_per_page),
            batch_size=args.batch_pages, shuffle=False,
            collate_fn=lambda batch: batch, num_workers=0,
        )
        train_epoch, evaluate = train_one_epoch_pages, evaluate_pages

    print(f"train examples={len(train_loader.dataset)}  "
          f"val examples={len(val_loader.dataset)}")
    if len(train_loader.dataset) == 0:
        raise SystemExit("empty training set")

    optimizer = build_optimizer(model, args.encoder_lr, args.head_lr,
                               args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    # Same function the main model calls, so "inverse" means the identical
    # weighting here and in train.py.
    class_weights = train.compute_class_weights(
        train_docs, num_classes, args.class_weight_mode
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    log_path = output_dir / "train.log"
    log = open(log_path, "a", encoding="utf-8")

    def emit(message: str) -> None:
        print(message, flush=True)
        log.write(message + "\n")
        log.flush()

    emit(f"[run] {time.strftime('%Y-%m-%d %H:%M:%S')} baseline={args.baseline} "
         f"epochs={args.epochs} seed={args.seed} device={device}")
    emit(f"[run] split train={split[0]} val={split[1]} test={split[2]} "
         f"encoder={args.encoder_name} (fine-tuned end to end)")
    emit(f"[run] class_weight_mode={args.class_weight_mode}"
         + ("" if class_weights is None else
            " weights=" + ",".join(f"{w:.3f}" for w in class_weights.tolist())))

    best = -1.0
    start_epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        if state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        best = float(state.get("val_accuracy") or -1.0)
        start_epoch = int(state.get("epoch", 0)) + 1
        carried = output_dir / "history.json"
        if carried.is_file():
            history = json.loads(carried.read_text(encoding="utf-8"))
        emit(f"[resume] {args.resume} -> starting at epoch {start_epoch} "
             f"(best={best:.4f}, carried {len(history)} epoch(s))")
        if start_epoch > args.epochs:
            emit("[resume] nothing left to run")
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        loss, steps = train_epoch(
            model, train_loader, optimizer, device, num_classes, use_amp,
            args.grad_clip, class_weights=class_weights,
        )
        metrics = evaluate(model, val_loader, device, num_classes, use_amp)
        scheduler.step(metrics["accuracy"])
        wall = time.time() - epoch_started
        peak_mb = (
            torch.cuda.max_memory_allocated() / 2**20
            if device.type == "cuda" else 0.0
        )
        history.append({
            "epoch": epoch,
            "phase": "epoch",
            "loss": loss,
            "steps": steps,
            "wall_seconds": round(wall, 1),
            "peak_vram_mb": round(peak_mb, 1),
            "val": train._jsonable(metrics),
        })
        emit(f"  [ep{epoch}] loss={loss:.4f} acc={metrics['accuracy']:.4f} "
             f"macro_f1={metrics.get('macro_f1', 0.0):.4f} "
             f"wall={wall/60:.1f}min peak_vram={peak_mb/1024:.2f}GB")
        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        if metrics["accuracy"] > best:
            best = metrics["accuracy"]
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_accuracy": best}, output_dir / "best_model.pt")
            emit(f"  -> best so far (acc={best:.4f})")
        # last.pt carries the optimizer and scheduler too, so --resume picks the
        # trajectory back up instead of restarting Adam's moments.
        torch.save({"epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": best,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()},
                   output_dir / "last.pt")

    last = history[-1]
    result = {
        "baseline": args.baseline,
        "status": "completed",
        "argv": sys.argv[1:],
        "split": {"train": split[0], "val": split[1], "test": split[2]},
        "epochs": args.epochs,
        "class_weight_mode": args.class_weight_mode,
        "resumed_from": args.resume or "",
        "best_val_accuracy": best,
        "best_val_macro_f1": max(e["val"]["macro_f1"] for e in history),
        "final_val_accuracy": last["val"]["accuracy"],
        "final_val_macro_f1": last["val"]["macro_f1"],
        "total_wall_seconds": round(time.time() - started, 1),
    }
    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    emit(f"[run] {time.strftime('%Y-%m-%d %H:%M:%S')} done "
         f"({result['total_wall_seconds']/60:.1f} min)")
    log.close()


if __name__ == "__main__":
    main()
