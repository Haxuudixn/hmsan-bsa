"""F13 checks: mid-epoch checkpoint cadence and resume step accounting.

The 2026-09-25 23:25 stage-1 crash (exit=-1073741819) killed the process at
step 85/306 of epoch 1 and, because ``last.pt`` was only written at epoch
boundaries, threw away the whole epoch. F13 writes ``last.pt`` and
``history.json`` every N steps and lets ``--resume`` re-enter the interrupted
epoch, skipping the batches that were already trained.

These checks run on CPU against a stub model, so they cost no GPU time. They
cover the part that is new and easy to get wrong: the step accounting when a
resumed epoch skips a prefix of batches.

Run:  python scripts/verify_ckpt_resume.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location("hmsan_train", REPO / "train.py")
train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train)

NUM_CLASSES = 5
NUM_BOUNDARIES = 2
BLOCKS_PER_PAGE = 3
PAGES = 2
DOCS = 10
DEVICE = torch.device("cpu")

failures: list[str] = []


def check(name: str, condition: bool, detail: object = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  ({detail})")
        failures.append(name)


class StubModel(nn.Module):
    """Emits exactly the three logit tensors train_epoch expects, per page."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(1, NUM_CLASSES)

    def forward(self, *, texts, ocr_texts, image_blocks, block_type_ids,
                bbox_norm, font_size, is_bold, is_italic, font_color,
                page_splits):
        seed = torch.zeros(1, 1, device=block_type_ids.device)
        block_logits, boundary_logits, section_logits = [], [], []
        prev = 0
        for end in page_splits:
            n_blocks = int(end) - prev
            prev = int(end)
            block_logits.append(
                self.head(seed).view(1, 1, -1).expand(1, n_blocks, NUM_CLASSES)
            )
            boundary_logits.append(
                self.head(seed).view(1, -1)[:, :NUM_BOUNDARIES]
            )
            section_logits.append(self.head(seed).view(1, -1))
        return {
            "block_logits": block_logits,
            "boundary_logits": boundary_logits,
            "section_logits": section_logits,
        }


def make_batch(doc_id: int) -> dict:
    n_blocks = BLOCKS_PER_PAGE * PAGES
    return {
        "pdf_name": f"doc{doc_id:03d}",
        "texts": [""] * n_blocks,
        "ocr_texts": [""] * n_blocks,
        "image_blocks": [None] * n_blocks,
        "block_type_ids": torch.zeros(n_blocks, dtype=torch.long),
        "bbox_norm": torch.zeros(n_blocks, 4),
        "font_size": torch.zeros(n_blocks, 1),
        "is_bold": torch.zeros(n_blocks, 1),
        "is_italic": torch.zeros(n_blocks, 1),
        "font_color": torch.zeros(n_blocks, 3),
        "labels": torch.randint(0, NUM_CLASSES, (n_blocks,)),
        # One entry per block: train_epoch slices boundaries[page_start:page_start+1].
        "boundaries": torch.zeros(n_blocks, dtype=torch.long),
        "page_splits": [BLOCKS_PER_PAGE, BLOCKS_PER_PAGE * PAGES],
    }


def run(skip_batches: int):
    model = StubModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    seen: list[int] = []

    def on_step(step: int) -> bool:
        seen.append(step)
        return False

    metrics = train.train_epoch(
        model,
        [make_batch(i) for i in range(DOCS)],
        optimizer,
        DEVICE,
        num_classes=NUM_CLASSES,
        current_doc=[None],
        step_callback=on_step,
        skip_batches=skip_batches,
    )
    return metrics, seen


def main() -> int:
    print("F13: train_epoch step accounting with a skipped prefix")

    metrics, seen = run(0)
    check("uninterrupted run visits every step", seen == list(range(1, DOCS + 1)), seen)
    check("uninterrupted run reports the full epoch",
          metrics["steps_completed"] == DOCS, metrics["steps_completed"])

    metrics, seen = run(4)
    check("resuming at step 4 skips the first 4 batches",
          seen == list(range(5, DOCS + 1)), seen)
    check("skipped batches still count toward the epoch length",
          metrics["steps_completed"] == DOCS, metrics["steps_completed"])

    metrics, seen = run(DOCS)
    check("an over-large skip is clamped to leave one batch", seen == [DOCS], seen)
    check("a clamped run still reports the full epoch length",
          metrics["steps_completed"] == DOCS, metrics["steps_completed"])

    print("F13: checkpoint plumbing is wired end to end")
    source = (REPO / "train.py").read_text(encoding="utf-8")
    check("train.py declares --ckpt_every_steps", "--ckpt_every_steps" in source)
    check("the step callback writes rolling checkpoints",
          "save_rolling_checkpoint(epoch_number, steps_done)" in source)
    check("mid-epoch snapshots record the step they were taken at",
          '"step": step_number,' in source)
    check("a mid-epoch snapshot rewinds start_epoch by one",
          "start_epoch = max(0, start_epoch - 1)" in source)
    runner = (REPO / "run_train_final.ps1").read_text(encoding="utf-8")
    check("run_train_final.ps1 forwards --ckpt_every_steps",
          '"--ckpt_every_steps"' in runner)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("All F13 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
