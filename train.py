"""Training script for HMSAN-BSA on annotated bid document data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bid_slicing.data.dataset import (
    BidDocumentDataset,
    load_documents_from_zips,
    collate_document,
    IGNORE_INDEX,
)
from bid_slicing.models.hmsan_bsa import HMSAN_BSA
from bid_slicing.models.ablation import AblationFlags, PRESETS, resolve_preset
from bid_slicing.data.label_schema import LabelSchema


_DEBUG_MEM = os.environ.get("HMSAN_DEBUG_MEM") == "1"


def parse_args():
    p = argparse.ArgumentParser(description="Train HMSAN-BSA")
    p.add_argument("--data_dir", type=str, default=r"D:\投标文件切片\新建文件夹",
                   help="Directory containing training data ZIPs")
    p.add_argument("--output_dir", type=str, default="outputs/hmsan_bsa_v1",
                   help="Output directory for checkpoints and logs")
    p.add_argument("--label_config", type=str, default="configs/labels.yaml",
                   help="Path to label YAML config")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Deprecated, use --encoder_lr/--head_lr")
    p.add_argument("--encoder_lr", type=float, default=2e-5,
                   help="Learning rate for pretrained text/image encoders")
    p.add_argument("--head_lr", type=float, default=1e-4,
                   help="Learning rate for task heads and other modules")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient clipping max norm (0 disables)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_every_steps", type=int, default=0,
                   help="Validate every N optimizer steps inside an epoch "
                        "(0 disables). Locates the true optimum instead of "
                        "guessing an epoch budget from the epoch endpoints.")
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="Stop after this many consecutive mid-epoch validation "
                        "checks without improvement (0 disables). Requires "
                        "--val_every_steps.")
    p.add_argument("--early_stop_min_delta", type=float, default=0.0,
                   help="Minimum mid-epoch val-accuracy gain that counts as an "
                        "improvement.")
    p.add_argument("--ckpt_every_steps", type=int, default=0,
                   help="Write last.pt and flush history.json every N optimizer "
                        "steps inside an epoch (0 disables). Bounds the loss "
                        "from a native crash or power loss to N steps instead "
                        "of a whole epoch. A mid-epoch resume is approximate: "
                        "the shuffled batch order is not reproduced, so the "
                        "skipped prefix is a different draw than the one "
                        "already trained on.")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader worker processes for the train and val "
                        "loaders. 0 keeps zip reads and PNG decodes serial with "
                        "GPU compute: that is ~9%% of the step on an RTX 3060, but "
                        "a much larger share on a fast rented card, where it caps "
                        "the achievable speedup. Set 4-8 on a machine with >=8 "
                        "CPU cores. Batch order is unaffected (shuffling happens "
                        "in the main process), so results stay comparable.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--alpha_boundary", type=float, default=0.5)
    p.add_argument("--beta_section", type=float, default=0.5)
    p.add_argument("--text_freeze", action=argparse.BooleanOptionalAction, default=False,
                   help="Freeze the pretrained text encoder")
    p.add_argument("--image_freeze", action=argparse.BooleanOptionalAction, default=True,
                   help="Freeze the pretrained image encoder")
    p.add_argument("--scheduler", type=str, default="plateau",
                   choices=["none", "plateau", "cosine"],
                   help="LR scheduler to use")
    p.add_argument("--lr_patience", type=int, default=3,
                   help="ReduceLROnPlateau patience. The scheduler steps ONCE "
                        "PER EPOCH, on the epoch-end validation accuracy, so "
                        "this is measured in epochs: 3 = halve the LR after 3 "
                        "consecutive epochs without a new best val acc. 3 keeps "
                        "the historical behaviour.")
    p.add_argument("--lr_min_lr", type=float, default=0.0,
                   help="Floor for ReduceLROnPlateau (applied to every param "
                        "group). 0 = no floor, the historical behaviour. A "
                        "positive value caps how far the LR can decay.")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout probability for task modules")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing for block classification cross entropy")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="bfloat16 autocast on CUDA (halves activation memory)")
    p.add_argument("--max_blocks_per_sample", type=int, default=0,
                   help="Split long documents into page ranges of at most this "
                        "many blocks to cap VRAM per step (0 disables)")
    p.add_argument("--max_pages_per_sample", type=int, default=0,
                   help="Split long documents into page ranges of at most this "
                        "many pages to cap VRAM per step (0 disables)")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from a checkpoint path")
    p.add_argument("--init_from", type=str, default=None,
                   help="Warm-start from a checkpoint's model weights only "
                        "(epoch counter, optimizer and best score are reset)")
    p.add_argument("--class_weight_mode", type=str, default="none",
                   choices=["none", "inverse", "sqrt", "power"],
                   help="Class weight mode for block classification (none recommended for overall accuracy)")
    p.add_argument("--class_weight_power", type=float, default=0.2,
                   help="Power exponent used with --class_weight_mode power")
    p.add_argument("--focal_loss", action="store_true",
                   help="Use focal loss for block classification (not recommended for overall accuracy)")
    p.add_argument("--focal_gamma", type=float, default=2.0,
                   help="Focal loss gamma (only used with --focal_loss)")
    p.add_argument("--ablation_preset", type=str, default=None,
                   choices=sorted(PRESETS),
                   help="Architecture ablation preset (default: full model)")
    p.add_argument("--ablation_set", nargs="*", default=None,
                   metavar="KEY=VALUE",
                   help="Override individual architecture switches, e.g. "
                        "--ablation_set g2_value_gate=false gated_ffn=false")
    p.add_argument("--vit_cache_dir", type=str, default=None,
                   help="Offline frozen-ViT feature cache (see "
                        "scripts/paper/build_vit_cache.py).  Requires "
                        "--image_freeze; aborts if --no-image_freeze is given, "
                        "because cached features would silently go stale.")
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def resolve_ablation_flags(args) -> AblationFlags:
    """Build the AblationFlags for this run from --ablation_preset/--ablation_set.

    Every ablation result must record the exact switch set it was produced with,
    otherwise the paper's table cannot be audited.
    """
    if args.ablation_preset:
        try:
            _, flags = resolve_preset(args.ablation_preset)
        except KeyError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        flags = AblationFlags()

    overrides = {}
    for item in args.ablation_set or []:
        if "=" not in item:
            raise SystemExit(f"--ablation_set expects KEY=VALUE, got {item!r}")
        key, raw = (part.strip() for part in item.split("=", 1))
        if key not in AblationFlags.__dataclass_fields__:
            raise SystemExit(
                f"unknown ablation flag {key!r}; valid: "
                + ", ".join(sorted(AblationFlags.__dataclass_fields__))
            )
        low = raw.lower()
        if low in _BOOL_TRUE:
            value = True
        elif low in _BOOL_FALSE:
            value = False
        elif raw.lstrip("-").isdigit():
            value = int(raw)
        else:
            value = raw
        overrides[key] = value

    return flags.replace(**overrides) if overrides else flags


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_run_meta(output_dir: Path, run_meta: dict) -> None:
    """Persist machine-readable run metadata for the matrix runner.

    Written once before training and again on completion, so a reader can tell
    a crashed run (status stays "running") from a finished one without parsing
    the log.
    """
    tmp = output_dir / "run_meta.json.tmp"
    tmp.write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    os.replace(tmp, output_dir / "run_meta.json")


def load_vit_cache(cache_dir, image_encoder_name: str, image_freeze: bool):
    """Open the frozen-ViT cache, refusing any combination it would corrupt.

    Cached features are constant by construction.  Training a ViT while feeding
    it frozen features would silently train against stale inputs, so that
    combination is an error rather than a warning.
    """
    if not cache_dir:
        return None
    if not image_freeze:
        raise SystemExit(
            "--vit_cache_dir requires a frozen image encoder: with "
            "--no-image_freeze the cached features are stale by definition. "
            "Drop the cache for the fully-unfrozen rows."
        )
    from bid_slicing.data.vit_cache import ViTFeatureCache

    cache = ViTFeatureCache.open(cache_dir)
    if cache.model_name and cache.model_name != image_encoder_name:
        raise SystemExit(
            f"cache {cache.root} was built with {cache.model_name!r} but this "
            f"run uses {image_encoder_name!r}"
        )
    print(f"ViT feature cache: {cache.describe()}")
    if cache.partial:
        print("  WARNING: the cache is partial; misses fall back to a live "
              "ViT forward and cost the usual time.")
    return cache


def _autocast(device, use_amp: bool):
    """bfloat16 autocast context; a no-op on CPU or when --no-amp is given."""
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
    )


def _on_target_device(tensor: torch.Tensor, device) -> bool:
    """Is ``tensor`` on ``device``?

    ``torch.device("cuda") != torch.device("cuda:0")`` compares unequal even
    though they mean the same thing, so a naive ``!=`` reports every tensor of
    an index-free target as stray.
    """
    target = torch.device(device)
    if tensor.device.type != target.type:
        return False
    if target.index is None:
        return True
    return tensor.device.index == target.index


def _assert_model_on_device(model, device, context: str = "model") -> None:
    """Fail loudly if any parameter or buffer was left off ``device``.

    A silently-CPU weight does not surface until ``backward()``, where it
    appears as ``mat2 is on cpu`` with no hint of which module is at fault
    (row A4, 2026-09-25 21:27).  Auditing up front turns that into a named,
    reproducible error at startup instead.
    """
    stray = [
        f"{name} ({tensor.device})"
        for name, tensor in
        list(model.named_parameters()) + list(model.named_buffers())
        if not _on_target_device(tensor, device)
    ]
    if stray:
        preview = ", ".join(stray[:5])
        more = f" (+{len(stray) - 5} more)" if len(stray) > 5 else ""
        raise RuntimeError(
            f"{context}: {len(stray)} parameter(s)/buffer(s) are not on "
            f"{device}: {preview}{more}"
        )


def _assert_tensors_on_device(tensors: dict, device, context: str = "inputs") -> None:
    """Fail loudly if any tensor handed to the model is not on ``device``.

    ``dataset.py`` builds its batch on CPU and the trainer moves each entry
    with ``.to(device)``, which returns a *new* tensor -- the batch dict keeps
    its CPU originals, so auditing the dict would flag every step.  Audit the
    moved locals instead: that is exactly what the model receives, and a key
    somebody forgot to move gets named here rather than surfacing later as an
    anonymous ``mat2 is on cpu`` inside ``backward()``.
    """
    stray = [
        f"{name} ({tensor.device})"
        for name, tensor in tensors.items()
        if not _on_target_device(tensor, device)
    ]
    if stray:
        raise RuntimeError(
            f"{context}: tensor(s) not moved to {device}: {', '.join(stray)}"
        )


def _jsonable(value):
    """Coerce tensors/numpy scalars to plain Python, keeping nested structure.

    history.json now carries the boundary/section metric sub-dicts, so a flat
    ``float(v)`` comprehension would raise on them.
    """
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> dict:
    """Compute overall and per-class metrics."""
    mask = targets != ignore_index
    if mask.sum() == 0:
        result = {"accuracy": 0.0, "valid_count": 0, "total_count": 0}
        if num_classes is not None:
            for c in range(num_classes):
                result[f"recall_{c}"] = 0.0
                result[f"precision_{c}"] = 0.0
                result[f"f1_{c}"] = 0.0
                result[f"support_{c}"] = 0
        return result

    valid_preds = preds[mask]
    valid_targets = targets[mask]

    acc = (valid_preds == valid_targets).float().mean().item()
    result = {
        "accuracy": acc,
        "valid_count": int(mask.sum().item()),
        "total_count": int(targets.numel()),
    }

    if num_classes is not None:
        recalls = []
        precisions = []
        f1s = []
        for c in range(num_classes):
            tp = int(((valid_preds == c) & (valid_targets == c)).sum().item())
            fp = int(((valid_preds == c) & (valid_targets != c)).sum().item())
            fn = int(((valid_preds != c) & (valid_targets == c)).sum().item())
            support = tp + fn
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / support if support > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            result[f"recall_{c}"] = recall
            result[f"precision_{c}"] = precision
            result[f"f1_{c}"] = f1
            result[f"support_{c}"] = support
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
        result["macro_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
    return result


def compute_class_weights(documents, num_classes: int, mode: str = "inverse", power: float = 0.2):
    """Compute class weights from the training documents.

    Modes:
      - inverse: 1 / count
      - sqrt:    1 / sqrt(count)
      - power:   1 / count^power (mild balancing)
      - none:    disable weighting
    """
    if mode == "none":
        return None

    counts = torch.zeros(num_classes, dtype=torch.float)
    for doc in documents:
        for page in doc.pages:
            for block in page.blocks:
                label_id = block.label_id
                if label_id != IGNORE_INDEX:
                    counts[label_id] += 1

    if counts.sum() == 0:
        return None

    if mode == "inverse":
        weights = 1.0 / (counts + 1e-6)
    elif mode == "sqrt":
        weights = 1.0 / torch.sqrt(counts + 1e-6)
    elif mode == "power":
        weights = 1.0 / (counts + 1e-6).pow(power)
    else:
        return None

    # Normalize so the mean weight is 1.0.
    weights = weights / weights.sum() * num_classes
    return weights


def block_cls_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    ignore_index: int = IGNORE_INDEX,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Block classification loss with optional class weights and focal loss."""
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)

    if not use_focal:
        if class_weights is not None:
            # Match dtype as well as device: under bfloat16 autocast the logits
            # are bf16 while these weights are built in fp32, and cross_entropy
            # rejects the pair with "expected scalar type BFloat16 but found
            # Float".  (F8 turned class weighting on by default, which is what
            # made this reachable.)
            class_weights = class_weights.to(
                device=logits.device, dtype=logits.dtype
            )
        return F.cross_entropy(
            logits,
            targets,
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
    ce = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        ignore_index=ignore_index,
    )
    valid = targets != ignore_index
    if not valid.any():
        return torch.tensor(0.0, device=logits.device)

    pt = torch.exp(-ce)
    focal = (1.0 - pt) ** focal_gamma * ce
    if class_weights is not None:
        class_weights = class_weights.to(device=logits.device, dtype=focal.dtype)
        focal = focal * class_weights[targets.clamp(min=0)]
    return focal[valid].mean()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float = 0.5,
    beta: float = 0.5,
    class_weights: torch.Tensor | None = None,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    num_classes: int | None = None,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.0,
    use_amp: bool = False,
    current_doc: list | None = None,
    step_callback=None,
    skip_batches: int = 0,
) -> dict:
    """Train one epoch."""
    if skip_batches:
        # Never skip the whole epoch: at least one batch must run, otherwise the
        # metric aggregation below has nothing to concatenate.
        skip_batches = max(0, min(skip_batches, len(dataloader) - 1))
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_bound_loss = 0.0
    total_sec_loss = 0.0
    n_batches = 0
    stop_requested = False

    all_cls_preds = []
    all_cls_targets = []

    for step_index, batch in enumerate(
        tqdm(dataloader, desc="Training", leave=False), start=1
    ):
        # Crash recovery: a mid-epoch checkpoint records the step it was written
        # at, and the resumed process re-enters this epoch skipping that prefix.
        # The shuffled batch order is not reproduced, so the skipped prefix is a
        # different draw than the one already trained on -- approximate, but it
        # keeps the rest of the epoch instead of discarding all of it.
        n_batches = step_index
        if step_index <= skip_batches:
            continue
        if current_doc is not None:
            current_doc[0] = batch["pdf_name"]
        # Move to device
        block_type_ids = batch["block_type_ids"].to(device)
        bbox_norm = batch["bbox_norm"].to(device)
        font_size = batch["font_size"].to(device)
        is_bold = batch["is_bold"].to(device)
        is_italic = batch["is_italic"].to(device)
        font_color = batch["font_color"].to(device)
        labels = batch["labels"].to(device)
        boundaries = batch["boundaries"].to(device)
        page_splits = batch["page_splits"]
        _assert_tensors_on_device(
            {
                "block_type_ids": block_type_ids,
                "bbox_norm": bbox_norm,
                "font_size": font_size,
                "is_bold": is_bold,
                "is_italic": is_italic,
                "font_color": font_color,
                "labels": labels,
                "boundaries": boundaries,
            },
            device,
            context="train batch",
        )

        optimizer.zero_grad()

        # Forward pass (images are loaded lazily by the model per page)
        with _autocast(device, use_amp):
            output = model(
                texts=batch["texts"],
                ocr_texts=batch["ocr_texts"],
                image_blocks=batch["image_blocks"],
                block_type_ids=block_type_ids,
                bbox_norm=bbox_norm,
                font_size=font_size,
                is_bold=is_bold,
                is_italic=is_italic,
                font_color=font_color,
                page_splits=page_splits,
            )

        # Compute losses
        cls_loss = torch.tensor(0.0, device=device)
        bound_loss = torch.tensor(0.0, device=device)
        sec_loss = torch.tensor(0.0, device=device)

        for i, (bl, bd, sl) in enumerate(zip(
            output["block_logits"],
            output["boundary_logits"],
            output["section_logits"],
        )):
            # Block classification loss
            page_start = 0 if i == 0 else page_splits[i - 1]
            page_end = page_splits[i]
            page_labels = labels[page_start:page_end].unsqueeze(0)  # [1, Ni]
            cls_loss += block_cls_loss(
                bl,
                page_labels,
                class_weights=class_weights,
                use_focal=use_focal,
                focal_gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )

            # Boundary loss
            page_bound = boundaries[page_start:page_start + 1]  # 1 boundary per page
            bound_loss += F.cross_entropy(bd, page_bound, ignore_index=IGNORE_INDEX)

            # Section loss
            section_target = _get_section_label(page_labels)
            sec_loss += F.cross_entropy(sl, section_target, ignore_index=IGNORE_INDEX)

            # Track predictions (flatten per-page)
            with torch.no_grad():
                all_cls_preds.append(bl.argmax(dim=-1).view(-1).cpu())
                all_cls_targets.append(page_labels.view(-1).cpu())

        n_pages = len(output["block_logits"])
        # Average per-page loss
        avg_cls = cls_loss / max(n_pages, 1)
        avg_bound = bound_loss / max(n_pages, 1)
        avg_sec = sec_loss / max(n_pages, 1)
        total = avg_cls + alpha * avg_bound + beta * avg_sec
        total.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Release the cached blocks of this document: with sysmem fallback the
        # allocator would otherwise keep growing into host RAM across documents.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if _DEBUG_MEM and device.type == "cuda":
            torch.cuda.synchronize()
            print(f"    [mem] doc={batch['pdf_name']} blocks={len(batch['texts'])} "
                  f"pages={len(page_splits)} "
                  f"peak_alloc={torch.cuda.max_memory_allocated() / 2**30:.2f}GB "
                  f"peak_reserved={torch.cuda.max_memory_reserved() / 2**30:.2f}GB",
                  flush=True)
            torch.cuda.reset_peak_memory_stats()

        total_loss += total.item()
        total_cls_loss += avg_cls.item()
        total_bound_loss += avg_bound.item()
        total_sec_loss += avg_sec.item()

        # Mid-epoch hook: lets main() validate on a step cadence so the real
        # optimum can be located instead of guessed from three epoch endpoints.
        if step_callback is not None and step_callback(n_batches):
            stop_requested = True
            break

    # Compute metrics
    all_preds = torch.cat(all_cls_preds).view(-1)
    all_targets = torch.cat(all_cls_targets).view(-1)
    metrics = compute_metrics(all_preds, all_targets, num_classes=num_classes)

    return {
        "loss": total_loss / max(n_batches, 1),
        "cls_loss": total_cls_loss / max(n_batches, 1),
        "bound_loss": total_bound_loss / max(n_batches, 1),
        "sec_loss": total_sec_loss / max(n_batches, 1),
        **metrics,
        "steps_completed": n_batches,
        "stop_requested": stop_requested,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    alpha: float = 0.5,
    beta: float = 0.5,
    class_weights: torch.Tensor | None = None,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    num_classes: int | None = None,
    num_boundaries: int = 4,
    section_start_id: int = 1,
    label_smoothing: float = 0.0,
    use_amp: bool = False,
) -> dict:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_cls_preds = []
    all_cls_targets = []
    all_bound_preds = []
    all_bound_targets = []
    all_sec_preds = []
    all_sec_targets = []

    for batch in tqdm(dataloader, desc="Validation", leave=False):
        block_type_ids = batch["block_type_ids"].to(device)
        bbox_norm = batch["bbox_norm"].to(device)
        font_size = batch["font_size"].to(device)
        is_bold = batch["is_bold"].to(device)
        is_italic = batch["is_italic"].to(device)
        font_color = batch["font_color"].to(device)
        labels = batch["labels"].to(device)
        boundaries = batch["boundaries"].to(device)
        page_splits = batch["page_splits"]
        _assert_tensors_on_device(
            {
                "block_type_ids": block_type_ids,
                "bbox_norm": bbox_norm,
                "font_size": font_size,
                "is_bold": is_bold,
                "is_italic": is_italic,
                "font_color": font_color,
                "labels": labels,
                "boundaries": boundaries,
            },
            device,
            context="val batch",
        )

        with _autocast(device, use_amp):
            output = model(
                texts=batch["texts"],
                ocr_texts=batch["ocr_texts"],
                image_blocks=batch["image_blocks"],
                block_type_ids=block_type_ids,
                bbox_norm=bbox_norm,
                font_size=font_size,
                is_bold=is_bold,
                is_italic=is_italic,
                font_color=font_color,
                page_splits=page_splits,
            )

        cls_loss = torch.tensor(0.0, device=device)
        bound_loss = torch.tensor(0.0, device=device)
        sec_loss = torch.tensor(0.0, device=device)

        for i, (bl, bd, sl) in enumerate(zip(
            output["block_logits"],
            output["boundary_logits"],
            output["section_logits"],
        )):
            page_start = 0 if i == 0 else page_splits[i - 1]
            page_end = page_splits[i]
            page_labels = labels[page_start:page_end].unsqueeze(0)
            cls_loss += block_cls_loss(
                bl,
                page_labels,
                class_weights=class_weights,
                use_focal=use_focal,
                focal_gamma=focal_gamma,
                label_smoothing=label_smoothing,
            )
            page_bound = boundaries[page_start:page_start + 1]
            bound_loss += F.cross_entropy(bd, page_bound, ignore_index=IGNORE_INDEX)
            section_target = _get_section_label(page_labels)
            sec_loss += F.cross_entropy(sl, section_target, ignore_index=IGNORE_INDEX)
            all_cls_preds.append(bl.argmax(dim=-1).view(-1).cpu())
            all_cls_targets.append(page_labels.view(-1).cpu())
            all_bound_preds.append(bd.argmax(dim=-1).view(-1).cpu())
            all_bound_targets.append(page_bound.view(-1).cpu())
            all_sec_preds.append(sl.argmax(dim=-1).view(-1).cpu())
            all_sec_targets.append(section_target.view(-1).cpu())

        n_pages = len(output["block_logits"])
        avg_cls = cls_loss / max(n_pages, 1)
        avg_bound = bound_loss / max(n_pages, 1)
        avg_sec = sec_loss / max(n_pages, 1)
        total = avg_cls + alpha * avg_bound + beta * avg_sec
        total_loss += total.item()
        n_batches += 1

    all_preds = torch.cat(all_cls_preds).view(-1)
    all_targets = torch.cat(all_cls_targets).view(-1)
    metrics = compute_metrics(all_preds, all_targets, num_classes=num_classes)

    # The structural heads carry half the training objective (alpha = beta = 0.5)
    # yet no metric for them existed anywhere in the pipeline. Both are
    # page-level: one boundary call and one section call per page.
    bound_preds = torch.cat(all_bound_preds).view(-1)
    bound_targets = torch.cat(all_bound_targets).view(-1)
    sec_preds = torch.cat(all_sec_preds).view(-1)
    sec_targets = torch.cat(all_sec_targets).view(-1)
    bound_metrics = compute_metrics(
        bound_preds, bound_targets, num_classes=num_boundaries
    )
    sec_metrics = compute_metrics(
        sec_preds, sec_targets, num_classes=num_classes
    )

    # Binary view of the boundary head: "does a section start on this page?" is
    # the question the boundary gate is meant to answer.
    valid = bound_targets != IGNORE_INDEX
    is_start_pred = (bound_preds == section_start_id) & valid
    is_start_true = (bound_targets == section_start_id) & valid
    tp = int((is_start_pred & is_start_true).sum().item())
    fp = int((is_start_pred & ~is_start_true & valid).sum().item())
    fn = int((~is_start_pred & is_start_true).sum().item())
    start_precision = tp / (tp + fp) if (tp + fp) else 0.0
    start_recall = tp / (tp + fn) if (tp + fn) else 0.0
    start_f1 = (
        2 * start_precision * start_recall / (start_precision + start_recall)
        if (start_precision + start_recall) else 0.0
    )

    n_sec = int(num_classes or 0)
    return {
        "val_loss": total_loss / max(n_batches, 1),
        **metrics,
        "boundary": {
            "accuracy": bound_metrics["accuracy"],
            "macro_f1": bound_metrics.get("macro_f1", 0.0),
            "valid_count": bound_metrics["valid_count"],
            "per_class_f1": [
                bound_metrics.get(f"f1_{c}", 0.0) for c in range(num_boundaries)
            ],
            "per_class_support": [
                bound_metrics.get(f"support_{c}", 0) for c in range(num_boundaries)
            ],
            "section_start_precision": start_precision,
            "section_start_recall": start_recall,
            "section_start_f1": start_f1,
        },
        "section": {
            "accuracy": sec_metrics["accuracy"],
            "macro_f1": sec_metrics.get("macro_f1", 0.0),
            "valid_count": sec_metrics["valid_count"],
            "per_class_f1": [sec_metrics.get(f"f1_{c}", 0.0) for c in range(n_sec)],
            "per_class_support": [sec_metrics.get(f"support_{c}", 0) for c in range(n_sec)],
        },
    }


def _get_section_label(page_labels: torch.Tensor) -> torch.Tensor:
    """Get majority label for a page as section label."""
    valid = page_labels[page_labels != IGNORE_INDEX]
    if valid.numel() == 0:
        return torch.tensor([IGNORE_INDEX], device=page_labels.device)
    mode = torch.mode(valid).values
    return mode.unsqueeze(0)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    use_amp = bool(args.amp) and device.type == "cuda"
    print(f"bfloat16 autocast: {use_amp}")

    # Open the frozen-ViT cache before anything expensive: an illegal
    # combination (cache + trainable ViT) or a missing cache should fail before
    # ~10 s and 1.3 GB are spent loading the corpus.
    vit_cache = load_vit_cache(
        args.vit_cache_dir, "google/vit-base-patch16-224", args.image_freeze
    )

    # Load label schema
    label_schema = LabelSchema.from_yaml(args.label_config)
    print(f"Classes: {label_schema.num_classes}")
    print(f"Boundaries: {label_schema.num_boundaries}")
    # "B-SECTION" marks the first block of a section; the boundary gate exists to
    # find it, so the metric reads it out by name instead of assuming an index.
    section_start_id = label_schema.boundary_to_id.get("B-SECTION", 1)

    # Find data ZIPs
    data_dir = Path(args.data_dir)
    zip_files = sorted(data_dir.glob("training_data_*.zip"))
    zip_paths = [str(z) for z in zip_files]
    print(f"Found {len(zip_paths)} ZIP files:")
    for z in zip_paths:
        print(f"  {Path(z).name}")

    if not zip_paths:
        print("ERROR: No training data ZIPs found!")
        return

    # Load documents
    print("\nLoading documents...")
    documents = load_documents_from_zips(
        zip_paths, args.max_blocks_per_sample, args.max_pages_per_sample
    )
    print(f"Loaded {len(documents)} documents")

    total_blocks = sum(
        len(p.blocks) for doc in documents for p in doc.pages
    )
    labeled_blocks = sum(
        1 for doc in documents for p in doc.pages for b in p.blocks
        if b.label_id != IGNORE_INDEX
    )
    print(f"Total blocks: {total_blocks}")
    print(f"Labeled blocks: {labeled_blocks} ({100*labeled_blocks/max(total_blocks,1):.1f}%)")

    # Split
    dataset = BidDocumentDataset(documents)
    train_ds, val_ds, test_ds = dataset.split_by_documents(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Compute class weights from training documents only.
    class_weights = compute_class_weights(
        train_ds.documents, label_schema.num_classes, args.class_weight_mode,
        power=args.class_weight_power,
    )
    if class_weights is not None:
        print("Class weights (mean=1.0, computed on train split):")
        for i, w in enumerate(class_weights.tolist()):
            name = label_schema.decode_class(i) or "(ignore)"
            print(f"  {name}: {w:.3f}")
    else:
        print("Class weights: disabled")

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        collate_fn=collate_document, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=collate_document, num_workers=args.num_workers,
    )

    # Model
    ablation_flags = resolve_ablation_flags(args)
    print("\nInitializing model...")
    print(f"Ablation flags: {ablation_flags.signature()}"
          + ("" if ablation_flags.is_default() else "  [reduced model]"))
    model = HMSAN_BSA(
        text_encoder_name="hfl/chinese-roberta-wwm-ext",
        text_max_length=510,
        text_freeze=args.text_freeze,
        image_encoder_name="google/vit-base-patch16-224",
        image_freeze=args.image_freeze,
        hidden_size=128,
        num_classes=label_schema.num_classes,
        num_boundaries=label_schema.num_boundaries,
        page_encoder_layers=2,
        page_encoder_heads=4,
        gsa_k_base=16,
        gsa_k_min=4,
        gsa_k_max=32,
        gsa_indexer_heads=4,
        gsa_indexer_dim=32,
        num_global_tokens=4,
        inter_page_window=3,
        dropout=args.dropout,
        flags=ablation_flags,
        vit_cache=vit_cache,
    ).to(device)
    _assert_model_on_device(model, device, context="model after .to(device)")

    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}")
    print(f"Total params: {total:,}")

    # Machine-readable record of the run, so the matrix runner never has to
    # parse a log to learn what was configured or how it ended.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "schema": 1,
        "status": "running",
        "started_at": _now_iso(),
        "finished_at": None,
        "argv": sys.argv[1:],
        "data_dir": str(args.data_dir),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "seed": args.seed,
        "val_every_steps": args.val_every_steps,
        "ckpt_every_steps": args.ckpt_every_steps,
        "device": str(device),
        "amp": use_amp,
        "text_freeze": bool(args.text_freeze),
        "image_freeze": bool(args.image_freeze),
        "vit_cache": vit_cache.to_dict() if vit_cache is not None else None,
        "ablation_preset": args.ablation_preset,
        "ablation_signature": ablation_flags.signature(),
        "ablation_flags": ablation_flags.to_dict(),
        "params_total": int(total),
        "params_trainable": int(trainable),
        "optimizer": {
            "encoder_lr": args.encoder_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "scheduler": args.scheduler,
            "lr_patience": args.lr_patience,
            "lr_min_lr": args.lr_min_lr,
            "class_weight_mode": args.class_weight_mode,
            "dropout": args.dropout,
        },
        "caps": {
            "max_blocks_per_sample": args.max_blocks_per_sample,
            "max_pages_per_sample": args.max_pages_per_sample,
        },
        "split": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
        },
        "labeled_blocks": int(labeled_blocks),
        "total_blocks": int(total_blocks),
    }
    _write_run_meta(output_dir, run_meta)

    start_epoch = 0
    start_step = 0
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    if args.resume:
        resume_path = Path(args.resume)
        print(f"Resuming from {resume_path}")
        checkpoint_data = torch.load(resume_path, map_location=device, weights_only=False)
        src_sig = checkpoint_data.get("ablation_signature")
        if src_sig and src_sig != ablation_flags.signature():
            raise SystemExit(
                f"checkpoint {resume_path} was trained with ablation flags "
                f"{src_sig!r}, but this run uses {ablation_flags.signature()!r}"
            )
        model.load_state_dict(checkpoint_data["model_state_dict"])
        start_epoch = int(checkpoint_data.get("epoch", 0))
        start_step = int(checkpoint_data.get("step", 0) or 0)
        best_val_acc = float(checkpoint_data.get("val_accuracy", 0.0))
        if start_step > 0:
            # The checkpoint was written mid-epoch rather than at an epoch
            # boundary: re-enter that same epoch and skip the steps already
            # trained, instead of discarding the whole epoch.
            print(f"Resumed mid-epoch checkpoint: epoch={start_epoch} "
                  f"step={start_step} (approximate: the shuffled batch order is "
                  f"not reproduced, so the skipped prefix is a different draw "
                  f"than the one already trained on)")
            start_epoch = max(0, start_epoch - 1)
        else:
            print(f"Resumed epoch={start_epoch}, best_val_acc={best_val_acc:.4f}")
    elif args.init_from:
        init_path = Path(args.init_from)
        print(f"Warm-starting model weights from {init_path}")
        init_data = torch.load(init_path, map_location="cpu", weights_only=False)
        init_state = init_data.get("model_state_dict", init_data)
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        print(f"  source epoch={init_data.get('epoch', 'n/a')} "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  missing keys (first 5): {missing[:5]}")
        if unexpected:
            print(f"  unexpected keys (first 5): {unexpected[:5]}")
        del init_data, init_state

    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("block_encoder.text_encoder.encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)

    param_groups = [{"params": head_params, "lr": args.head_lr}]
    if encoder_params:
        param_groups.insert(0, {"params": encoder_params, "lr": args.encoder_lr})
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
    )
    print(f"Text encoder params: {len(encoder_params):,}")
    print(f"Head/other params: {len(head_params):,}")

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=args.lr_patience,
            min_lr=args.lr_min_lr,
        )
        print(f"LR scheduler: plateau factor=0.5 "
              f"patience={args.lr_patience} epoch-end checks, "
              f"min_lr={args.lr_min_lr}")
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    else:
        scheduler = None

    # (output_dir and run_meta.json were created right after the model was
    # built, so they exist even if setup below fails)

    # Documents that blew up VRAM in an earlier attempt are skipped, so a
    # pathological sample can never trap the run in a crash/restart loop.
    skipped_path = output_dir / "skipped_docs.json"
    skipped: set[str] = set()
    if skipped_path.exists():
        try:
            skipped = set(json.loads(skipped_path.read_text(encoding="utf-8")))
        except Exception:
            skipped = set()
    if skipped:
        train_ds = BidDocumentDataset(
            [d for d in train_ds.documents if d.pdf_name not in skipped]
        )
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True,
            collate_fn=collate_document, num_workers=args.num_workers,
        )
        print(f"Skipping {len(skipped)} document(s) from {skipped_path.name}")

    # Training loop
    history = []
    if args.resume:
        history_path = output_dir / "history.json"
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                history = []

    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}")

    # ---- mid-epoch validation + early stopping (F5) ----
    # Validating only at epoch boundaries gave three data points per run with a
    # 3-7pp swing between them, so the optimum could not be located. A step
    # cadence costs about 1.7 min per check against a ~1.8 h epoch (~6%).
    mid_best = -1.0
    mid_best_step = 0
    mid_checks = 0
    checks_without_improvement = 0

    def build_checkpoint(epoch_number, step_number):
        """Snapshot that --resume can continue from.

        step_number == 0 means the epoch finished; a non-zero step means the
        snapshot was taken mid-epoch and that epoch must be re-entered.
        """
        return {
            "epoch": epoch_number,
            "step": step_number,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": best_val_acc,
            "ablation_flags": ablation_flags.to_dict(),
            "ablation_signature": ablation_flags.signature(),
            "text_freeze": bool(args.text_freeze),
            "image_freeze": bool(args.image_freeze),
            "seed": int(args.seed),
            "vit_cache": vit_cache.to_dict() if vit_cache is not None else None,
        }

    def save_rolling_checkpoint(epoch_number, step_number):
        """Atomically replace last.pt so a crash never leaves it truncated."""
        tmp_path = output_dir / "last.pt.tmp"
        torch.save(build_checkpoint(epoch_number, step_number), tmp_path)
        os.replace(tmp_path, output_dir / "last.pt")

    def flush_history():
        """Persist history so mid-epoch metrics survive a crash too."""
        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def make_step_callback(epoch_number):
        """Build the per-step hook for one epoch."""
        def on_step(steps_done):
            nonlocal mid_best, mid_best_step, mid_checks, checks_without_improvement
            stop_requested = False

            if (args.val_every_steps > 0
                    and steps_done % args.val_every_steps == 0):
                mid = validate(
                    model, val_loader, device,
                    alpha=args.alpha_boundary, beta=args.beta_section,
                    class_weights=None,
                    use_focal=False,
                    focal_gamma=args.focal_gamma,
                    num_classes=label_schema.num_classes,
                    num_boundaries=label_schema.num_boundaries,
                    section_start_id=section_start_id,
                    label_smoothing=args.label_smoothing,
                    use_amp=use_amp,
                )
                # validate() switched the model to eval(); training must continue.
                model.train()
                mid_checks += 1
                monitor = mid["accuracy"]
                if monitor > mid_best + args.early_stop_min_delta:
                    mid_best = monitor
                    mid_best_step = steps_done
                    checks_without_improvement = 0
                else:
                    checks_without_improvement += 1
                history.append({
                    "epoch": epoch_number,
                    "step": steps_done,
                    "phase": "mid_epoch",
                    "val": _jsonable(mid),
                })
                print(f"  [mid ep{epoch_number} step {steps_done}] "
                      f"acc={mid['accuracy']:.4f} "
                      f"macro_f1={mid.get('macro_f1', 0.0):.4f} "
                      f"boundary_f1={mid['boundary']['macro_f1']:.4f} "
                      f"section_f1={mid['section']['macro_f1']:.4f} "
                      f"start_f1={mid['boundary']['section_start_f1']:.4f}",
                      flush=True)
                if (args.early_stop_patience > 0
                        and checks_without_improvement >= args.early_stop_patience):
                    print(f"  [early stop] {checks_without_improvement} "
                          f"mid-epoch checks without improvement (best "
                          f"acc={mid_best:.4f} at step {mid_best_step}); "
                          f"stopping.", flush=True)
                    stop_requested = True

            # Crash recovery is independent of the validation cadence, so a run
            # with --val_every_steps 0 still gets resumable checkpoints.
            if (args.ckpt_every_steps > 0
                    and steps_done % args.ckpt_every_steps == 0):
                flush_history()
                # History first, checkpoint second.  last.pt is ~1.6 GB, so the
                # write is a wide window: the 2026-09-26 crash landed inside it
                # and lost the step-150 validation point that had already been
                # computed and printed.  The JSON is small and quick, so
                # flushing it first keeps the metrics even when the checkpoint
                # write is cut off mid-file.
                save_rolling_checkpoint(epoch_number, steps_done)
                print(f"  [ckpt] last.pt written at ep{epoch_number} "
                      f"step {steps_done}", flush=True)

            return stop_requested
        return on_step


    # A mid-epoch resume skips a prefix of exactly one epoch (the one that was
    # interrupted); every later epoch starts from step 1 as usual.
    first_epoch_after_resume = start_epoch + 1

    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        doc_holder: list = [None]
        while True:
            try:
                train_metrics = train_epoch(
                    model, train_loader, optimizer, device,
                    alpha=args.alpha_boundary, beta=args.beta_section,
                    class_weights=class_weights,
                    use_focal=args.focal_loss,
                    focal_gamma=args.focal_gamma,
                    num_classes=label_schema.num_classes,
                    grad_clip=args.grad_clip,
                    label_smoothing=args.label_smoothing,
                    use_amp=use_amp,
                    current_doc=doc_holder,
                    step_callback=make_step_callback(epoch),
                    skip_batches=(
                        start_step if epoch == first_epoch_after_resume else 0
                    ),
                )
                break
            except torch.cuda.OutOfMemoryError:
                bad_doc = doc_holder[0]
                if not bad_doc:
                    raise
                print(f"[oom] CUDA OOM on {bad_doc}: skipping it and "
                      f"restarting epoch {epoch}", flush=True)
                skipped.add(bad_doc)
                skipped_path.write_text(
                    json.dumps(sorted(skipped), ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )
                train_ds = BidDocumentDataset(
                    [d for d in train_ds.documents if d.pdf_name not in skipped]
                )
                train_loader = DataLoader(
                    train_ds, batch_size=1, shuffle=True,
                    collate_fn=collate_document, num_workers=args.num_workers,
                )
                doc_holder = [None]
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if train_metrics.get("stop_requested"):
            print(f"  [early stop] requested during epoch {epoch}; stopping.",
                  flush=True)
            break

        val_metrics = validate(
            model, val_loader, device,
            alpha=args.alpha_boundary, beta=args.beta_section,
            class_weights=None,
            use_focal=False,
            focal_gamma=args.focal_gamma,
            num_classes=label_schema.num_classes,
            num_boundaries=label_schema.num_boundaries,
            section_start_id=section_start_id,
            label_smoothing=args.label_smoothing,
            use_amp=use_amp,
        )

        print(f"Train - loss: {train_metrics['loss']:.4f}, "
              f"cls: {train_metrics['cls_loss']:.4f}, "
              f"acc: {train_metrics['accuracy']:.4f}, "
              f"macro_f1: {train_metrics.get('macro_f1', 0.0):.4f}")
        print(f"Val   - loss: {val_metrics['val_loss']:.4f}, "
              f"acc: {val_metrics['accuracy']:.4f}, "
              f"macro_f1: {val_metrics.get('macro_f1', 0.0):.4f} "
              f"({val_metrics['valid_count']}/{val_metrics['total_count']} valid)")
        val_recalls = [
            val_metrics.get(f"recall_{c}", 0.0)
            for c in range(label_schema.num_classes)
        ]
        print("Val class recall: " + "  ".join(
            f"{label_schema.decode_class(c)}:{r:.3f}"
            for c, r in enumerate(val_recalls)
        ))

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_metrics["accuracy"])
            else:
                scheduler.step()

        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / 2 ** 20
            if device.type == "cuda" else 0.0
        )
        history.append({
            "epoch": epoch,
            "peak_vram_mb": round(peak_vram_mb, 1),
            "train": _jsonable(train_metrics),
            "val": _jsonable(val_metrics),
        })

        # Save best
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_val_macro_f1 = val_metrics.get("macro_f1", 0.0)
            torch.save(build_checkpoint(epoch, 0), output_dir / "best_model.pt")
            print(f"  -> Best model saved (acc={best_val_acc:.4f})")

        # Save periodic checkpoint
        if epoch % 5 == 0:
            torch.save(build_checkpoint(epoch, 0),
                       output_dir / f"checkpoint_epoch{epoch}.pt")

        # Rolling checkpoint for crash recovery: write then atomically replace,
        # so an interrupted run never leaves a half-written last.pt behind.
        # step=0 marks the epoch as complete, so --resume enters the next epoch
        # rather than re-entering this one.
        save_rolling_checkpoint(epoch, 0)

        # Persist history every epoch so progress survives a crash.
        flush_history()

    # Save training history
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # history also carries mid_epoch entries now (F5/F6), so len(history) would
    # report 4-5 "epochs" for a 1-epoch run.  Only records without a phase tag
    # are real epoch boundaries, and the last-epoch metrics that the report
    # convention keys on must be read from those, never from the tail of the
    # raw list (a mid-epoch check can out-score every epoch-end value).
    epoch_records = [h for h in history if h.get("phase") != "mid_epoch"]
    last_epoch_val = epoch_records[-1]["val"] if epoch_records else {}
    run_meta.update({
        "status": "completed",
        "finished_at": _now_iso(),
        "epochs_completed": len(epoch_records),
        "best_val_accuracy": best_val_acc,
        "best_val_macro_f1": best_val_macro_f1,
        "final_val_accuracy": last_epoch_val.get("accuracy"),
        "final_val_macro_f1": last_epoch_val.get("macro_f1"),
        "peak_vram_mb": max(
            (entry.get("peak_vram_mb", 0.0) for entry in history), default=0.0
        ),
        "skipped_docs": sorted(skipped),
    })
    _write_run_meta(output_dir, run_meta)

    print(f"\n{'='*60}")
    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
