"""Training script for HMSAN-BSA on annotated bid document data."""

from __future__ import annotations

import argparse
import json
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
from bid_slicing.data.label_schema import LabelSchema


def parse_args():
    p = argparse.ArgumentParser(description="Train HMSAN-BSA")
    p.add_argument("--data_dir", type=str, default=r"D:\投标文件切片\新建文件夹",
                   help="Directory containing training data ZIPs")
    p.add_argument("--output_dir", type=str, default="outputs/hmsan_bsa_v1",
                   help="Output directory for checkpoints and logs")
    p.add_argument("--label_config", type=str, default="configs/labels.yaml",
                   help="Path to label YAML config")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--alpha_boundary", type=float, default=0.5)
    p.add_argument("--beta_section", type=float, default=0.5)
    p.add_argument("--class_weight_mode", type=str, default="inverse",
                   choices=["none", "inverse", "sqrt"],
                   help="Class weight mode for block classification (inverse/sqrt/none)")
    p.add_argument("--focal_loss", action="store_true",
                   help="Use focal loss for block classification")
    p.add_argument("--focal_gamma", type=float, default=2.0,
                   help="Focal loss gamma (only used with --focal_loss)")
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> dict:
    """Compute accuracy and per-class metrics."""
    mask = targets != ignore_index
    if mask.sum() == 0:
        return {"accuracy": 0.0, "valid_count": 0}

    valid_preds = preds[mask]
    valid_targets = targets[mask]

    acc = (valid_preds == valid_targets).float().mean().item()

    return {
        "accuracy": acc,
        "valid_count": int(mask.sum().item()),
        "total_count": int(targets.numel()),
    }


def compute_class_weights(documents, num_classes: int, mode: str = "inverse"):
    """Compute class weights from the training documents.

    Modes:
      - inverse: 1 / count
      - sqrt:    1 / sqrt(count)
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
) -> torch.Tensor:
    """Block classification loss with optional class weights and focal loss."""
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)

    if not use_focal:
        return F.cross_entropy(
            logits,
            targets,
            weight=class_weights,
            ignore_index=ignore_index,
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
        class_weights = class_weights.to(logits.device)
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
) -> dict:
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_bound_loss = 0.0
    total_sec_loss = 0.0
    n_batches = 0

    all_cls_preds = []
    all_cls_targets = []

    for batch in tqdm(dataloader, desc="Training", leave=False):
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

        optimizer.zero_grad()

        # Forward pass (images are loaded lazily by the model per page)
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
        optimizer.step()

        total_loss += total.item()
        total_cls_loss += avg_cls.item()
        total_bound_loss += avg_bound.item()
        total_sec_loss += avg_sec.item()
        n_batches += 1

    # Compute metrics
    all_preds = torch.cat(all_cls_preds).view(-1)
    all_targets = torch.cat(all_cls_targets).view(-1)
    metrics = compute_metrics(all_preds, all_targets)

    return {
        "loss": total_loss / max(n_batches, 1),
        "cls_loss": total_cls_loss / max(n_batches, 1),
        "bound_loss": total_bound_loss / max(n_batches, 1),
        "sec_loss": total_sec_loss / max(n_batches, 1),
        **metrics,
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
) -> dict:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_cls_preds = []
    all_cls_targets = []

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
            )
            page_bound = boundaries[page_start:page_start + 1]
            bound_loss += F.cross_entropy(bd, page_bound, ignore_index=IGNORE_INDEX)
            section_target = _get_section_label(page_labels)
            sec_loss += F.cross_entropy(sl, section_target, ignore_index=IGNORE_INDEX)
            all_cls_preds.append(bl.argmax(dim=-1).view(-1).cpu())
            all_cls_targets.append(page_labels.view(-1).cpu())

        n_pages = len(output["block_logits"])
        avg_cls = cls_loss / max(n_pages, 1)
        avg_bound = bound_loss / max(n_pages, 1)
        avg_sec = sec_loss / max(n_pages, 1)
        total = avg_cls + alpha * avg_bound + beta * avg_sec
        total_loss += total.item()
        n_batches += 1

    all_preds = torch.cat(all_cls_preds).view(-1)
    all_targets = torch.cat(all_cls_targets).view(-1)
    metrics = compute_metrics(all_preds, all_targets)

    return {"val_loss": total_loss / max(n_batches, 1), **metrics}


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

    # Load label schema
    label_schema = LabelSchema.from_yaml(args.label_config)
    print(f"Classes: {label_schema.num_classes}")
    print(f"Boundaries: {label_schema.num_boundaries}")

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
    documents = load_documents_from_zips(zip_paths)
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
        train_ds.documents, label_schema.num_classes, args.class_weight_mode
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
        collate_fn=collate_document, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=collate_document, num_workers=0,
    )

    # Model
    print("\nInitializing model...")
    model = HMSAN_BSA(
        text_encoder_name="hfl/chinese-roberta-wwm-ext",
        text_max_length=510,
        text_freeze=True,
        image_encoder_name="google/vit-base-patch16-224",
        image_freeze=True,
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
        dropout=0.1,
    ).to(device)

    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}")
    print(f"Total params: {total:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_acc = 0.0
    history = []

    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            alpha=args.alpha_boundary, beta=args.beta_section,
            class_weights=class_weights,
            use_focal=args.focal_loss,
            focal_gamma=args.focal_gamma,
        )
        val_metrics = validate(
            model, val_loader, device,
            alpha=args.alpha_boundary, beta=args.beta_section,
            class_weights=None,
            use_focal=False,
            focal_gamma=args.focal_gamma,
        )

        print(f"Train - loss: {train_metrics['loss']:.4f}, "
              f"cls: {train_metrics['cls_loss']:.4f}, "
              f"acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   - loss: {val_metrics['val_loss']:.4f}, "
              f"acc: {val_metrics['accuracy']:.4f} "
              f"({val_metrics['valid_count']}/{val_metrics['total_count']} valid)")

        history.append({
            "epoch": epoch,
            "train": {k: v if isinstance(v, (int, float)) else float(v) for k, v in train_metrics.items()},
            "val": {k: v if isinstance(v, (int, float)) else float(v) for k, v in val_metrics.items()},
        })

        # Save best
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_acc,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"  -> Best model saved (acc={best_val_acc:.4f})")

        # Save periodic checkpoint
        if epoch % 5 == 0:
            torch.save(checkpoint, output_dir / f"checkpoint_epoch{epoch}.pt")

    # Save training history
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()