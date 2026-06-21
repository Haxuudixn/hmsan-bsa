from __future__ import annotations

from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from bid_slicing.features.layout_features import LayoutEncoder
from bid_slicing.features.text_features import RoBERTaTextEncoder
from bid_slicing.features.image_features import ViTImageEncoder


class BlockType(IntEnum):
    TEXT = 0
    IMAGE = 1
    MIXED = 2


class BlockMultiModalEncoder(nn.Module):
    """Three-branch multi-modal encoder with gated modality+layout fusion.

    Branches:
      - TEXT:  RoBERTa-wwm-ext only
      - IMAGE: ViT-B/16 only
      - MIXED: OCR text → RoBERTa + ViT-B/16 → concat → Linear projection

    Gated Fusion:
      modality_proj = Linear(modality_feat, hidden_size)      (768->128)
      layout_proj   = Linear(layout_feat,   hidden_size)      (64->128)
      gate          = sigmoid(Linear([modality_feat; layout_feat], hidden_size))
      out           = ReLU(LayerNorm(gate * modality_proj + (1-gate) * layout_proj))
    """

    def __init__(
        self,
        text_encoder_name: str = "hfl/chinese-roberta-wwm-ext",
        text_max_length: int = 510,
        text_freeze: bool = False,
        image_encoder_name: str = "google/vit-base-patch16-224",
        image_freeze: bool = False,
        hidden_size: int = 128,
        layout_bbox_dim: int = 8,
        layout_bbox_hidden: int = 32,
        layout_color_dim: int = 3,
        layout_color_hidden: int = 16,
        layout_block_type_count: int = 3,
        layout_block_type_emb: int = 16,
        layout_output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # ── Pre-trained encoders ──
        self.text_encoder = RoBERTaTextEncoder(
            model_name=text_encoder_name,
            max_length=text_max_length,
            output_dim=768,
            freeze=text_freeze,
        )
        self.image_encoder = ViTImageEncoder(
            model_name=image_encoder_name,
            output_dim=768,
            freeze=image_freeze,
        )

        # ── Layout encoder (shared across all block types) ──
        self.layout_encoder = LayoutEncoder(
            bbox_dim=layout_bbox_dim,
            bbox_hidden=layout_bbox_hidden,
            color_dim=layout_color_dim,
            color_hidden=layout_color_hidden,
            block_type_count=layout_block_type_count,
            block_type_emb=layout_block_type_emb,
            output_dim=layout_output_dim,
            dropout=dropout,
        )

        # ── Mixed modality projection ──
        self.mixed_proj = nn.Sequential(
            nn.Linear(768 + 768, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Gated Fusion ──
        self.modality_proj = nn.Linear(768, hidden_size)
        self.layout_proj = nn.Linear(layout_output_dim, hidden_size)
        self.fusion_gate = nn.Sequential(
            nn.Linear(768 + layout_output_dim, hidden_size),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)

    def forward(
        self,
        texts: list[str],
        ocr_texts: list[str],
        images: list[Image.Image | None],
        block_type_ids: torch.Tensor,
        bbox_norm: torch.Tensor,
        font_size: torch.Tensor,
        is_bold: torch.Tensor,
        is_italic: torch.Tensor,
        font_color: torch.Tensor,
    ) -> torch.Tensor:
        """Return block vectors [B, hidden_size]."""
        B = len(texts)
        device = bbox_norm.device

        # ── Layout features (all blocks) ──
        layout_feat = self.layout_encoder(
            bbox_norm, font_size, is_bold, is_italic, font_color, block_type_ids
        )  # [B, layout_output_dim]

        # ── Determine block types ──
        bt = block_type_ids.cpu().tolist()
        text_idx = [i for i, t in enumerate(bt) if t == BlockType.TEXT]
        image_idx = [i for i, t in enumerate(bt) if t == BlockType.IMAGE]
        mixed_idx = [i for i, t in enumerate(bt) if t == BlockType.MIXED]

        modality_feat = torch.zeros(B, 768, device=device)

        # ── Text branch ──
        if text_idx:
            modality_feat[text_idx] = self.text_encoder(
                [texts[i] for i in text_idx]
            )

        # ── Image branch ──
        if image_idx:
            modality_feat[image_idx] = self.image_encoder(
                [images[i] for i in image_idx]
            )

        # ── Mixed branch ──
        if mixed_idx:
            ocr_emb = self.text_encoder([ocr_texts[i] for i in mixed_idx])
            img_emb = self.image_encoder([images[i] for i in mixed_idx])
            mixed_emb = self.mixed_proj(torch.cat([ocr_emb, img_emb], dim=-1))
            modality_feat[mixed_idx] = mixed_emb

        # ── Gated Fusion ──
        m_proj = self.modality_proj(modality_feat)                 # [B, H]
        l_proj = self.layout_proj(layout_feat)                     # [B, H]
        gate = self.fusion_gate(torch.cat([modality_feat, layout_feat], dim=-1))
        fused = gate * m_proj + (1.0 - gate) * l_proj              # [B, H]
        return self.fusion_dropout(F.relu(self.fusion_norm(fused)))