from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayoutEncoder(nn.Module):
    """Encode bbox, font, color, and block_type into a layout feature vector."""

    def __init__(
        self,
        bbox_dim: int = 8,
        bbox_hidden: int = 32,
        color_dim: int = 3,
        color_hidden: int = 16,
        block_type_count: int = 3,
        block_type_emb: int = 16,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bbox_proj = nn.Sequential(
            nn.Linear(bbox_dim, bbox_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.color_proj = nn.Sequential(
            nn.Linear(color_dim, color_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.block_type_emb = nn.Embedding(block_type_count, block_type_emb)

        concat_dim = bbox_hidden + 1 + 1 + 1 + color_hidden + block_type_emb
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        bbox_norm: torch.Tensor,        # [*, 8]
        font_size: torch.Tensor,        # [*, 1]
        is_bold: torch.Tensor,          # [*, 1]
        is_italic: torch.Tensor,        # [*, 1]
        font_color: torch.Tensor,       # [*, 3]
        block_type_id: torch.Tensor,    # [*] int
    ) -> torch.Tensor:
        b = bbox_norm.shape[:-1]
        bbox_f = self.bbox_proj(bbox_norm)
        color_f = self.color_proj(font_color)
        bt_f = self.block_type_emb(block_type_id)

        # ensure scalar shape alignment
        font_size = font_size.view(*b, 1)
        is_bold = is_bold.view(*b, 1)
        is_italic = is_italic.view(*b, 1)

        cat = torch.cat([bbox_f, font_size, is_bold, is_italic, color_f, bt_f], dim=-1)
        return self.fusion(cat)