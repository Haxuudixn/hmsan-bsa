from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from bid_slicing.features.gated_lilt_encoder import GatedLiLTBlockEncoder
from bid_slicing.models.dilated_gated_attention import DilatedGatedPageEncoder
from bid_slicing.models.gated_infini_memory import GatedInfiniMemory
from bid_slicing.models.boundary_head import BoundaryHead, ClassificationHead, SectionHead


class HMSAN_BSA_v3(nn.Module):
    """HMSAN-BSA v3 with three Gated Sparse Attention integrations.

    Improvements over v1:
      1. DilatedGatedPageEncoder: LongNet dilation + gated token selection
         within dilated windows (replaces fixed local window)
      2. GatedInfiniMemory: compressive Infini-attention with gated write/read,
         infinite context capacity (replaces fixed K-slot GRU memory)
      3. GatedLiLTBlockEncoder: LiLT layout bias + gated modality fusion,
         dynamic modality importance (replaces simple concat+Linear)
    """

    def __init__(
        self,
        text_encoder_name: str = "hfl/chinese-roberta-wwm-ext",
        text_max_length: int = 510,
        text_freeze: bool = False,
        image_encoder_name: str = "google/vit-base-patch16-224",
        image_freeze: bool = False,
        hidden_size: int = 128,
        num_classes: int = 19,
        num_boundaries: int = 4,
        # Dilated Gated Page Encoder
        page_encoder_layers: int = 4,
        page_encoder_heads: int = 4,
        base_window_size: int = 5,
        num_global_tokens: int = 4,
        gated_sparse_k: int = 16,
        # Infini Memory
        infini_memory_slots: int = 4,
        infini_d_key: int = 64,
        # LiLT
        use_layout_bias: bool = True,
        # General
        inter_page_window: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.inter_page_window = inter_page_window

        # ── Block Encoder (Gated LiLT) ──
        self.block_encoder = GatedLiLTBlockEncoder(
            text_encoder_name=text_encoder_name,
            text_max_length=text_max_length,
            text_freeze=text_freeze,
            image_encoder_name=image_encoder_name,
            image_freeze=image_freeze,
            hidden_size=hidden_size,
            use_layout_bias=use_layout_bias,
            dropout=dropout,
        )

        # ── Page Encoder (Dilated + Gated Sparse) ──
        self.page_encoder = DilatedGatedPageEncoder(
            dim=hidden_size,
            num_heads=page_encoder_heads,
            num_layers=page_encoder_layers,
            base_window_size=base_window_size,
            num_global=num_global_tokens,
            sparsity_k=gated_sparse_k,
            dropout=dropout,
        )

        # ── Memory (Gated Infini) ──
        self.page_memory = GatedInfiniMemory(
            dim=hidden_size,
            num_slots=infini_memory_slots,
            d_key=infini_d_key,
            dropout=dropout,
        )
        self.section_memory = GatedInfiniMemory(
            dim=hidden_size,
            num_slots=infini_memory_slots,
            d_key=infini_d_key,
            dropout=dropout,
        )

        # ── Heads ──
        self.boundary_head = BoundaryHead(
            dim=hidden_size,
            num_boundaries=num_boundaries,
            dropout=dropout,
        )
        self.classification_head = ClassificationHead(
            dim=hidden_size,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.section_head = SectionHead(
            dim=hidden_size,
            num_classes=num_classes,
            dropout=dropout,
        )

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
        page_splits: list[int],
        page_mem: torch.Tensor | None = None,
        page_mem_norm: torch.Tensor | None = None,
        section_mem: torch.Tensor | None = None,
        section_mem_norm: torch.Tensor | None = None,
        page_history: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Process document page-by-page with gated Infini memory propagation.

        Returns dict with:
          - block_logits, boundary_logits, section_logits
          - page_mem, page_mem_norm, section_mem, section_mem_norm
          - page_history
        """
        device = block_type_ids.device
        B = 1

        if page_history is None:
            page_history = torch.zeros(B, 0, self.hidden_size, device=device)

        block_logits_list = []
        boundary_logits_list = []
        section_logits_list = []

        prev_end = 0
        for end in page_splits:
            # ── Extract page blocks ──
            p_texts = texts[prev_end:end]
            p_ocr = ocr_texts[prev_end:end]
            p_images = images[prev_end:end]
            p_bt = block_type_ids[prev_end:end]
            p_bbox = bbox_norm[prev_end:end]
            p_fs = font_size[prev_end:end]
            p_bold = is_bold[prev_end:end]
            p_italic = is_italic[prev_end:end]
            p_color = font_color[prev_end:end]

            # ── Gated LiLT Block Encoding ──
            block_vecs = self.block_encoder(
                texts=p_texts,
                ocr_texts=p_ocr,
                images=p_images,
                block_type_ids=p_bt,
                bbox_norm=p_bbox,
                font_size=p_fs,
                is_bold=p_bold,
                is_italic=p_italic,
                font_color=p_color,
            )  # [Ni, dim]
            block_vecs = block_vecs.unsqueeze(0)  # [1, Ni, dim]

            # ── Dilated Gated Page Encoding ──
            page_repr, encoded_blocks = self.page_encoder(block_vecs)

            # ── Inter-page context (sliding window) ──
            if page_history.size(1) > 0:
                # Simple cross-attention with history
                inter_weight = torch.softmax(
                    torch.bmm(page_repr.unsqueeze(1), page_history.transpose(1, 2)) /
                    (self.hidden_size ** 0.5),
                    dim=-1
                )
                inter_context = torch.bmm(inter_weight, page_history).squeeze(1)
                page_repr = page_repr + 0.1 * inter_context

            # ── Gated Infini Page Memory ──
            page_fused, page_mem, page_mem_norm, page_readout = self.page_memory(
                page_repr=page_repr,
                local_out=page_repr,
                memory=page_mem,
                norm=page_mem_norm,
            )

            # ── Boundary Prediction ──
            bound_logits = self.boundary_head(page_fused)
            boundary_logits_list.append(bound_logits)

            # ── Gated Infini Section Memory ──
            sect_fused, section_mem, section_mem_norm, section_readout = (
                self.section_memory(
                    page_repr=page_fused,
                    local_out=page_fused,
                    memory=section_mem,
                    norm=section_mem_norm,
                )
            )

            # ── Block Classification ──
            block_logits = self.classification_head(encoded_blocks)
            block_logits_list.append(block_logits)

            # ── Section Classification ──
            section_logits = self.section_head(page_fused, section_readout)
            section_logits_list.append(section_logits)

            # ── Track page history ──
            page_history = torch.cat([page_history, page_fused.unsqueeze(1)], dim=1)
            if page_history.size(1) > self.inter_page_window:
                page_history = page_history[:, -self.inter_page_window:, :]

            prev_end = end

        return {
            "block_logits": block_logits_list,
            "boundary_logits": boundary_logits_list,
            "section_logits": section_logits_list,
            "page_mem": page_mem,
            "page_mem_norm": page_mem_norm,
            "section_mem": section_mem,
            "section_mem_norm": section_mem_norm,
            "page_history": page_history,
        }