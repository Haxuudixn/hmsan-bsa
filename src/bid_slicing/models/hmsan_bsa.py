from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from PIL import Image

from bid_slicing.models.block_encoder import BlockMultiModalEncoder
from bid_slicing.models.page_encoder import PageEncoder, InterPageAttention
from bid_slicing.models.ablation import AblationFlags
from bid_slicing.models.memory import InfiniPageMemory, InfiniSectionMemory
from bid_slicing.models.boundary_head import BoundaryHead, ClassificationHead, SectionHead

class HMSAN_BSA(nn.Module):
    """Hierarchical Memory Sparse Attention Network with Boundary-aware Section Aggregation.

    Full model including:
      - BlockMultiModalEncoder  (text / image / mixed)
      - PageEncoder             (sparse local attention with global tokens)
      - InfiniPageMemory        (compressive memory via Infini-attention)
      - InfiniSectionMemory     (compressive section memory with boundary gating)
      - InterPageAttention      (cross-page sliding window)
      - BoundaryHead            (B/I/E/O prediction)
      - ClassificationHead      (block-level label)
      - SectionHead             (section-level label)
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
        page_encoder_layers: int = 2,
        page_encoder_heads: int = 4,
        gsa_k_base: int = 16,
        gsa_k_min: int = 4,
        gsa_k_max: int = 32,
        gsa_indexer_heads: int = 4,
        gsa_indexer_dim: int = 32,
        num_global_tokens: int = 4,
        inter_page_window: int = 3,
        dropout: float = 0.1,
        flags: AblationFlags | None = None,
        vit_cache: Any | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_global_tokens = num_global_tokens
        self.inter_page_window = inter_page_window
        self.flags = flags or AblationFlags()
        flags = self.flags

        # ── Block Encoder ──
        self.block_encoder = BlockMultiModalEncoder(
            text_encoder_name=text_encoder_name,
            text_max_length=text_max_length,
            text_freeze=text_freeze,
            image_encoder_name=image_encoder_name,
            image_freeze=image_freeze,
            hidden_size=hidden_size,
            dropout=dropout,
            gated_fusion=flags.gated_fusion,
            vit_cache=vit_cache,
        )

        # ── Page Encoder ──
        self.page_encoder = PageEncoder(
            dim=hidden_size,
            num_heads=page_encoder_heads,
            num_layers=page_encoder_layers,
            k_base=gsa_k_base,
            k_min=gsa_k_min,
            k_max=gsa_k_max,
            indexer_heads=gsa_indexer_heads,
            indexer_dim=gsa_indexer_dim,
            num_global=num_global_tokens,
            dropout=dropout,
            flags=flags,
        )

        # ── Infini-attention Compressive Memory ──
        # A6 / A6+A7 drop the modules entirely rather than bypassing them.
        self.page_memory = (
            InfiniPageMemory(dim=hidden_size, dropout=dropout)
            if flags.page_memory
            else None
        )
        self.section_memory = (
            InfiniSectionMemory(
                dim=hidden_size,
                num_boundaries=num_boundaries,
                dropout=dropout,
                boundary_gate=flags.boundary_gate,
            )
            if flags.section_memory
            else None
        )

        # ── Inter-page Attention ──
        self.inter_page_attn = InterPageAttention(
            dim=hidden_size,
            num_heads=page_encoder_heads,
            window=inter_page_window,
            dropout=dropout,
            gated=flags.gated_interpage,
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
        image_blocks: list[Any],
        block_type_ids: torch.Tensor,      # [N] int
        bbox_norm: torch.Tensor,            # [N, 8]
        font_size: torch.Tensor,            # [N, 1]
        is_bold: torch.Tensor,              # [N, 1]
        is_italic: torch.Tensor,            # [N, 1]
        font_color: torch.Tensor,           # [N, 3]
        page_splits: list[int],             # cumulative block counts per page

        # Infini-attention state: (M, z) tuples
        page_M: torch.Tensor | None = None,       # [1, d, d]
        page_z: torch.Tensor | None = None,       # [1, d]
        section_M: torch.Tensor | None = None,    # [1, d, d]
        section_z: torch.Tensor | None = None,    # [1, d]
        page_history: torch.Tensor | None = None, # [1, T, dim]
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Process a document page-by-page, propagating compressed memory.

        Args:
            page_splits: cumulative block counts, e.g. [5, 12, 18] for 3 pages.

        Returns dict with:
            block_logits:    list of [1, Ni, num_classes] per page
            boundary_logits: list of [1, 4] per page
            section_logits:  list of [1, num_classes] per page
            page_reprs:      list of [1, dim] per page
            page_M:          final page memory [1, d, d]
            page_z:          final page normalization [1, d]
            section_M:       final section memory [1, d, d]
            section_z:       final section normalization [1, d]
            page_history:    recent page reprs [1, T', dim]
        """
        device = block_type_ids.device
        B = 1

        if self.page_memory is not None and page_M is None:
            page_M, page_z = self.page_memory.init_state(B, device)
        if self.section_memory is not None and section_M is None:
            section_M, section_z = self.section_memory.init_state(B, device)
        if page_history is None:
            page_history = torch.zeros(B, 0, self.hidden_size, device=device)

        block_logits_list: list[torch.Tensor] = []
        boundary_logits_list: list[torch.Tensor] = []
        section_logits_list: list[torch.Tensor] = []
        page_reprs_list: list[torch.Tensor] = []

        prev_end = 0
        for end in page_splits:
            # ── Extract page blocks ──
            p_texts = texts[prev_end:end]
            p_ocr = ocr_texts[prev_end:end]
            p_image_blocks = image_blocks[prev_end:end]
            # With an active frozen-ViT cache the block encoder resolves images
            # from the key index, so decoding every block up front would throw
            # away the point of the cache.
            if self.block_encoder.image_cache_active:
                p_images = None
            else:
                p_images = [
                    b.load_image() if b is not None else None
                    for b in p_image_blocks
                ]
            p_bt = block_type_ids[prev_end:end]
            p_bbox = bbox_norm[prev_end:end]
            p_fs = font_size[prev_end:end]
            p_bold = is_bold[prev_end:end]
            p_italic = is_italic[prev_end:end]
            p_color = font_color[prev_end:end]

            # ── Block Encoding ──
            block_vecs = self.block_encoder(
                texts=p_texts,
                ocr_texts=p_ocr,
                images=p_images,
                image_blocks=p_image_blocks,
                block_type_ids=p_bt,
                bbox_norm=p_bbox,
                font_size=p_fs,
                is_bold=p_bold,
                is_italic=p_italic,
                font_color=p_color,
            )  # [Ni, dim]
            block_vecs = block_vecs.unsqueeze(0)  # [1, Ni, dim]

            # ── Page Encoding (sparse attention) ──
            page_repr, encoded_blocks = self.page_encoder(block_vecs)

            # ── Inter-page Attention ──
            page_repr = self.inter_page_attn(page_repr, page_history)

            # ── Boundary Prediction ──
            bound_logits = self.boundary_head(page_repr)  # [1, 4]
            boundary_logits_list.append(bound_logits)

            # ── Infini Page Memory: compress + retrieve ──
            if self.page_memory is not None:
                enhanced_page, page_M, page_z = self.page_memory.step(
                    page_repr, page_M, page_z
                )
            else:
                enhanced_page = page_repr

            # ── Infini Section Memory: boundary-gated compress + retrieve ──
            if self.section_memory is not None:
                enhanced_section, section_M, section_z = self.section_memory.step(
                    page_repr, bound_logits, section_M, section_z
                )
            else:
                enhanced_section = torch.zeros_like(page_repr)

            # ── Combine memory readouts for classification ──
            # enhanced_page already has memory context; add section context too
            combined_page = enhanced_page + enhanced_section  # residual merge

            # ── Block Classification ──
            # Inject combined memory context into encoded blocks
            context_blocks = encoded_blocks + combined_page.unsqueeze(1)
            block_logits = self.classification_head(context_blocks)
            block_logits_list.append(block_logits)

            # ── Section Classification ──
            section_logits = self.section_head(combined_page, enhanced_section)
            section_logits_list.append(section_logits)

            # ── Track page reprs for inter-page attention ──
            page_reprs_list.append(combined_page)
            page_history = torch.cat([page_history, combined_page.unsqueeze(1)], dim=1)
            if page_history.size(1) > self.inter_page_window:
                page_history = page_history[:, -self.inter_page_window:, :]

            prev_end = end

        return {
            "block_logits": block_logits_list,
            "boundary_logits": boundary_logits_list,
            "section_logits": section_logits_list,
            "page_reprs": page_reprs_list,
            "page_M": page_M,
            "page_z": page_z,
            "section_M": section_M,
            "section_z": section_z,
            "page_history": page_history,
        }