from __future__ import annotations

from enum import IntEnum
from typing import Any

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
        gated_fusion: bool = True,
        vit_cache: "Any | None" = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.gated_fusion = gated_fusion
        # Frozen-ViT feature cache (bid_slicing.data.vit_cache).  Cached
        # features are constant by construction, so the cache is dropped rather
        # than silently serving stale values once the ViT takes gradients.
        self.vit_cache = vit_cache if (vit_cache is not None and image_freeze) else None
        self.vit_cache_misses = 0

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
        if gated_fusion:
            self.fusion_gate = nn.Sequential(
                nn.Linear(768 + layout_output_dim, hidden_size),
                nn.Sigmoid(),
            )
        else:
            # A4: plain concat + Linear, no learned gate.
            self.fusion_concat = nn.Linear(
                768 + layout_output_dim, hidden_size
            )
        self.fusion_norm = nn.LayerNorm(hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)

    def _encode_text_chunked(self, texts: list[str], chunk_size: int = 32) -> torch.Tensor:
        """Encode text blocks in chunks to limit peak GPU memory.

        Blocks are sorted by character length before batching, so every chunk
        holds similarly sized sequences and padding stays small.  Results are
        scattered back to the original order afterwards.  A long document is
        dominated by padding waste otherwise: block texts average ~14 tokens,
        but a mixed-length chunk of 16 pads every row to the longest one.
        """
        if len(texts) <= chunk_size:
            return self.text_encoder(texts)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        outputs = []
        for start in range(0, len(order), chunk_size):
            idx = order[start:start + chunk_size]
            outputs.append(self.text_encoder([texts[i] for i in idx]))
        sorted_out = torch.cat(outputs, dim=0)
        inverse = [0] * len(order)
        for position, original in enumerate(order):
            inverse[original] = position
        restore = torch.as_tensor(inverse, device=sorted_out.device)
        return sorted_out[restore]

    @property
    def image_cache_active(self) -> bool:
        """True when image features come from the cache (no ZIP decode)."""
        return self.vit_cache is not None

    def _block_image(self, index: int, images, image_blocks):
        """PIL image for one block, from whichever source is available.

        A supplied ``images`` list wins outright: ``None`` in it means "this
        block has no usable image", so a decoded batch is never re-decoded.
        Decoding on demand happens only when no list was passed, which is the
        cached run's fast path.
        """
        if images is not None:
            return images[index]
        if image_blocks is not None and image_blocks[index] is not None:
            return image_blocks[index].load_image()
        return None

    def _has_image(self, index: int, images, image_blocks) -> bool:
        """Is a visual feature available for this block?

        With an active cache the answer comes from the key index, so a warm run
        never pays the ~14 ms ZIP read plus PNG decode.  Keys that are neither
        cached nor known-bad are still resolved with a real decode, so a
        truncated or stale cache can only cost speed, never change a result.

        That fallback decodes the image here to learn whether it exists at all,
        and the encode path may decode it a second time; an incomplete cache
        therefore costs up to two decodes per miss.  A complete cache (what
        ``verify_vit_cache.py`` checks) never enters this branch.
        """
        block = image_blocks[index] if image_blocks is not None else None
        if self.vit_cache is not None and block is not None:
            key = block.image_cache_key
            if key and self.vit_cache.has(key):
                return True
            if key and self.vit_cache.is_failed(key):
                return False
            if key:
                # Unknown key: resolve with a real decode, so a truncated or
                # stale cache can cost speed but never change a result.
                self.vit_cache_misses += 1
            return block.load_image() is not None
        return self._block_image(index, images, image_blocks) is not None

    def _vit_features(self, indices, images, image_blocks, device) -> torch.Tensor:
        """ViT [CLS] features ``[len(indices), 768]`` for the given blocks."""
        if self.vit_cache is not None and image_blocks is not None:
            keys = [image_blocks[i].image_cache_key for i in indices]
            if all(keys):
                cached = self.vit_cache.get_many(keys, device=device)
                if cached is not None:
                    return cached
        return self._encode_image_chunked(
            [self._block_image(i, images, image_blocks) for i in indices]
        )

    def _encode_image_chunked(self, images: list[Image.Image], chunk_size: int = 16) -> torch.Tensor:
        """Encode image blocks in small chunks to limit peak GPU memory.

        Reached only on a cache miss; a complete cache skips this path.
        """
        outputs = []
        for start in range(0, len(images), chunk_size):
            outputs.append(self.image_encoder(images[start:start + chunk_size]))
        return torch.cat(outputs, dim=0)

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
        image_blocks: list[Any] | None = None,
    ) -> torch.Tensor:
        """Return block vectors [B, hidden_size].

        ``image_blocks`` carries the dataset ``Block`` objects, which is what
        lets a cached run skip decoding entirely; ``images`` stays the
        already-decoded path used by callers that pass PIL images.
        """
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
            modality_feat[text_idx] = self._encode_text_chunked(
                [texts[i] for i in text_idx]
            )

        # ── Image branch ──
        if image_idx:
            has_image = {i: self._has_image(i, images, image_blocks) for i in image_idx}
            ocr_image_idx = [
                i for i in image_idx
                if has_image[i] and (ocr_texts[i] or "").strip()
            ]
            visual_image_idx = [
                i for i in image_idx
                if has_image[i] and not (ocr_texts[i] or "").strip()
            ]
            ocr_only_image_idx = [
                i for i in image_idx
                if not has_image[i] and (ocr_texts[i] or "").strip()
            ]

            # Image with OCR text: fuse RoBERTa(ocr_text) + ViT(image).
            if ocr_image_idx:
                ocr_emb = self._encode_text_chunked(
                    [ocr_texts[i] for i in ocr_image_idx]
                )
                img_emb = self._vit_features(
                    ocr_image_idx, images, image_blocks, device
                )
                modality_feat[ocr_image_idx] = self.mixed_proj(
                    torch.cat([ocr_emb, img_emb], dim=-1)
                )

            # Image without OCR text: keep pure ViT branch.
            if visual_image_idx:
                modality_feat[visual_image_idx] = self._vit_features(
                    visual_image_idx, images, image_blocks, device
                )

            # OCR text available but image missing: use text-only fallback.
            if ocr_only_image_idx:
                modality_feat[ocr_only_image_idx] = self._encode_text_chunked(
                    [ocr_texts[i] for i in ocr_only_image_idx]
                )

        # ── Mixed branch ──
        if mixed_idx:
            ocr_emb = self._encode_text_chunked(
                [ocr_texts[i] if ocr_texts[i] else '' for i in mixed_idx]
            )
            # Resolved once per block: asking twice would double-count a cache
            # miss, and with an active cache ``images`` is None, so the old
            # `images[i] is None` probe below would raise TypeError.
            mixed_has_image = {
                i: self._has_image(i, images, image_blocks) for i in mixed_idx
            }
            valid_mixed = [i for i in mixed_idx if mixed_has_image[i]]
            if valid_mixed:
                img_emb = self._vit_features(
                    valid_mixed, images, image_blocks, device
                )
                # For blocks with images: full OCR+Image fusion
                ocr_sub = ocr_emb[[mixed_idx.index(i) for i in valid_mixed]]
                mixed_emb = self.mixed_proj(torch.cat([ocr_sub, img_emb], dim=-1))
                modality_feat[valid_mixed] = mixed_emb
            # For MIXED blocks without images: fall back to text-only
            no_img_mixed = [i for i in mixed_idx if not mixed_has_image[i]]
            if no_img_mixed:
                modality_feat[no_img_mixed] = ocr_emb[[mixed_idx.index(i) for i in no_img_mixed]]

        # ── Gated Fusion ──
        m_proj = self.modality_proj(modality_feat)                 # [B, H]
        l_proj = self.layout_proj(layout_feat)                     # [B, H]
        if self.gated_fusion:
            gate = self.fusion_gate(torch.cat([modality_feat, layout_feat], dim=-1))
            fused = gate * m_proj + (1.0 - gate) * l_proj          # [B, H]
        else:
            fused = self.fusion_concat(
                torch.cat([modality_feat, layout_feat], dim=-1)
            )                                                      # [B, H]
        return self.fusion_dropout(F.relu(self.fusion_norm(fused)))
