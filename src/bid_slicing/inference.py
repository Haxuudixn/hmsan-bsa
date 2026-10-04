"""Inference helpers for HMSAN-BSA.

The predictor applies the same PaddleOCR preprocessing rule used in training:
image blocks without OCR text are OCR'd with PaddleOCR, and image blocks with
non-empty OCR text are treated as MIXED by the dataset loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from bid_slicing.data.dataset import collate_document, load_documents_from_zips
from bid_slicing.data.label_schema import LabelSchema
from bid_slicing.features.ocr import extract_text
from bid_slicing.models.hmsan_bsa import HMSAN_BSA


class HMSANPredictor:
    """Load a trained HMSAN-BSA checkpoint and run document-level inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        label_config: str | Path = "configs/labels.yaml",
        device: str = "cuda",
        dropout: float = 0.2,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.label_schema = LabelSchema.from_yaml(label_config)

        self.model = HMSAN_BSA(
            text_encoder_name="hfl/chinese-roberta-wwm-ext",
            text_max_length=510,
            text_freeze=False,
            image_encoder_name="google/vit-base-patch16-224",
            image_freeze=True,
            hidden_size=128,
            num_classes=self.label_schema.num_classes,
            num_boundaries=self.label_schema.num_boundaries,
            page_encoder_layers=2,
            page_encoder_heads=4,
            gsa_k_base=16,
            gsa_k_min=4,
            gsa_k_max=32,
            gsa_indexer_heads=4,
            gsa_indexer_dim=32,
            num_global_tokens=4,
            inter_page_window=3,
            dropout=dropout,
        ).to(self.device)

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def ensure_ocr(self, documents: list[Any], use_gpu: bool = False) -> list[Any]:
        """Run PaddleOCR on image blocks that do not yet have OCR text."""
        for doc in documents:
            for page in doc.pages:
                for block in page.blocks:
                    if (block.block_type or "").strip().lower() != "image":
                        continue
                    if (block.ocr_text or "").strip():
                        continue
                    image = block.load_image()
                    if image is None:
                        continue
                    try:
                        block.ocr_text = extract_text(
                            image,
                            use_gpu=use_gpu,
                            prefer_paddleocr=True,
                        )
                    except Exception:
                        block.ocr_text = ""
        return documents

    @torch.no_grad()
    def predict_documents(
        self,
        documents: list[Any],
        use_gpu: bool = False,
    ) -> list[dict[str, Any]]:
        """Predict labels for all blocks in a list of documents."""
        self.ensure_ocr(documents, use_gpu=use_gpu)
        results: list[dict[str, Any]] = []

        for doc in documents:
            batch = collate_document([doc])
            output = self.model(
                texts=batch["texts"],
                ocr_texts=batch["ocr_texts"],
                image_blocks=batch["image_blocks"],
                block_type_ids=batch["block_type_ids"].to(self.device),
                bbox_norm=batch["bbox_norm"].to(self.device),
                font_size=batch["font_size"].to(self.device),
                is_bold=batch["is_bold"].to(self.device),
                is_italic=batch["is_italic"].to(self.device),
                font_color=batch["font_color"].to(self.device),
                page_splits=batch["page_splits"],
            )

            block_predictions: list[tuple[int, float]] = []
            for block_logits in output["block_logits"]:
                probs = torch.softmax(block_logits[0], dim=-1)
                confidence, class_id = probs.max(dim=-1)
                block_predictions.extend(
                    zip(
                        class_id.cpu().tolist(),
                        confidence.cpu().tolist(),
                    )
                )

            pred_blocks: list[dict[str, Any]] = []
            cursor = 0
            for page in doc.pages:
                for block in page.blocks:
                    class_id, confidence = block_predictions[cursor]
                    cursor += 1
                    pred_blocks.append({
                        "block_id": block.block_id,
                        "page_num": block.page_num,
                        "block_type": block.block_type,
                        "predicted_label": self.label_schema.decode_class(class_id),
                        "confidence": round(float(confidence), 6),
                    })

            results.append({
                "pdf_name": doc.pdf_name,
                "blocks": pred_blocks,
            })

        return results

    def predict_zips(self, zip_paths: list[str], use_gpu: bool = False) -> list[dict[str, Any]]:
        """Load annotated ZIPs, OCR missing image blocks, and predict labels."""
        documents = load_documents_from_zips(zip_paths)
        return self.predict_documents(documents, use_gpu=use_gpu)

    def predict_to_json(
        self,
        documents: list[Any],
        output_path: str | Path,
        use_gpu: bool = False,
    ) -> list[dict[str, Any]]:
        results = self.predict_documents(documents, use_gpu=use_gpu)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return results
