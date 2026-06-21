from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from PIL import Image


class ViTImageEncoder(nn.Module):
    """ViT-B/16 image encoder with CLS pooling."""

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        output_dim: int = 768,
        freeze: bool = False,
    ):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.output_dim = output_dim

    def forward(self, images: list[Image.Image]) -> torch.Tensor:
        """Encode a list of PIL Images → [batch, output_dim] (CLS pooling)."""
        inputs = self.processor(images=images, return_tensors="pt")
        device = next(self.encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.encoder(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device