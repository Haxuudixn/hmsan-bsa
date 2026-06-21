from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class RoBERTaTextEncoder(nn.Module):
    """Chinese RoBERTa-wwm-ext encoder with head+tail truncation for long texts."""

    def __init__(
        self,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        max_length: int = 510,
        output_dim: int = 768,
        freeze: bool = False,
    ):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.output_dim = output_dim
        self.head_tail_ratio = 0.5  # split equally for head+tail truncation

    def _truncate_text(self, text: str) -> str:
        """Head (first half) + tail (last half) truncation for texts exceeding max_length."""
        max_chars = self.max_length - 2  # reserve for [CLS] and [SEP]
        if len(text) <= max_chars:
            return text
        head_len = max_chars // 2
        tail_len = max_chars - head_len
        return text[:head_len] + text[-tail_len:]

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of text strings → [batch, output_dim] (CLS pooling)."""
        truncated = [self._truncate_text(t) for t in texts]
        tokens = self.tokenizer(
            truncated,
            padding=True,
            truncation=True,
            max_length=self.max_length + 2,
            return_tensors="pt",
        )
        device = next(self.encoder.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        outputs = self.encoder(**tokens)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device