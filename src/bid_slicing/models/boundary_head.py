from __future__ import annotations

import torch
import torch.nn as nn


class BoundaryHead(nn.Module):
    """Predict B-SECTION / I-SECTION / E-SECTION / O from page representations."""

    def __init__(self, dim: int = 128, num_boundaries: int = 4, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_boundaries),
        )

    def forward(self, page_repr: torch.Tensor) -> torch.Tensor:
        """Returns boundary logits [B, 4]."""
        return self.classifier(page_repr)


class ClassificationHead(nn.Module):
    """Classify each block into its section label."""

    def __init__(self, dim: int = 128, num_classes: int = 19, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, block_vectors: torch.Tensor) -> torch.Tensor:
        """Returns class logits [B, num_classes] or [B, N, num_classes]."""
        return self.classifier(block_vectors)


class SectionHead(nn.Module):
    """Predict section label from page-level context."""

    def __init__(self, dim: int = 128, num_classes: int = 19, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),  # page_repr + section_readout
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(
        self,
        page_repr: torch.Tensor,
        section_readout: torch.Tensor,
    ) -> torch.Tensor:
        """Returns section logits [B, num_classes]."""
        return self.classifier(torch.cat([page_repr, section_readout], dim=-1))