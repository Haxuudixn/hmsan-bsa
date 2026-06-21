from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bid_slicing.utils.io import read_yaml


@dataclass(frozen=True)
class LabelSchema:
    class_labels: list[str]
    boundary_labels: list[str]
    ignore_index: int = -100
    mixed_section_label: str = "MIXED"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LabelSchema":
        cfg: dict[str, Any] = read_yaml(path)
        return cls(
            class_labels=list(cfg["class_labels"]),
            boundary_labels=list(cfg["boundary_labels"]),
            ignore_index=int(cfg.get("ignore_index", -100)),
            mixed_section_label=str(cfg.get("mixed_section_label", "MIXED")),
        )

    @property
    def class_to_id(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.class_labels)}

    @property
    def boundary_to_id(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.boundary_labels)}

    @property
    def num_classes(self) -> int:
        return len(self.class_labels)

    @property
    def num_boundaries(self) -> int:
        return len(self.boundary_labels)

    def encode_class(self, label: str | None) -> int:
        if label is None:
            return self.ignore_index
        text = str(label).strip()
        if not text:
            return self.ignore_index
        return self.class_to_id.get(text, self.ignore_index)

    def encode_boundary(self, label: str | None) -> int:
        if label is None:
            return self.boundary_to_id["O"]
        return self.boundary_to_id.get(str(label).strip(), self.boundary_to_id["O"])

    def decode_class(self, label_id: int) -> str:
        if label_id == self.ignore_index:
            return ""
        return self.class_labels[label_id]

    def decode_boundary(self, boundary_id: int) -> str:
        return self.boundary_labels[boundary_id]
