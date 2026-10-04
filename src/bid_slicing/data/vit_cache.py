"""Offline cache for frozen ViT block features (Phase 0, item E0.1).

When ``image_freeze=True`` the image branch is a constant function of the input
image, so its 768-d [CLS] feature can be computed once and reused by every
epoch of every architecture row that keeps the ViT frozen.

Two measured costs disappear with the cache (RTX 3060 12 GB, 2026-09-21):

* ViT forward, batch 32, bf16: 21.0 ms per image
* image decode out of the export ZIP: 14.3 ms per image with a warm handle,
  128.5 ms when ``ZipFile.open`` has to re-read the central directory

The corpus holds 84,806 image blocks (12.6% of 456,602 blocks); per training
epoch that is roughly 35 min of the measured 2.54 h, i.e. ~23%.

On-disk layout, one directory per (model name, precision) pair::

    <root>/index.json        metadata: model, precision, counts, provenance
    <root>/keys.txt          one ``<zip-basename>::<member>`` per line,
                             line number == row number in the feature matrix
    <root>/failed.txt        keys whose member is absent or will not decode
    <root>/features.f32.npy  raw (count, hidden) float32, read via memmap

``keys.txt`` is a plain text file rather than a JSON map on purpose: 84,806
keys parse in well under a second and the file stays greppable, whereas the
equivalent JSON object is ~3.4 MB of quoted strings held in memory.

Rows are stored **float32** and cast at use.  The live path under bf16 autocast
produces bf16 activations that are immediately assigned into a float32
``modality_feat`` buffer, so the cached value is the same number to within one
bf16 rounding step; ``bf16`` builds store exactly what the live path produced.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

CACHE_VERSION = 1
META_FILE = "index.json"
KEYS_FILE = "keys.txt"
FAILED_FILE = "failed.txt"
FEATURES_FILE = "features.f32.npy"


def block_key(zip_path: str, member: str) -> str:
    """Stable cache key for one image: ``<zip file name>::<member>``.

    The ZIP *file name* is used instead of the full path so a cache built on
    one machine stays valid after the corpus is moved; ``index.json`` records
    the absolute paths the cache was built from.
    """
    if not zip_path or not member:
        return ""
    return f"{Path(zip_path).name}::{member}"


def default_cache_dir(base: str | Path, model_name: str, amp: bool = True) -> Path:
    """Cache directory for one (encoder, precision) pair."""
    tag = model_name.replace("/", "__")
    return Path(base) / f"{tag}-{'bf16' if amp else 'fp32'}"


class ViTFeatureCache:
    """Read-only view of an offline ViT feature cache."""

    def __init__(
        self,
        root: Path,
        meta: dict,
        key_to_row: dict[str, int],
        failed: set[str],
        features: np.ndarray,
    ) -> None:
        self.root = Path(root)
        self.meta = meta
        self._key_to_row = key_to_row
        self._failed = failed
        self._features = features

    # ── construction ──

    @classmethod
    def open(cls, root: str | Path) -> "ViTFeatureCache":
        root = Path(root)
        meta_path = root / META_FILE
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"no ViT feature cache at {root} (missing {META_FILE}); "
                f"build one with scripts/paper/build_vit_cache.py"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        version = int(meta.get("version", 0))
        if version != CACHE_VERSION:
            raise RuntimeError(
                f"cache {root} has version {version}, this code expects "
                f"{CACHE_VERSION}; rebuild it"
            )

        keys = [
            line for line in
            (root / KEYS_FILE).read_text(encoding="utf-8").splitlines()
            if line
        ]
        failed_path = root / FAILED_FILE
        failed = set()
        if failed_path.is_file():
            failed = {
                line for line in
                failed_path.read_text(encoding="utf-8").splitlines() if line
            }

        features = np.load(root / FEATURES_FILE, mmap_mode="r")
        hidden_size = int(meta["hidden_size"])
        if tuple(features.shape) != (len(keys), hidden_size):
            raise RuntimeError(
                f"cache {root} is inconsistent: features{tuple(features.shape)} "
                f"but {len(keys)} keys x {hidden_size}-d"
            )

        key_to_row = {key: row for row, key in enumerate(keys)}
        if len(key_to_row) != len(keys):
            raise RuntimeError(f"cache {root} has duplicate keys in {KEYS_FILE}")
        return cls(root, meta, key_to_row, failed, features)

    # ── metadata ──

    @property
    def count(self) -> int:
        return len(self._key_to_row)

    @property
    def failed_count(self) -> int:
        return len(self._failed)

    @property
    def hidden_size(self) -> int:
        return int(self.meta.get("hidden_size", 0))

    @property
    def model_name(self) -> str:
        return str(self.meta.get("model_name", ""))

    @property
    def amp(self) -> bool:
        """Was the cache built under bf16 autocast (i.e. matching training)?"""
        return bool(self.meta.get("amp", True))

    @property
    def partial(self) -> bool:
        """True when the build was truncated, so misses fall back to the GPU."""
        return bool(self.meta.get("partial", False))

    def to_dict(self) -> dict:
        """Compact record for run_meta.json / checkpoints."""
        return {
            "dir": str(self.root),
            "count": self.count,
            "failed": self.failed_count,
            "model_name": self.model_name,
            "hidden_size": self.hidden_size,
            "amp": self.amp,
            "partial": self.partial,
            "built_at": self.meta.get("built_at"),
        }

    def describe(self) -> str:
        parts = [
            f"{self.count} images",
            f"{self.hidden_size}-d",
            "bf16-built" if self.amp else "fp32-built",
            str(self.root),
        ]
        if self.failed_count:
            parts.append(f"{self.failed_count} undecodable")
        if self.partial:
            parts.append("PARTIAL - misses fall back to a live ViT forward")
        return ", ".join(parts)

    # ── lookup ──

    def has(self, key: str) -> bool:
        return bool(key) and key in self._key_to_row

    def is_failed(self, key: str) -> bool:
        return bool(key) and key in self._failed

    def row(self, key: str) -> int | None:
        return self._key_to_row.get(key)

    def get_many(
        self,
        keys: list[str],
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor | None:
        """Return ``[len(keys), hidden_size]`` features, or None if unavailable.

        Returning None rather than a zero row is deliberate: a truncated or
        stale cache can then only cost time (the caller falls back to a live
        forward), never silently change the model's input.
        """
        rows = [self._key_to_row.get(key) for key in keys]
        if not rows or any(row is None for row in rows):
            return None
        # Fancy-index the memmap: only the requested rows are read from disk.
        array = np.asarray(self._features[rows], dtype=np.float32)
        tensor = torch.from_numpy(array)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor.to(dtype=dtype)


def open_features_for_write(path: str | Path, count: int, hidden_size: int):
    """Allocate the feature matrix as an on-disk memmap and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(count, hidden_size)
    )


def save_cache(root: str | Path, meta: dict, keys: list[str], failed) -> None:
    """Write ``index.json`` / ``keys.txt`` / ``failed.txt`` for a cache build."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / KEYS_FILE).write_text("\n".join(keys) + "\n", encoding="utf-8")
    failed_sorted = sorted(failed)
    (root / FAILED_FILE).write_text(
        "\n".join(failed_sorted) + ("\n" if failed_sorted else ""),
        encoding="utf-8",
    )
    (root / META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )