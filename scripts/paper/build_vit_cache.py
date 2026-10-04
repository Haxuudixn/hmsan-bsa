"""Build the offline frozen-ViT feature cache (Phase 0, item E0.1).

Enumerates every image block in the corpus, decodes it exactly the way the
dataset loader does, runs the frozen ViT over it once and stores the 768-d
[CLS] feature.  Training then reads features from a memmap instead of paying
~21 ms of ViT forward plus ~14 ms of ZIP decode per image, every epoch, for
every architecture row that keeps the image branch frozen.

Run it once per corpus; the result is shared by all rows whose manifest entry
says ``"vit_cache": true``.  An existing complete cache is left alone unless
``--force`` is given.

    C:\\hmsan\\.venv\\Scripts\\python.exe scripts\\paper\\build_vit_cache.py

Smoke test without touching the GPU (a handful of images on CPU):

    ... build_vit_cache.py --limit 8 --device cpu --no-amp --out_dir cache_smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# ``_get_zip`` / ``_read_image`` are the loader's own primitives; reusing them
# (rather than re-implementing ZIP access here) is what guarantees the cache is
# keyed on the same member resolution and decoded the same way as training.
from bid_slicing.data.dataset import (  # noqa: E402
    _get_zip,
    _read_image,
    load_documents_from_zips,
)
from bid_slicing.data.vit_cache import (  # noqa: E402
    CACHE_VERSION,
    FEATURES_FILE,
    default_cache_dir,
    open_features_for_write,
    save_cache,
)
from bid_slicing.features.image_features import ViTImageEncoder  # noqa: E402

# Measured on the RTX 3060, 2026-09-21, used only for the closing estimate.
MEASURED_VIT_MS_PER_IMAGE = 21.0
MEASURED_DECODE_MS_PER_IMAGE = 14.3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data_dir",
        default=r"C:\Users\Administrator\Desktop\训练数据\final_train",
        help="directory holding training_data_*.zip exports",
    )
    parser.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "cache"),
        help="parent directory; the cache lands in <out_dir>/<model>-<precision>",
    )
    parser.add_argument("--model_name", default="google/vit-base-patch16-224")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="build under bf16 autocast, matching the live training path",
    )
    parser.add_argument(
        "--max_blocks",
        type=int,
        default=0,
        help="0 = no cap; keep 0 so the cache covers every capped training run",
    )
    parser.add_argument("--max_pages", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="stop after N keys (smoke test); marks the cache partial",
    )
    parser.add_argument("--force", action="store_true", help="rebuild over an existing cache")
    parser.add_argument("--log_every", type=int, default=20, help="batches between log lines")
    return parser.parse_args()


def enumerate_keys(data_dir: Path, max_blocks: int, max_pages: int):
    """Return ``(zip_paths, {key: (source_zip, member)})`` for every image block.

    The caps are deliberately left at 0 by default: ``_split_long_document``
    only regroups pages, it never drops blocks, so an uncapped enumeration is a
    superset of every capped training run's key set.
    """
    zip_paths = sorted(str(p) for p in data_dir.glob("training_data_*.zip"))
    if not zip_paths:
        raise SystemExit(f"no training_data_*.zip under {data_dir}")
    print(f"[build] {len(zip_paths)} ZIP exports under {data_dir}")

    documents = load_documents_from_zips(list(zip_paths), max_blocks, max_pages)
    blocks = sum(len(page.blocks) for doc in documents for page in doc.pages)
    print(f"[build] {len(documents)} segments, {blocks:,} blocks")

    keys: dict[str, tuple[str, str]] = {}
    for doc in documents:
        for page in doc.pages:
            for block in page.blocks:
                if block.block_type_id not in (1, 2):
                    continue
                key = block.image_cache_key
                if not key or key in keys:
                    continue
                keys[key] = (block.source_zip, block.image_member)
    print(f"[build] {len(keys):,} distinct image keys")
    return zip_paths, keys


def absent_keys(keys: dict[str, tuple[str, str]]) -> set[str]:
    """Keys whose member is not in its ZIP at all (known without decoding)."""
    absent: set[str] = set()
    names_cache: dict[str, set[str]] = {}
    for key, (zip_path, member) in keys.items():
        names = names_cache.get(zip_path)
        if names is None:
            names = set(_get_zip(zip_path).namelist())
            names_cache[zip_path] = names
        if member not in names:
            absent.add(key)
    return absent


def encode_keys(
    encoder,
    key_sources: dict[str, tuple[str, str]],
    keys: list[str],
    feature_path,
    batch_size: int,
    device: torch.device,
    amp: bool,
    log_every: int = 20,
    log_prefix: str = "[build]",
    failed: set[str] | None = None,
):
    """Fill the feature memmap for ``keys`` in order; return (rows, failed).

    Decoding happens inside the batch loop, so peak host memory stays at
    ``batch_size`` images even for the 84k-image corpus.  A key that fails to
    decode is added to ``failed`` and leaves a zero row, which
    ``ViTFeatureCache.is_failed`` masks at use time.
    """
    features = open_features_for_write(feature_path, len(keys), int(encoder.output_dim))
    failed = set() if failed is None else set(failed)
    rows = 0
    started = time.time()
    batches = max(1, (len(keys) + batch_size - 1) // batch_size)

    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(keys), batch_size)):
            images = []
            for key in keys[start:start + batch_size]:
                zip_path, member = key_sources[key]
                image = _read_image(zip_path, member)
                if image is None:
                    failed.add(key)
                    continue
                images.append(image)
            if images:
                use_amp = bool(amp) and device.type == "cuda"
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
                ):
                    batch_features = encoder(images)
                features[rows:rows + len(images)] = batch_features.float().cpu().numpy()
                rows += len(images)

            if (batch_index + 1) % log_every == 0 or batch_index + 1 == batches:
                elapsed = time.time() - started
                done = min(start + batch_size, len(keys))
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (len(keys) - done) / rate if rate > 0 else 0.0
                print(
                    f"{log_prefix} {done:,}/{len(keys):,} "
                    f"({100 * done / max(len(keys), 1):.1f}%) {rate:.1f} img/s "
                    f"elapsed {elapsed / 60:.1f} min eta {eta / 60:.1f} min",
                    flush=True,
                )

    features.flush()
    del features
    return rows, failed, time.time() - started


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"data dir missing: {data_dir}")

    cache_dir = default_cache_dir(args.out_dir, args.model_name, amp=bool(args.amp))
    feature_path = cache_dir / FEATURES_FILE
    if feature_path.is_file() and not args.force:
        print(f"[build] cache already present, nothing to do: {cache_dir}")
        print("[build] pass --force to rebuild it")
        return 0

    zip_paths, keys = enumerate_keys(data_dir, args.max_blocks, args.max_pages)
    ordered = sorted(keys)
    if args.limit > 0:
        ordered = ordered[: args.limit]
        print(f"[build] --limit {args.limit}: building a PARTIAL cache")

    absent = absent_keys({key: keys[key] for key in ordered})
    present = [key for key in ordered if key not in absent]
    print(f"[build] {len(present):,} decodable, {len(absent):,} missing members")
    if not present:
        raise SystemExit("no decodable images; wrong --data_dir?")

    device = torch.device(args.device)
    print(f"[build] device={device} amp={bool(args.amp)} batch={args.batch_size}")
    encoder = ViTImageEncoder(args.model_name, freeze=True).to(device).eval()

    rows, failed, elapsed = encode_keys(
        encoder,
        keys,
        present,
        feature_path,
        args.batch_size,
        device,
        bool(args.amp),
        log_every=args.log_every,
        failed=absent,
    )

    save_cache(
        cache_dir,
        {
            "version": CACHE_VERSION,
            "model_name": args.model_name,
            "hidden_size": int(encoder.output_dim),
            "dtype": "float32",
            "amp": bool(args.amp),
            "count": rows,
            "failed": len(failed),
            "partial": bool(args.limit > 0) or rows != len(present),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "build_seconds": round(elapsed, 1),
            "source": {
                "data_dir": str(data_dir),
                "zip_count": len(zip_paths),
                "zips": [Path(p).name for p in zip_paths],
                "max_blocks_per_sample": args.max_blocks,
                "max_pages_per_sample": args.max_pages,
                "limit": args.limit,
            },
            "note": (
                "Frozen-ViT [CLS] features. Rows are float32; under bf16 the "
                "live path produces the same values to within one rounding "
                "step, and a bf16 build reproduces them exactly."
            ),
        },
        [key for key in present if key not in failed],
        failed,
    )

    saved_ms = (MEASURED_VIT_MS_PER_IMAGE + MEASURED_DECODE_MS_PER_IMAGE) * rows
    print(f"[build] wrote {rows:,} rows to {cache_dir}")
    print(f"[build] {len(failed):,} undecodable key(s); build took {elapsed / 60:.1f} min")
    print(
        f"[build] estimated saving: {saved_ms / 60000:.1f} min per epoch "
        f"({saved_ms * 120 / 3600000:.1f} h across 120 epochs of the 15-row matrix)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())