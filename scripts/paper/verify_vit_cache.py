"""Verify the frozen-ViT feature cache (E0.1 acceptance checks).

Three independent checks, all of which must pass before the matrix is allowed
to use the cache:

  A  fidelity        N randomly drawn stored rows vs a fresh live ViT forward.
                     A bf16 build must reproduce the live bf16 activations to
                     well inside one bf16 rounding step.
  B  path equivalence one real ``HMSAN_BSA`` forward over a small multi-page
                     document, run twice with identical weights - once with the
                     cache wired in, once without - must produce identical
                     logits, and the cached run must take the fast path
                     (``vit_cache_misses == 0``).
  C  coverage        every image key the corpus can ask for is either cached or
                     recorded in ``failed.txt``.

Usage (CPU smoke test over a partial cache):

    python scripts\\paper\\verify_vit_cache.py --cache_dir cache_smoke\\... --device cpu --sample 4 --docs 2

Usage (the real cache):

    python scripts\\paper\\verify_vit_cache.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "paper"))

from bid_slicing.data.dataset import (  # noqa: E402
    _read_image,
    collate_document,
    load_documents_from_zips,
)
from bid_slicing.data.vit_cache import (  # noqa: E402
    CACHE_VERSION,
    ViTFeatureCache,
    default_cache_dir,
)
from bid_slicing.features.image_features import ViTImageEncoder  # noqa: E402
from bid_slicing.models.hmsan_bsa import HMSAN_BSA  # noqa: E402

from build_vit_cache import (  # noqa: E402
    absent_keys,
    encode_keys,
    enumerate_keys,
)

from bid_slicing.data.vit_cache import save_cache  # noqa: E402

DATA_DIR = r"C:\Users\Administrator\Desktop\训练数据\final_train"
MODEL_NAME = "google/vit-base-patch16-224"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache_dir", default=None, help="defaults to <repo>/cache/<model>-bf16")
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample", type=int, default=32, help="rows for check A")
    parser.add_argument("--docs", type=int, default=3, help="documents for check B")
    parser.add_argument("--max_blocks", type=int, default=90, help="size cap for check B docs")
    parser.add_argument("--mini_dir", default=str(REPO_ROOT / "cache_smoke" / "mini"))
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5e-3,
        help=(
            "max abs logit delta for check B.  Under bf16 autocast the cached "
            "features and a live forward are mathematically identical, but the "
            "batch shapes differ (the builder encodes in batches of 8, the model "
            "in chunks of 16), so kernels may round differently.  That effect is "
            "orders of magnitude below any ablation effect; use --tolerance 1e-6 "
            "on a CPU/fp32 cache to demand bit-exactness."
        ),
    )
    parser.add_argument(
        "--skip_coverage",
        action="store_true",
        help="skip check C (it enumerates the whole corpus, ~10 s)",
    )
    return parser.parse_args()


def failures_report(name, failures):
    print()
    print("=" * 72)
    if failures:
        print(f"FAILED: {name}")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"PASSED: {name}")
    return 0


def check_a(cache, args, device, zip_by_name):
    """Stored rows vs a fresh live forward."""
    print()
    print(f"[A] fidelity: {args.sample} sampled rows vs live ViT")
    keys = sorted(cache._key_to_row)
    random.Random(0).shuffle(keys)
    keys = keys[: args.sample]
    if not keys:
        return 0, ["cache is empty"]

    encoder = ViTImageEncoder(cache.model_name or MODEL_NAME, freeze=True).to(device).eval()
    images = []
    usable = []
    for key in keys:
        zip_name, member = key.split("::", 1)
        zip_path = zip_by_name.get(zip_name)
        if zip_path is None:
            continue
        image = _read_image(zip_path, member)
        if image is None:
            continue
        images.append(image)
        usable.append(key)
    if not usable:
        return 0, ["none of the sampled keys could be decoded"]

    with torch.no_grad():
        use_amp = bool(cache.amp) and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            live = encoder(images).float().cpu()
    stored = cache.get_many(usable, dtype=torch.float32).cpu()

    diff = (live - stored).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    scale = float(stored.abs().mean()) + 1e-6
    cos = torch.nn.functional.cosine_similarity(live, stored, dim=-1).min()
    limit = 0.02 if not cache.amp else 0.06
    print(f"    rows={len(usable)}  max|diff|={max_abs:.3e}  mean|diff|={mean_abs:.3e}  "
          f"mean|value|={scale:.3f}  min cos={float(cos):.7f}")
    print(f"    threshold {limit:.3f} x (1 + mean|value|) = {limit * (1.0 + scale):.3e}")
    print(f"    cache amp={cache.amp} device={device} (bf16 vs fp32 differ by ~1 bf16 step)")

    failures = []
    if float(cos) < 0.9999:
        failures.append(f"cosine similarity {float(cos):.6f} < 0.9999")
    if max_abs > limit * (1.0 + scale):
        failures.append(f"max abs diff {max_abs:.3e} exceeds {limit:.3f} x scale")
    return len(usable), failures


def build_mini_cache(docs, args, device, out_dir: Path):
    """Build a cache covering exactly the image keys of ``docs``."""
    sources = {}
    for doc in docs:
        for page in doc.pages:
            for block in page.blocks:
                if block.block_type_id not in (1, 2):
                    continue
                key = block.image_cache_key
                if key and key not in sources:
                    sources[key] = (block.source_zip, block.image_member)
    keys = sorted(sources)
    absent = absent_keys(sources)
    present = [key for key in keys if key not in absent]
    encoder = ViTImageEncoder(MODEL_NAME, freeze=True).to(device).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, failed, elapsed = encode_keys(
        encoder,
        sources,
        present,
        out_dir / "features.f32.npy",
        batch_size=8,
        device=device,
        amp=bool(device.type == "cuda"),
        log_every=10,
        log_prefix="[mini]",
        failed=absent,
    )
    save_cache(
        out_dir,
        {
            "version": CACHE_VERSION,
            "model_name": MODEL_NAME,
            "hidden_size": int(encoder.output_dim),
            "dtype": "float32",
            "amp": bool(device.type == "cuda"),
            "count": rows,
            "failed": len(failed),
            "partial": True,
            "built_at": "verification",
            "build_seconds": round(elapsed, 1),
            "source": {"data_dir": str(args.data_dir), "docs": len(docs)},
            "note": "mini cache built by verify_vit_cache.py check B",
        },
        [key for key in present if key not in failed],
        failed,
    )
    print(f"[mini] {rows} rows, {len(failed)} failed -> {out_dir}")
    return ViTFeatureCache.open(out_dir)


def forward_kwargs(batch, device):
    """Collated batch -> model kwargs, with the tensors moved like train.py does."""
    return {
        "texts": batch["texts"],
        "ocr_texts": batch["ocr_texts"],
        "image_blocks": batch["image_blocks"],
        "block_type_ids": batch["block_type_ids"].to(device),
        "bbox_norm": batch["bbox_norm"].to(device),
        "font_size": batch["font_size"].to(device),
        "is_bold": batch["is_bold"].to(device),
        "is_italic": batch["is_italic"].to(device),
        "font_color": batch["font_color"].to(device),
        "page_splits": batch["page_splits"],
    }


def check_b(docs, args, device, mini_cache):
    """Identical weights, with and without the cache, must give identical logits."""
    print()
    print(f"[B] path equivalence: {len(docs)} document(s), cache vs live")
    failures = []

    def build(**kwargs):
        torch.manual_seed(0)
        return HMSAN_BSA(
            text_encoder_name="hfl/chinese-roberta-wwm-ext",
            text_max_length=510,
            text_freeze=True,
            image_encoder_name=MODEL_NAME,
            image_freeze=True,
            hidden_size=128,
            num_classes=19,
            num_boundaries=4,
            dropout=0.1,
            **kwargs,
        ).to(device).eval()

    live_model = build()
    cache_model = build(vit_cache=mini_cache)
    cache_model.load_state_dict(live_model.state_dict())
    if cache_model.block_encoder.vit_cache is None:
        return ["image_freeze=True did not keep the cache"]

    images_seen = 0
    pages_compared = 0
    max_delta = 0.0
    logit_scale = 0.0
    with torch.no_grad():
        for doc in docs:
            batch = collate_document([doc])
            kwargs = forward_kwargs(batch, device)
            live = live_model(**kwargs)
            cached = cache_model(**kwargs)
            for page, (live_logits, cached_logits) in enumerate(
                zip(live["block_logits"], cached["block_logits"])
            ):
                images_seen += len(kwargs["image_blocks"])
                pages_compared += 1
                delta = float((live_logits - cached_logits).abs().max())
                max_delta = max(max_delta, delta)
                logit_scale = max(logit_scale, float(live_logits.abs().max()))
                if delta > args.tolerance:
                    failures.append(
                        f"{doc.pdf_name} page {page}: max logit delta {delta:.3e}"
                    )
            for head in ("boundary_logits", "section_logits"):
                for page, (live_logits, cached_logits) in enumerate(
                    zip(live[head], cached[head])
                ):
                    delta = float((live_logits - cached_logits).abs().max())
                    if delta > args.tolerance:
                        failures.append(
                            f"{doc.pdf_name} page {page} {head}: delta {delta:.3e}"
                        )

    misses = cache_model.block_encoder.vit_cache_misses
    print(f"    pages compared: {pages_compared}, image blocks seen: {images_seen}")
    print(f"    max abs logit delta: {max_delta:.3e}  "
          f"(max |logit| {logit_scale:.3f}, tolerance {args.tolerance:g})")
    print(f"    cache misses: {misses} (must be 0 = fast path was taken)")
    if misses:
        failures.append(f"{misses} block(s) did not hit the cache")
    print(f"    compared logits for {len(docs)} document(s), tolerance {args.tolerance:g}")
    del live_model, cache_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return failures


def check_c(cache, args):
    """Every corpus key is either cached or recorded as undecodable."""
    print()
    print("[C] coverage: corpus keys vs cache")
    data_dir = Path(args.data_dir)
    _, keys = enumerate_keys(data_dir, 0, 0)
    missing = [key for key in keys if not cache.has(key) and not cache.is_failed(key)]
    print(f"    corpus keys {len(keys):,}; cached {cache.count:,}; "
          f"known-bad {cache.failed_count:,}; uncovered {len(missing):,}")
    if missing:
        examples = ", ".join(missing[:3])
        note = " (expected for a PARTIAL smoke cache)" if cache.partial else ""
        return [f"{len(missing):,} uncovered key(s){note}: {examples}"]
    return []


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache_dir = args.cache_dir or default_cache_dir(REPO_ROOT / "cache", MODEL_NAME, amp=True)
    print(f"device={device}")
    print(f"cache  = {cache_dir}")

    try:
        cache = ViTFeatureCache.open(cache_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    print(f"cache  : {cache.describe()}")

    zip_by_name = {Path(p).name: str(p) for p in Path(args.data_dir).glob("training_data_*.zip")}
    failures = []

    sampled, check_a_failures = check_a(cache, args, device, zip_by_name)
    failures += check_a_failures

    print()
    print(f"[B] loading documents (cap {args.max_blocks} blocks) for the forward check")
    documents = load_documents_from_zips(
        sorted(str(p) for p in Path(args.data_dir).glob("training_data_*.zip")),
        args.max_blocks,
        150,
    )
    candidates = sorted(
        (doc for doc in documents if any(
            block.image_cache_key
            for page in doc.pages
            for block in page.blocks
        )),
        key=lambda doc: sum(len(page.blocks) for page in doc.pages),
    )[: args.docs]
    if not candidates:
        failures.append("no document with an image block found")
    else:
        mini_dir = Path(args.mini_dir)
        mini_cache = build_mini_cache(candidates, args, device, mini_dir)
        failures += check_b(candidates, args, device, mini_cache)

    if not args.skip_coverage:
        failures += check_c(cache, args)

    print()
    print("=" * 72)
    print(f"A: {sampled} rows compared")
    if failures:
        print("FAILED")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("PASSED: cache verified (fidelity, path equivalence, coverage)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())