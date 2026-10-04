"""Structural validation for HMSAN-BSA model with Infini-attention memory."""
from __future__ import annotations

import sys
from types import SimpleNamespace
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

sys.path.insert(0, "src")

# Mock heavy encoders.  The mocks echo the real batch size back, so tests are
# free to use any number of blocks per page instead of being pinned to 2.
mock_roberta = MagicMock()
mock_roberta.side_effect = lambda **kw: SimpleNamespace(
    last_hidden_state=torch.randn(kw["input_ids"].shape[0], 512, 768)
)
mock_vit = MagicMock()
mock_vit.side_effect = lambda **kw: SimpleNamespace(
    last_hidden_state=torch.randn(kw["pixel_values"].shape[0], 197, 768)
)

class MockTokenizer:
    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        B = len(texts)
        return {
            "input_ids": torch.randint(0, 21128, (B, 512)),
            "attention_mask": torch.ones(B, 512),
        }
    @staticmethod
    def from_pretrained(name):
        return MockTokenizer()

mock_img_proc = MagicMock()
mock_img_proc.side_effect = lambda images=None, **kw: {
    "pixel_values": torch.randn(len(images), 3, 224, 224)
}

with patch("transformers.AutoModel.from_pretrained") as mock_am, \
     patch("transformers.AutoTokenizer.from_pretrained", return_value=MockTokenizer()), \
     patch("transformers.AutoImageProcessor.from_pretrained", return_value=mock_img_proc):
    mock_am.side_effect = lambda model_name, **kw: (
        mock_roberta if "roberta" in model_name else mock_vit
    )

    from bid_slicing.features import LayoutEncoder, RoBERTaTextEncoder, ViTImageEncoder
    from bid_slicing.models.block_encoder import BlockMultiModalEncoder, BlockType
    from bid_slicing.models.memory import InfiniMemory, InfiniPageMemory, InfiniSectionMemory
    from bid_slicing.models.page_encoder import PageEncoder, InterPageAttention, SparseLocalAttention
    from bid_slicing.models.boundary_head import BoundaryHead, ClassificationHead, SectionHead
    from bid_slicing.models.hmsan_bsa import HMSAN_BSA


class _ImageBlock:
    """Minimal stand-in for a dataset image block: the model calls load_image()."""

    def __init__(self, image):
        self._image = image

    def load_image(self):
        return self._image


def test_infini_memory_core():
    """Test InfiniMemory core retrieval and update."""
    mem = InfiniMemory(dim=128)
    B, D = 2, 128
    M, z = mem._init_state(B, torch.device("cpu"))
    x = torch.randn(B, D)

    # First step: memory is zero, readout should be near zero
    out1, M, z = mem.forward(x, M, z)
    assert out1.shape == (B, D), f"Expected ({B},{D}), got {out1.shape}"
    assert M.shape == (B, D, D)
    assert z.shape == (B, D)
    assert not torch.allclose(M, torch.zeros_like(M)), "Memory should be updated"

    # Second step: memory has content, readout should carry context
    x2 = torch.randn(B, D)
    out2, M, z = mem.forward(x2, M, z)
    assert out2.shape == (B, D)
    print("  ✓ InfiniMemory core (retrieve + update)")

    # Test update_mask
    M0, z0 = mem._init_state(B, torch.device("cpu"))
    mask = torch.tensor([0.0, 1.0])
    _, M_masked, z_masked = mem.forward(x, M0, z0, update_mask=mask)
    # Batch item 0 with mask=0 should have zero contribution
    assert torch.allclose(M_masked[0], torch.zeros(D, D)), "Mask=0 should skip update"
    assert not torch.allclose(M_masked[1], torch.zeros(D, D)), "Mask=1 should update"
    print("  ✓ InfiniMemory update_mask")

    # Test delta-rule
    mem_delta = InfiniMemory(dim=128, use_delta_rule=True)
    M0, z0 = mem_delta._init_state(2, torch.device("cpu"))
    out, _, _ = mem_delta.forward(torch.randn(2, 128), M0, z0)
    assert out.shape == (2, 128)
    print("  ✓ InfiniMemory delta-rule")


def test_infini_page_memory():
    """Test InfiniPageMemory step interface."""
    mem = InfiniPageMemory(dim=128)
    B, D = 2, 128
    M, z = mem.init_state(B, torch.device("cpu"))

    # Simulate 5 sequential pages
    for _ in range(5):
        page = torch.randn(B, D)
        enhanced, M, z = mem.step(page, M, z)
        assert enhanced.shape == (B, D)
        assert M.shape == (B, D, D)
        assert z.shape == (B, D)
    print("  ✓ InfiniPageMemory 5-step sequence")


def test_infini_section_memory():
    """Test InfiniSectionMemory with boundary gating."""
    mem = InfiniSectionMemory(dim=128, num_boundaries=4)
    B, D = 2, 128
    M, z = mem.init_state(B, torch.device("cpu"))

    # B-SECTION boundary logits (high B-score)
    b_logits = torch.tensor([[10.0, -10.0, -10.0, -10.0],  # B-SECTION
                             [-10.0, -10.0, 10.0, -10.0]])  # E-SECTION
    page = torch.randn(B, D)
    enhanced, M, z = mem.step(page, b_logits, M, z)
    assert enhanced.shape == (B, D)
    # B-SECTION → update_mask should be lower than E-SECTION
    print("  ✓ InfiniSectionMemory boundary-gated step")


def test_layout_encoder():
    enc = LayoutEncoder(output_dim=64)
    B = 4
    out = enc(
        bbox_norm=torch.randn(B, 8),
        font_size=torch.rand(B, 1),
        is_bold=torch.randint(0, 2, (B, 1)).float(),
        is_italic=torch.randint(0, 2, (B, 1)).float(),
        font_color=torch.rand(B, 3),
        block_type_id=torch.randint(0, 3, (B,)),
    )
    assert out.shape == (B, 64)
    print("  ✓ LayoutEncoder (4,64)")


def test_text_encoder():
    enc = RoBERTaTextEncoder(max_length=510, freeze=True)
    out = enc(["测试文本1", "测试文本2"])
    assert out.shape == (2, 768)
    print("  ✓ RoBERTaTextEncoder (2,768)")


def test_image_encoder():
    from PIL import Image
    enc = ViTImageEncoder(freeze=True)
    img = Image.new("RGB", (224, 224), color="red")
    out = enc([img, img])
    assert out.shape == (2, 768)
    print("  ✓ ViTImageEncoder (2,768)")


def test_block_encoder():
    from PIL import Image
    enc = BlockMultiModalEncoder(text_freeze=True, image_freeze=True)
    B = 6
    img = Image.new("RGB", (224, 224), color="blue")
    out = enc(
        texts=["text1", "text2", "", "", "mixed_ocr", "mixed_ocr"],
        ocr_texts=["", "", "", "", "ocr text 1", "ocr text 2"],
        images=[None, None, img, img, img, img],
        block_type_ids=torch.tensor([0, 0, 1, 1, 2, 2]),
        bbox_norm=torch.randn(B, 8),
        font_size=torch.rand(B, 1),
        is_bold=torch.randint(0, 2, (B, 1)).float(),
        is_italic=torch.randint(0, 2, (B, 1)).float(),
        font_color=torch.rand(B, 3),
    )
    assert out.shape == (B, 128)
    print("  ✓ BlockMultiModalEncoder (6,128)")


def test_sparse_attention():
    attn = SparseLocalAttention(dim=128, num_heads=4, num_global=4)
    x = torch.randn(2, 14, 128)
    out = attn(x)
    assert out.shape == x.shape
    print("  ✓ SparseLocalAttention (2,14,128)")


def test_page_encoder():
    enc = PageEncoder(dim=128, num_layers=2, window_size=5, num_global=4)
    blocks = torch.randn(2, 12, 128)
    page_repr, encoded = enc(blocks)
    assert page_repr.shape == (2, 128)
    assert encoded.shape == (2, 12, 128)
    print("  ✓ PageEncoder")


def test_inter_page_attention():
    attn = InterPageAttention(dim=128, window=3)
    page_repr = torch.randn(2, 128)
    history = torch.randn(2, 3, 128)
    out = attn(page_repr, history)
    assert out.shape == (2, 128)
    print("  ✓ InterPageAttention")


def test_heads():
    B, D = 2, 128
    bh = BoundaryHead(dim=D, num_boundaries=4)
    assert bh(torch.randn(B, D)).shape == (B, 4)

    ch = ClassificationHead(dim=D, num_classes=19)
    assert ch(torch.randn(B, 5, D)).shape == (B, 5, 19)

    sh = SectionHead(dim=D, num_classes=19)
    assert sh(torch.randn(B, D), torch.randn(B, D)).shape == (B, 19)
    print("  ✓ BoundaryHead + ClassificationHead + SectionHead")


def test_hmsan_bsa_full():
    """End-to-end HMSAN_BSA forward with Infini-attention memory."""
    from PIL import Image
    model = HMSAN_BSA(text_freeze=True, image_freeze=True)
    img = Image.new("RGB", (224, 224), color="green")
    N = 12

    out = model(
        texts=["text"] * N,
        ocr_texts=[""] * N,
        image_blocks=[_ImageBlock(img)] * N,
        block_type_ids=torch.randint(0, 3, (N,)),
        bbox_norm=torch.randn(N, 8),
        font_size=torch.rand(N, 1),
        is_bold=torch.randint(0, 2, (N, 1)).float(),
        is_italic=torch.randint(0, 2, (N, 1)).float(),
        font_color=torch.rand(N, 3),
        page_splits=[4, 9, 12],
    )

    assert len(out["block_logits"]) == 3
    assert len(out["boundary_logits"]) == 3
    assert len(out["section_logits"]) == 3
    assert out["block_logits"][0].shape == (1, 4, 19)
    assert out["block_logits"][1].shape == (1, 5, 19)
    assert out["block_logits"][2].shape == (1, 3, 19)
    assert out["boundary_logits"][0].shape == (1, 4)
    assert out["section_logits"][0].shape == (1, 19)

    # Infini-attention state shapes
    D = 128
    assert out["page_M"].shape == (1, D, D)
    assert out["page_z"].shape == (1, D)
    assert out["section_M"].shape == (1, D, D)
    assert out["section_z"].shape == (1, D)
    print("  ✓ HMSAN_BSA full forward with Infini-attention (3 pages, 12 blocks) ✓")
    print(f"    page_M: {out['page_M'].shape}, page_z: {out['page_z'].shape}")
    print(f"    section_M: {out['section_M'].shape}, section_z: {out['section_z'].shape}")




def test_ablation_flags():
    """AblationFlags defaults, serialisation and preset registry."""
    from bid_slicing.models.ablation import AblationFlags, PRESETS, resolve_preset

    flags = AblationFlags()
    assert flags.is_default()
    assert flags.signature() == "gsa-g11-g21-adk1-gf1-ffn1-ip1-pm1-sm1-bg1"

    # to_dict/from_dict round-trip; unknown keys ignored, missing keys defaulted
    assert AblationFlags.from_dict(flags.to_dict()) == flags
    assert AblationFlags.from_dict(None).is_default()
    assert AblationFlags.from_dict({}).is_default()
    assert AblationFlags.from_dict(
        {"g1_output_gate": False, "not_a_flag": 1}
    ) == AblationFlags(g1_output_gate=False)

    # Every preset is reachable by name and has a unique signature
    sigs = set()
    for name in PRESETS:
        label, preset_flags = resolve_preset(name)
        assert label
        sigs.add(preset_flags.signature())
    assert len(sigs) == len(PRESETS), "two presets share a signature"

    for bad in ("nope", ""):
        try:
            resolve_preset(bad)
        except KeyError:
            pass
        else:
            raise AssertionError(f"resolve_preset({bad!r}) should raise")

    # Invalid values are rejected at construction time
    for kwargs in ({"attention_mode": "magic"}, {"local_window": 0}):
        try:
            AblationFlags(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"AblationFlags(**{kwargs}) should raise")
    print(f"  OK AblationFlags ({len(PRESETS)} presets, unique signatures)")


def test_ablation_modules_removed():
    """A switched-off component must not allocate parameters at all."""
    from bid_slicing.models.ablation import AblationFlags, resolve_preset

    model = HMSAN_BSA(
        text_freeze=True, image_freeze=True,
        flags=resolve_preset("A11_no_gates")[1],
    )
    attn = model.page_encoder.layers[0].attn
    assert not hasattr(attn, "g1_proj"), "A11 must drop G1"
    assert not hasattr(attn, "g2_proj"), "A11 must drop G2"
    assert not hasattr(model.page_encoder.layers[0].ffn, "w_value"), "A11 must drop the GLU"
    assert not hasattr(model.block_encoder, "fusion_gate"), "A11 must drop the fusion gate"
    assert hasattr(model.block_encoder, "fusion_concat"), "A11 needs the concat path"
    assert not hasattr(model.inter_page_attn, "gate"), "A11 must drop the inter-page gate"
    # Memory survives A11: only the gates are removed.
    assert model.page_memory is not None and model.section_memory is not None

    reduced = HMSAN_BSA(
        text_freeze=True, image_freeze=True,
        flags=AblationFlags(boundary_gate=False),
    )
    assert not hasattr(reduced.section_memory, "boundary_gate"), "A7 must drop the boundary gate"

    dense = HMSAN_BSA(
        text_freeze=True, image_freeze=True,
        flags=AblationFlags(attention_mode="dense"),
    )
    assert not hasattr(dense.page_encoder.layers[0].attn, "W_iq"), "dense mode needs no indexer"

    no_mem = HMSAN_BSA(
        text_freeze=True, image_freeze=True,
        flags=resolve_preset("A6A7_no_memory")[1],
    )
    assert no_mem.page_memory is None and no_mem.section_memory is None

    full_params = sum(p.numel() for p in HMSAN_BSA(
        text_freeze=True, image_freeze=True).parameters())
    a11_params = sum(p.numel() for p in model.parameters())
    assert a11_params < full_params, "A11 should have fewer parameters than the full model"
    print(f"  OK disabled components allocate no parameters "
          f"(A11 {a11_params:,} < full {full_params:,})")


def test_ablation_forward():
    """Every preset must run end-to-end and keep the output contract."""
    from bid_slicing.models.ablation import PRESETS

    N = 8
    for name in PRESETS:
        model = HMSAN_BSA(
            text_freeze=True, image_freeze=True,
            flags=PRESETS[name][1],
        )
        model.eval()
        with torch.no_grad():
            out = model(
                texts=["sample text"] * N,
                ocr_texts=[""] * N,
                image_blocks=[None] * N,
                block_type_ids=torch.zeros(N, dtype=torch.long),
                bbox_norm=torch.randn(N, 8),
                font_size=torch.rand(N, 1),
                is_bold=torch.zeros(N, 1),
                is_italic=torch.zeros(N, 1),
                font_color=torch.rand(N, 3),
                page_splits=[4, 8],
            )
        assert len(out["block_logits"]) == 2, name
        assert out["block_logits"][0].shape == (1, 4, 19), name
        assert out["block_logits"][1].shape == (1, 4, 19), name
        assert out["boundary_logits"][0].shape == (1, 4), name
        assert out["section_logits"][0].shape == (1, 19), name
        drops_page_mem = name in ("A6_no_page_memory", "A6A7_no_memory")
        drops_section_mem = name == "A6A7_no_memory"
        assert (out["page_M"] is None) == drops_page_mem, name
        assert (out["section_M"] is None) == drops_section_mem, name
        if not drops_page_mem:
            assert out["page_M"].shape == (1, 128, 128), name
            assert out["page_z"].shape == (1, 128), name
        if not drops_section_mem:
            assert out["section_M"].shape == (1, 128, 128), name
    print(f"  OK reduced-model forward ({len(PRESETS)} presets, 8 blocks / 2 pages)")


def test_vit_cache_paths():
    """Frozen-ViT cache: a hit must skip decoding, anything else must not lie."""
    import shutil
    import tempfile
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from bid_slicing.data.vit_cache import ViTFeatureCache, block_key, save_cache

    class _CachedImageBlock:
        """Stand-in for a dataset Block that counts how often it is decoded."""

        def __init__(self, zip_name, member, image):
            self.source_zip = zip_name
            self._member = member
            self._image = image
            self.decodes = 0

        @property
        def image_member(self):
            return self._member

        @property
        def image_cache_key(self):
            return block_key(self.source_zip, self._member)

        def load_image(self):
            self.decodes += 1
            return self._image

    cache_dir = Path(tempfile.mkdtemp(prefix="vit_cache_test_"))
    good = ["z1.zip::blocks/a.png", "z1.zip::blocks/b.png"]
    np.save(cache_dir / "features.f32.npy", np.random.rand(2, 768).astype("float32"))
    save_cache(
        cache_dir,
        {
            "version": 1, "model_name": "google/vit-base-patch16-224",
            "hidden_size": 768, "dtype": "float32", "amp": True,
            "count": 2, "failed": 1, "partial": False, "built_at": "test",
        },
        good,
        ["z1.zip::blocks/broken.png"],
    )
    cache = ViTFeatureCache.open(cache_dir)
    assert cache.count == 2 and cache.failed_count == 1, cache.describe()
    assert cache.get_many(good).shape == (2, 768)
    assert cache.has(good[0]) and cache.is_failed("z1.zip::blocks/broken.png")
    assert cache.get_many([good[0], "nope"]) is None, (
        "an incomplete lookup must return None so the caller can fall back"
    )

    image = Image.new("RGB", (224, 224), color="green")
    enc = BlockMultiModalEncoder(text_freeze=True, image_freeze=True, vit_cache=cache)

    def run(blocks):
        return enc(
            texts=[""] * len(blocks),
            ocr_texts=[""] * len(blocks),
            images=None,
            image_blocks=blocks,
            block_type_ids=torch.ones(len(blocks), dtype=torch.long),
            bbox_norm=torch.randn(len(blocks), 8),
            font_size=torch.rand(len(blocks), 1),
            is_bold=torch.zeros(len(blocks), 1),
            is_italic=torch.zeros(len(blocks), 1),
            font_color=torch.rand(len(blocks), 3),
        )

    def run_mixed(blocks):
        return enc(
            texts=[""] * len(blocks),
            ocr_texts=["ocr text 1", "ocr text 2"],
            images=None,
            image_blocks=blocks,
            block_type_ids=torch.full((len(blocks),), 2, dtype=torch.long),
            bbox_norm=torch.randn(len(blocks), 8),
            font_size=torch.rand(len(blocks), 1),
            is_bold=torch.zeros(len(blocks), 1),
            is_italic=torch.zeros(len(blocks), 1),
            font_color=torch.rand(len(blocks), 3),
        )

    assert enc.image_cache_active
    blocks = [
        _CachedImageBlock("z1.zip", "blocks/a.png", image),
        _CachedImageBlock("z1.zip", "blocks/b.png", image),
    ]
    assert run(blocks).shape == (2, 128)
    assert enc.vit_cache_misses == 0, "cached keys must not count as misses"
    assert sum(b.decodes for b in blocks) == 0, "cached blocks must not be decoded"

    broken = _CachedImageBlock("z1.zip", "blocks/broken.png", image)
    run([broken])
    assert broken.decodes == 0 and enc.vit_cache_misses == 0, (
        "a known-bad key means no image, without touching the ZIP"
    )

    unknown = _CachedImageBlock("z1.zip", "blocks/new.png", image)
    run([unknown])
    # A miss decodes for real; it may decode twice (once to learn that an image
    # exists, once to encode it), so only "at least once" is guaranteed here.
    assert unknown.decodes >= 1, "an unknown key must be decoded for real"
    assert enc.vit_cache_misses == 1, "and that fallback must be visible"

    # MIXED blocks (block_type 2) with an active cache: `images` is None, so the
    # text-only fallback must not index into it.  This exact shape crashed the
    # first GPU verification run, which is why it is pinned here.
    mixed_hit = _CachedImageBlock("z1.zip", "blocks/a.png", image)
    mixed_text_only = _CachedImageBlock("z1.zip", "blocks/broken.png", image)
    out = run_mixed([mixed_hit, mixed_text_only])
    assert out.shape == (2, 128)
    assert mixed_hit.decodes == 0, "cached MIXED block must not be decoded"
    assert mixed_text_only.decodes == 0, "known-bad MIXED block must not be decoded"

    unfrozen = BlockMultiModalEncoder(
        text_freeze=True, image_freeze=False, vit_cache=cache
    )
    assert unfrozen.vit_cache is None and not unfrozen.image_cache_active, (
        "cached features must be dropped once the image branch trains"
    )

    shutil.rmtree(cache_dir, ignore_errors=True)
    print("  OK frozen-ViT cache (hit / known-bad / miss / unfrozen)")


if __name__ == "__main__":
    print("\n=== HMSAN-BSA + Infini-attention Structural Validation ===\n")
    print("── InfiniMemory ──")
    test_infini_memory_core()
    test_infini_page_memory()
    test_infini_section_memory()
    print("\n── Features ──")
    test_layout_encoder()
    test_text_encoder()
    test_image_encoder()
    test_block_encoder()
    print("\n── Attention & Encoding ──")
    test_sparse_attention()
    test_page_encoder()
    test_inter_page_attention()
    print("\n── Heads ──")
    test_heads()
    print("\n── Full Model ──")
    test_hmsan_bsa_full()
    print("\n── Ablation Switches ──")
    test_ablation_flags()
    test_ablation_modules_removed()
    test_ablation_forward()
    print()
    print("-- Frozen-ViT Feature Cache --")
    test_vit_cache_paths()
    print("\n✅ All structural tests passed!")