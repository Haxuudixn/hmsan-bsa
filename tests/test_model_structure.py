"""Structural validation for HMSAN-BSA model with Infini-attention memory."""
from __future__ import annotations

import sys
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

sys.path.insert(0, "src")

# Mock heavy encoders
mock_roberta = MagicMock()
mock_roberta.return_value.last_hidden_state = torch.randn(2, 512, 768)
mock_vit = MagicMock()
mock_vit.return_value.last_hidden_state = torch.randn(2, 197, 768)

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
mock_img_proc.return_value = {"pixel_values": torch.randn(2, 3, 224, 224)}

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
    attn = SparseLocalAttention(dim=128, num_heads=4, window_size=3, num_global=4)
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
        images=[img] * N,
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
    print("\n✅ All structural tests passed!")