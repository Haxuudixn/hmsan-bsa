import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bid_slicing.data.dataset import load_documents_from_zips, collate_document
from bid_slicing.models.hmsan_bsa import HMSAN_BSA

zip_path = sys.argv[1]
docs = load_documents_from_zips([zip_path])
print("documents", len(docs))
total = sum(len(p.blocks) for d in docs for p in d.pages)
image_blocks = [b for d in docs for p in d.pages for b in p.blocks if b.block_type_id in (1, 2)]
print("total_blocks", total, "image/mixed_blocks", len(image_blocks))

batch = collate_document([docs[0]])
assert "image_blocks" in batch
assert len(batch["image_blocks"]) == len(batch["texts"])
print("batch_blocks", len(batch["texts"]), "page_splits", len(batch["page_splits"]))

# Simulate page-level lazy loading used by HMSAN_BSA.forward.
first_page_end = batch["page_splits"][0]
page_image_blocks = batch["image_blocks"][:first_page_end]
page_images = [b.load_image() if b is not None else None for b in page_image_blocks]
print("first_page_images_loaded", sum(im is not None for im in page_images))
print("first_page_total", len(page_images))

params = list(inspect.signature(HMSAN_BSA.forward).parameters)
print("forward_has_image_blocks", "image_blocks" in params)
print("forward_has_old_images", "images" in params)
