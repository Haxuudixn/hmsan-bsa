"""How much does the (frozen) ViT image branch actually contribute on the test set?

Two passes over the same 66 test segments:
  baseline  - the delivered last.pt as-is
  vit_zero  - ViT output replaced by zeros, i.e. the pure-visual signal removed

Results are broken down by block type so the *headroom* of fine-tuning ViT is
visible: only blocks whose block_type is IMAGE are ever touched by it.
"""
from __future__ import annotations
import json, sys, collections
from pathlib import Path
import torch
from torch.utils.data import DataLoader

REPO = Path(r"C:\Users\Administrator\Documents\Codex\2026-08-20\https\outputs\hmsan-bsa")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

from bid_slicing.data.dataset import BidDocumentDataset, load_documents_from_zips, collate_document, IGNORE_INDEX
from bid_slicing.data.label_schema import LabelSchema
from bid_slicing.models.hmsan_bsa import HMSAN_BSA
import train as train_mod

DATA = Path(r"C:\Users\Administrator\Desktop\训练数据\final_train")
CKPT = REPO / "outputs" / "hmsan_bsa_final" / "last.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
schema = LabelSchema.from_yaml(str(REPO / "configs" / "labels.yaml"))

zips = sorted(str(p) for p in DATA.glob("*.zip"))
docs = load_documents_from_zips(zips, max_blocks_per_sample=3000, max_pages_per_sample=150)
ds = BidDocumentDataset(docs)
train_ds, val_ds, test_ds = ds.split_by_documents(0.7, 0.15, 42)
print(f"test segments = {len(test_ds)}")
loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_document, num_workers=0)

model = HMSAN_BSA(
    text_encoder_name="hfl/chinese-roberta-wwm-ext", text_max_length=510, text_freeze=False,
    image_encoder_name="google/vit-base-patch16-224", image_freeze=True,
    hidden_size=128, num_classes=schema.num_classes, num_boundaries=schema.num_boundaries,
    page_encoder_layers=2, page_encoder_heads=4, gsa_k_base=16, gsa_k_min=4, gsa_k_max=32,
    gsa_indexer_heads=4, gsa_indexer_dim=32, num_global_tokens=4, inter_page_window=3, dropout=0.1,
).to(device)
sd = torch.load(CKPT, map_location=device, weights_only=False)
model.load_state_dict(sd["model_state_dict"])
model.eval()
print(f"loaded last.pt (epoch {sd.get('epoch')})")

vit = model.block_encoder.image_encoder
real_forward = vit.forward
ZERO = {"on": False}

def vit_forward(images):
    if ZERO["on"]:
        return torch.zeros(len(images), 768, device=device)
    return real_forward(images)

vit.forward = vit_forward

GROUPS = ["text", "image_with_ocr", "image_visual_only", "mixed"]
stats = {p: {g: [0, 0] for g in GROUPS} for p in ("baseline", "vit_zero")}

def group_of(bt, ocr):
    if bt == 0:
        return "text"
    if bt == 1:
        return "image_with_ocr" if (ocr or "").strip() else "image_visual_only"
    return "mixed"

with torch.no_grad():
    for bi, batch in enumerate(loader):
        btid = batch["block_type_ids"].to(device)
        ps = batch["page_splits"]
        common = dict(
            texts=batch["texts"], ocr_texts=batch["ocr_texts"], image_blocks=batch["image_blocks"],
            block_type_ids=btid, bbox_norm=batch["bbox_norm"].to(device),
            font_size=batch["font_size"].to(device), is_bold=batch["is_bold"].to(device),
            is_italic=batch["is_italic"].to(device), font_color=batch["font_color"].to(device),
            page_splits=ps,
        )
        labels = batch["labels"]
        for phase in ("baseline", "vit_zero"):
            ZERO["on"] = (phase == "vit_zero")
            with train_mod._autocast(device, True):
                out = model(**common)
            preds, tgts = [], []
            for i, logits in enumerate(out["block_logits"]):
                p0 = 0 if i == 0 else ps[i - 1]
                preds.append(logits.argmax(-1).squeeze(0).cpu())
                tgts.append(labels[p0:ps[i]])
            preds = torch.cat(preds); tgts = torch.cat(tgts)
            bt_all = btid.cpu()
            ocr_all = batch["ocr_texts"]
            for j in range(len(tgts)):
                if int(tgts[j]) == IGNORE_INDEX:
                    continue
                g = group_of(int(bt_all[j]), ocr_all[j])
                cell = stats[phase][g]
                cell[1] += 1
                if int(preds[j]) == int(tgts[j]):
                    cell[0] += 1
        if (bi + 1) % 20 == 0:
            print(f"  {bi+1}/{len(loader)} segments", flush=True)

print("\n" + "=" * 78)
print(f"{'block group':<22}{'n':>8}{'baseline':>12}{'ViT zeroed':>14}{'delta':>10}")
print("-" * 78)
for g in GROUPS:
    b, t = stats["baseline"][g], stats["vit_zero"][g]
    if t[1] == 0:
        print(f"{g:<22}{0:>8}{'-':>12}{'-':>14}{'-':>10}"); continue
    ba, za = b[0] / b[1], t[0] / t[1]
    print(f"{g:<22}{t[1]:>8}{ba:>12.4f}{za:>14.4f}{(za-ba)*100:>9.2f}pp")
tot_b = [sum(stats["baseline"][g][k] for g in GROUPS) for k in (0, 1)]
tot_z = [sum(stats["vit_zero"][g][k] for g in GROUPS) for k in (0, 1)]
print("-" * 78)
print(f"{'ALL':<22}{tot_b[1]:>8}{tot_b[0]/tot_b[1]:>12.4f}{tot_z[0]/tot_z[1]:>14.4f}"
      f"{(tot_z[0]/tot_z[1]-tot_b[0]/tot_b[1])*100:>9.2f}pp")

share = {g: stats["baseline"][g][1] / tot_b[1] for g in GROUPS}
print("\nshare of test blocks:", {g: f"{v*100:.1f}%" for g, v in share.items()})
print(f"\nblocks reachable by ViT (image_* only) = "
      f"{(share['image_with_ocr']+share['image_visual_only'])*100:.1f}% of the test set")
print(f"blocks where ViT is the ONLY modality    = {share['image_visual_only']*100:.1f}%")

REPO.joinpath("docs", "paper_materials").mkdir(parents=True, exist_ok=True)
out = {
    "checkpoint": "last.pt (text_freeze=False, image_freeze=True)",
    "test_segments": len(test_ds),
    "blocks": tot_b[1],
    "per_group": {
        g: {
            "n": stats["baseline"][g][1],
            "share": share[g],
            "baseline_accuracy": stats["baseline"][g][0] / stats["baseline"][g][1] if stats["baseline"][g][1] else None,
            "vit_zeroed_accuracy": stats["vit_zero"][g][0] / stats["vit_zero"][g][1] if stats["vit_zero"][g][1] else None,
        } for g in GROUPS if stats["baseline"][g][1]
    },
    "overall": {
        "baseline_accuracy": tot_b[0] / tot_b[1],
        "vit_zeroed_accuracy": tot_z[0] / tot_z[1],
    },
}
p = REPO / "docs" / "paper_materials" / "image_branch_contribution.json"
p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nwrote {p}")