"""Document-level significance test for the clean test/val split.

Why this file exists
--------------------
``docs/path3_results_summary_2026-10-04.md`` reports row differences in tenths
of a point, but every row is a single run, and the 79,851 test blocks are not
independent: they arrive in 62 documents, whose layout, template and class mix
differ, so errors inside one document are correlated.  A block-level bootstrap
therefore understates the real uncertainty, and a reviewer is entitled to say
so.  The resampling unit that matters is the document.

This script reads the per-block dumps written by ``dump_block_preds.py``
(pred / target / doc_index) and reports, for one or two models:

* macro F1 with a *block-level* bootstrap CI   -- the optimistic reading;
* macro F1 with a *document-level* bootstrap CI -- the honest reading;
* the design effect (variance ratio), i.e. how much the clustering inflates
  the spread;
* the paired document-level difference ``A - B`` with its CI and a two-sided
  bootstrap p-value, plus a per-document win/tie/loss summary;
* the same two things per label-support tier (``<100`` / ``100-999`` /
  ``1k-5k`` / ``5k+``), where the rare-class weakness lives.

Both models are always resampled with the *same* document weights, which is
what makes the difference paired: the resample removes, rather than doubles,
the document-level variance that the two models share.

Usage
-----
    python scripts/paper/doc_level_significance.py --split test \
        --a delivered_ep16 --b b3_cap3000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TIERS = [("<100", 0, 99), ("100-999", 100, 999),
         ("1k-5k", 1000, 4999), ("5k+", 5000, 10 ** 9)]
DEFAULT_DIR = (Path(__file__).resolve().parents[2] / "outputs" / "path3_diag"
               / "block_preds")
IGNORE_INDEX = -100
NUM_CLASSES = 19


def valid_blocks(pred, target, doc_index, ignore_index):
    """Keep labeled blocks only; the mask is applied once, here."""
    target = target.astype(np.int64)
    keep = target != ignore_index
    return (pred[keep].astype(np.int64), target[keep],
            doc_index[keep].astype(np.int64))


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_model(directory: Path, name: str, split: str):
    npz_path = directory / f"{name}__{split}.npz"
    meta_path = directory / f"{name}__{split}.meta.json"
    if not npz_path.is_file() or not meta_path.is_file():
        raise SystemExit(f"missing {npz_path.name} / {meta_path.name}; run "
                         f"dump_block_preds.py first")
    data = np.load(npz_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return data["pred"], data["target"], data["doc_index"], meta


def confusion_rows(pred, target, num_classes):
    """Per-document 19x19 confusion matrices, scaled rows = true class."""
    flat = target.astype(np.int64) * num_classes + pred.astype(np.int64)
    counts = np.bincount(flat, minlength=num_classes * num_classes)
    matrix = counts.reshape(num_classes, num_classes).astype(np.float64)
    return matrix


def per_document_matrices(pred, target, doc_index, num_docs, num_classes):
    """(num_docs, C*C) float64, one row per document."""
    out = np.zeros((num_docs, num_classes * num_classes), dtype=np.float64)
    for index in range(num_docs):
        mask = doc_index == index
        if not mask.any():
            continue
        flat = (target[mask].astype(np.int64) * num_classes
                + pred[mask].astype(np.int64))
        out[index] = np.bincount(flat, minlength=num_classes * num_classes)
    return out


# ---------------------------------------------------------------------------
# metrics (identical to train.compute_metrics / eval_clean_test.metrics)
# ---------------------------------------------------------------------------

def macro_f1_rows(rows, num_classes):
    """macro F1 for every stacked confusion matrix in ``rows`` (B, C*C)."""
    matrix = rows.reshape(-1, num_classes, num_classes)
    tp = np.diagonal(matrix, axis1=1, axis2=2)
    fp = matrix.sum(axis=1) - tp
    fn = matrix.sum(axis=2) - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.where(denom > 0, 2.0 * tp / np.where(denom > 0, denom, 1.0), 0.0)
    total = matrix.sum(axis=(1, 2))
    accuracy = np.where(total > 0, tp.sum(axis=1) / np.where(total > 0, total, 1.0),
                        0.0)
    return f1.mean(axis=1), accuracy, f1


def tier_index(support, lo, hi):
    return [c for c, s in enumerate(support) if lo <= int(s) <= hi]


def binom_two_sided(k, n):
    """Exact two-sided sign test, without pulling in scipy."""
    from math import comb
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


def describe(samples, level=95.0):
    lo = (100.0 - level) / 2.0
    return {"mean": float(np.mean(samples)),
            "lo": float(np.percentile(samples, lo)),
            "hi": float(np.percentile(samples, 100.0 - lo)),
            "sd": float(np.std(samples, ddof=1))}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def line(label, point, ci, width=30):
    return (f"{label:<{width}}{point:>9.4f}   [{ci['lo']:>7.4f}, {ci['hi']:>7.4f}]"
            f"   sd={ci['sd']:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument("--split", default="test")
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", default="")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    directory = Path(args.dir)
    pred_a, target_a, doc_a, meta_a = load_model(directory, args.a, args.split)
    names_a = meta_a["doc_names"]
    num_docs = len(names_a)
    num_classes = NUM_CLASSES
    pred_a, target_a, doc_a = valid_blocks(pred_a, target_a, doc_a,
                                           int(meta_a["ignored_index"]))

    models = [("A:" + args.a, pred_a, target_a, doc_a)]
    if args.b:
        pred_b, target_b, doc_b, meta_b = load_model(directory, args.b, args.split)
        pred_b, target_b, doc_b = valid_blocks(pred_b, target_b, doc_b,
                                               int(meta_b["ignored_index"]))
        if meta_b["doc_names"] != names_a:
            raise SystemExit("the two dumps scored different documents")
        models.append(("B:" + args.b, pred_b, target_b, doc_b))

    support = np.bincount(target_a, minlength=num_classes)
    rng = np.random.default_rng(args.seed)
    n_boot = args.bootstrap

    print(f"split={args.split}  documents={num_docs}  blocks={target_a.size}  "
          f"bootstrap={n_boot}  seed={args.seed}")
    for label, _, _, meta in ((f"A:{args.a}", 0, 0, meta_a),) + (
            ((f"B:{args.b}", 0, 0, meta_b),) if args.b else ()):
        print(f"  {label:<22} epoch={meta['source_epoch']:<3} "
              f"acc={meta['accuracy']:.4f} macroF1={meta['macro_f1']:.4f} "
              f"({meta['blocks_valid']} blocks, {meta['segments_scored']} docs)")

    print()
    print(f"label support ({args.split}): " + "  ".join(
        f"{name} n={len(tier_index(support, lo, hi))} "
        f"blocks={int(sum(support[c] for c in tier_index(support, lo, hi)))}"
        for name, lo, hi in TIERS))

    # --- bootstrap weight draws -------------------------------------------
    doc_weights = rng.multinomial(num_docs, np.full(num_docs, 1.0 / num_docs),
                                  size=n_boot).astype(np.float64)
    results = {}
    per_model = {}
    for label, pred, target, doc_index in models:
        # each model draws its own blocks: the block-level reading is the
        # optimistic one precisely because blocks are treated as independent
        block_pmf = confusion_rows(pred, target, num_classes).reshape(-1)
        block_counts = rng.multinomial(int(target.size), block_pmf / block_pmf.sum(),
                                       size=n_boot)
        doc_mats = per_document_matrices(pred, target, doc_index, num_docs,
                                        num_classes)
        pooled = doc_mats.sum(axis=0)
        point_f1, point_acc, point_per_class = macro_f1_rows(
            pooled.reshape(1, -1), num_classes)
        doc_f1, doc_acc, doc_per_class = macro_f1_rows(doc_weights @ doc_mats,
                                                       num_classes)
        blk_f1, blk_acc, _ = macro_f1_rows(block_counts, num_classes)
        per_model[label] = {
            "point_f1": float(point_f1[0]), "point_acc": float(point_acc[0]),
            "doc": doc_f1, "block": blk_f1, "doc_acc": doc_acc,
            "doc_per_class": doc_per_class, "point_per_class": point_per_class[0],
            "doc_mats": doc_mats,
        }

    print()
    print("macro F1, point estimate and 95% bootstrap CI")
    print(f"{'model':<30}{'point':>9}   {'95% CI':<19}   spread")
    print("-" * 78)
    for label in per_model:
        entry = per_model[label]
        print(line(label + "  block-boot", entry["point_f1"],
                   describe(entry["block"])))
        print(line(label + "  doc-boot  ", entry["point_f1"],
                   describe(entry["doc"])))
        design = (np.var(entry["doc"], ddof=1) / np.var(entry["block"], ddof=1)
                  if np.var(entry["block"], ddof=1) > 0 else float("nan"))
        results.setdefault(label, {})["design_effect"] = float(design)
        results.setdefault(label, {})["overall"] = {
            "point": float(entry["point_f1"]),
            "doc_ci": describe(entry["doc"]),
            "block_ci": describe(entry["block"]),
        }
        results.setdefault(label, {})["accuracy"] = {
            "point": float(entry["point_acc"]),
            "doc_ci": describe(entry["doc_acc"]),
        }
        print(f"{'':<30}{'design effect (doc/block var)':>9}   {design:.1f}x")
    print()

    if args.b:
        label_a, label_b = list(per_model)[0], list(per_model)[1]
        delta_point = per_model[label_a]["point_f1"] - per_model[label_b]["point_f1"]
        delta_doc = per_model[label_a]["doc"] - per_model[label_b]["doc"]
        delta_blk = per_model[label_a]["block"] - per_model[label_b]["block"]
        ci = describe(delta_doc)
        p_two = 2.0 * min(float(np.mean(delta_doc <= 0.0)),
                          float(np.mean(delta_doc >= 0.0)))
        p_two = min(1.0, p_two)
        print(f"paired difference  A - B  ({args.a} - {args.b})")
        print(f"{'  point (delta macro F1)':<30}{delta_point:>9.4f}")
        print(line("  document-level (paired)", delta_point, ci))
        print(line("  block-level (unpaired) ", delta_point, describe(delta_blk)))
        print(f"  P(delta > 0) = {np.mean(delta_doc > 0):.4f}   "
              f"two-sided bootstrap p = {p_two:.4f}")
        results["paired"] = {
            "a": args.a, "b": args.b, "delta_point": float(delta_point),
            "doc_ci": ci, "block_ci": describe(delta_blk),
            "p_two_sided": p_two, "p_delta_gt_zero": float(np.mean(delta_doc > 0)),
        }

        counts_a = np.bincount(doc_a, minlength=num_docs)
        counts_b = np.bincount(doc_b, minlength=num_docs)
        correct_a = np.zeros(num_docs)
        correct_b = np.zeros(num_docs)
        for index in range(num_docs):
            mask_a = doc_a == index
            mask_b = doc_b == index
            correct_a[index] = (pred_a[mask_a] == target_a[mask_a]).sum()
            correct_b[index] = (pred_b[mask_b] == target_b[mask_b]).sum()
        acc_a = correct_a / np.maximum(counts_a, 1)
        acc_b = correct_b / np.maximum(counts_b, 1)
        wins = int((acc_a > acc_b).sum())
        ties = int((acc_a == acc_b).sum())
        losses = int((acc_a < acc_b).sum())
        sign_p = binom_two_sided(wins, wins + losses)
        print(f"  per-document accuracy: A wins {wins}, ties {ties}, "
              f"B wins {losses}  (of {num_docs} documents; "
              f"exact sign test p = {sign_p:.2e})")
        results["paired"]["per_doc_wins"] = {"a": wins, "ties": ties,
                                             "b": losses, "sign_p": sign_p}

        # Paired document-level CI for the *accuracy* difference, using the
        # same document weights as the macro-F1 difference.  This is the
        # interval that backs the accuracy-first reading of the main claim.
        acc_doc_a = (doc_weights @ correct_a) / np.maximum(doc_weights @ counts_a, 1.0)
        acc_doc_b = (doc_weights @ correct_b) / np.maximum(doc_weights @ counts_b, 1.0)
        delta_acc = acc_doc_a - acc_doc_b
        acc_a_pt = float(correct_a.sum() / max(counts_a.sum(), 1))
        acc_b_pt = float(correct_b.sum() / max(counts_b.sum(), 1))
        acc_point = acc_a_pt - acc_b_pt
        acc_ci = describe(delta_acc)
        acc_p = min(1.0, 2.0 * min(float(np.mean(delta_acc <= 0.0)),
                                   float(np.mean(delta_acc >= 0.0))))
        print(line("  document-level (paired, accuracy)", acc_point, acc_ci)
              + f"   two-sided p = {acc_p:.4f}")
        results["paired"]["accuracy_delta"] = {
            "a": acc_a_pt, "b": acc_b_pt, "point": acc_point,
            "doc_ci": acc_ci, "p_two_sided": acc_p,
            "p_delta_gt_zero": float(np.mean(delta_acc > 0)),
        }

        print()
        print(f"support-filtered macro F1: A - B ({args.a} - {args.b}), "
              f"95% doc-level CI")
        filtered = {}
        for threshold in (100, 500):
            idx = [c for c in range(num_classes) if support[c] >= threshold]
            if not idx:
                continue
            point_a = float(np.mean(per_model[label_a]["point_per_class"][idx]))
            point_b = float(np.mean(per_model[label_b]["point_per_class"][idx]))
            samples = (per_model[label_a]["doc_per_class"][:, idx].mean(axis=1)
                       - per_model[label_b]["doc_per_class"][:, idx].mean(axis=1))
            ci = describe(samples)
            p_two = min(1.0, 2.0 * min(float(np.mean(samples <= 0.0)),
                                       float(np.mean(samples >= 0.0))))
            filtered[threshold] = {
                "n_classes": len(idx),
                "a": point_a, "b": point_b,
                "delta_point": point_a - point_b, "ci": ci, "p_two_sided": p_two,
                "a_doc_ci": describe(per_model[label_a]["doc_per_class"][:, idx]
                                     .mean(axis=1)),
            }
            print(line(f"  support>={threshold} (n={len(idx)})",
                       point_a - point_b, ci) +
                  f"   A={point_a:.4f}  B={point_b:.4f}  p={p_two:.3f}")
        results["paired"]["support_filtered"] = filtered

    print()
    print(f"support-stratified macro F1 ({args.split}, 95% doc-level CI)")
    header = f"{'model':<30}{'tier':<10}{'n':>3}{'point':>9}   {'95% CI':<19}"
    print(header)
    print("-" * len(header))
    for label in per_model:
        entry = per_model[label]
        tiers = {}
        for name, lo, hi in TIERS:
            idx = tier_index(support, lo, hi)
            if not idx:
                continue
            point = float(np.mean(entry["point_per_class"][idx]))
            samples = entry["doc_per_class"][:, idx].mean(axis=1)
            ci = describe(samples)
            tiers[name] = {"point": point, "ci": ci, "n_classes": len(idx)}
            print(line(f"{label}  {name}", point, ci))
        results.setdefault(label, {})["tiers"] = tiers
        print()

    if args.b:
        label_a, label_b = list(per_model)[0], list(per_model)[1]
        print(f"tier deltas  A - B  ({args.a} - {args.b})")
        tier_deltas = {}
        for name, lo, hi in TIERS:
            idx = tier_index(support, lo, hi)
            if not idx:
                continue
            samples = (per_model[label_a]["doc_per_class"][:, idx].mean(axis=1)
                       - per_model[label_b]["doc_per_class"][:, idx].mean(axis=1))
            point = (float(np.mean(per_model[label_a]["point_per_class"][idx]))
                     - float(np.mean(per_model[label_b]["point_per_class"][idx])))
            ci = describe(samples)
            p_two = min(1.0, 2.0 * min(float(np.mean(samples <= 0.0)),
                                       float(np.mean(samples >= 0.0))))
            tier_deltas[name] = {"point": point, "ci": ci, "p_two_sided": p_two,
                                 "n_classes": len(idx)}
            print(line(f"  {name}", point, ci) +
                  f"   P(>0)={np.mean(samples > 0):.3f}  p={p_two:.3f}")
        results["paired"]["tiers"] = tier_deltas

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        for entry in results.values():
            entry.pop("doc", None)
            entry.pop("block", None)
            entry.pop("doc_acc", None)
            entry.pop("doc_per_class", None)
            entry.pop("point_per_class", None)
            entry.pop("doc_mats", None)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())