"""Render the clean-split *test* pass as the tables the thesis draft needs.

Reads the two JSON reports produced on the training host --

    outputs/path3_diag/test_eval_clean.json      (eval_clean_test.py)
    outputs/path3_diag/test_eval_baselines.json  (baselines_test_eval.py)

-- and prints, for the test split only:

    * the headline table: val acc / val macro F1 / test acc / test macro F1,
      with the paired test deltas against a reference row (default
      delivered/last, the main model) in percentage points;
    * the support-stratified macro F1 tiers (overall / <100 / 100-999 /
      1k-5k / 5k+), which D6 makes mandatory next to every headline number;
    * the classes whose test F1 moved most against the reference, so a row
      that only wins on the aggregate cannot hide a collapse.

Accuracy and macro F1 are always printed as a pair: D6 forbids reporting one
without the other, because the split is imbalanced enough that either alone
misleads.

    python scripts/paper/report_test_eval.py
    python scripts/paper/report_test_eval.py --reference stage2_pilot_full_seed42/last
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MAIN = REPO / "outputs" / "path3_diag" / "test_eval_clean.json"
DEFAULT_BASELINES = REPO / "outputs" / "path3_diag" / "test_eval_baselines.json"
DEFAULT_REFERENCE = "delivered/last"

TIER_ORDER = ["<100", "100-999", "1k-5k", "5k+"]


def load(path: Path) -> list:
    if not path.is_file():
        return []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return report.get("models", [])


def label_of(entry: dict) -> str:
    if "key" in entry:                      # baselines report
        return entry["key"]
    return entry["id"]                      # matrix report


def test_set(entry: dict) -> dict:
    return (entry.get("sets") or {}).get("test") or {}


def pp(value) -> str:
    return "--" if value is None else f"{value:+.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help="row the paired deltas are measured against")
    parser.add_argument("--movers", type=int, default=8,
                        help="how many worst-falling classes to list")
    args = parser.parse_args()

    entries = load(args.main) + load(args.baselines)
    if not entries:
        print("no reports found -- run eval_clean_test.py on the training host "
              "and copy the JSON back first")
        return 1

    identity = {}
    for entry in entries:
        if test_set(entry):
            identity.setdefault(label_of(entry), entry)
    scored = list(identity.values())

    reference = identity.get(args.reference)
    if reference is None:
        print(f"warning: reference {args.reference!r} has no test row; "
              "deltas are left blank\n")
    ref_test = test_set(reference) if reference else {}

    print("test pass -- clean split, last-epoch checkpoints, selection by "
          "validation only")
    print(f"reference for deltas: {args.reference!r}\n")

    header = (f"{'row':<34}{'ep':>4}{'val acc':>9}{'val F1':>8}"
              f"{'test acc':>10}{'test F1':>9}{'d test F1':>11}{'fidel':>7}")
    print(header)
    print("-" * len(header))
    for entry in scored:
        val = (entry.get("sets") or {}).get("val") or {}
        test = test_set(entry)
        delta = None
        if ref_test.get("macro_f1") is not None:
            delta = 100.0 * (test["macro_f1"] - ref_test["macro_f1"])
        ok = entry.get("val_fidelity", {}).get("ok")
        print(f"{label_of(entry):<34}{entry.get('source_epoch', 0):>4}"
              f"{val.get('accuracy', float('nan')):>9.4f}"
              f"{val.get('macro_f1', float('nan')):>8.4f}"
              f"{test.get('accuracy', float('nan')):>10.4f}"
              f"{test.get('macro_f1', float('nan')):>9.4f}"
              f"{pp(delta):>11}"
              f"{('OK' if ok else 'FAIL' if ok is False else '?'):>7}")

    print("\nsupport-stratified macro F1 (test)")
    tier_header = f"{'row':<34}{'overall':>9}" + "".join(
        f"{name:>10}" for name in TIER_ORDER)
    print(tier_header)
    print("-" * len(tier_header))
    for entry in scored:
        test = test_set(entry)
        tiers = (test.get("tiers") or {}).get("macro_f1") or {}
        cells = "".join(
            f"{tiers[name]:>10.4f}" if tiers.get(name) is not None
            else f"{'n/a':>10}" for name in TIER_ORDER)
        print(f"{label_of(entry):<34}{test.get('macro_f1', float('nan')):>9.4f}"
              f"{cells}")

    if reference is not None and args.movers > 0:
        print(f"\nlargest test-F1 falls vs {args.reference} "
              "(support >= 100 blocks only)")
        ref_classes = {item["label"]: item
                       for item in ref_test.get("per_class", [])}
        for entry in scored:
            if label_of(entry) == args.reference:
                continue
            rows = []
            for item in test_set(entry).get("per_class", []):
                base = ref_classes.get(item["label"])
                if base is None or item["support"] < 100:
                    continue
                rows.append((item["f1"] - base["f1"], item, base))
            rows.sort(key=lambda row: row[0])
            if not rows:
                continue
            print(f"\n  {label_of(entry)}")
            for delta, item, base in rows[:args.movers]:
                print(f"    {item['label'][:38]:<40}"
                      f"support={item['support']:>6}  "
                      f"{base['f1']:.3f} -> {item['f1']:.3f}  "
                      f"{100 * delta:+.1f} pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())