import argparse, json, os, glob
TIERS = [("<100", 0, 99), ("100-999", 100, 999), ("1k-5k", 1000, 4999), ("5k+", 5000, 10**9)]

ap = argparse.ArgumentParser(description="macro F1 per label-support tier")
ap.add_argument("--dir", default="outputs/path3_matrix",
                help="dir with one <row>/history.json per row, or a single history.json")
ap.add_argument("--epoch_only", action="store_true",
                help="score epoch-boundary records only (ignore mid_epoch checks)")
args = ap.parse_args()

base = args.dir
rows = []
if os.path.isfile(base):
    cands = [(os.path.basename(os.path.dirname(base)), base)]
else:
    cands = [(d, os.path.join(base, d, "history.json")) for d in sorted(os.listdir(base))]
for d, p in cands:
    if not os.path.isfile(p):
        continue
    h = json.load(open(p, encoding="utf-8"))
    if args.epoch_only:
        h = [e for e in h if e.get("phase") != "mid_epoch"]
    if not h:
        continue
    # pair acc + macro_f1 from the acc-optimal epoch, same rule as the dashboard
    best = max(h, key=lambda e: e["val"]["accuracy"])
    v = best["val"]
    f1 = [float(v.get("f1_%d" % i, 0.0)) for i in range(19)]
    sup = [int(v.get("support_%d" % i, 0)) for i in range(19)]
    rows.append((d, best["epoch"], float(v["accuracy"]), float(v["macro_f1"]), f1, sup))

if not rows:
    raise SystemExit("no history.json found")

sup_ref = rows[0][5]
print("val support per tier (class counts):")
for name, lo, hi in TIERS:
    idx = [i for i, s in enumerate(sup_ref) if lo <= s <= hi]
    print("  %-9s n_classes=%2d  blocks=%6d  (%.1f%% of val)" % (
        name, len(idx), sum(sup_ref[i] for i in idx),
        100.0 * sum(sup_ref[i] for i in idx) / sum(sup_ref)))
print()
hdr = "%-22s %3s %7s %8s | %s" % ("row", "ep", "acc", "macroF1", " ".join("%9s" % t[0] for t in TIERS))
print(hdr)
print("-" * len(hdr))
for d, ep, acc, mf1, f1, sup in rows:
    cells = []
    for name, lo, hi in TIERS:
        idx = [i for i, s in enumerate(sup) if lo <= s <= hi]
        cells.append("%9.4f" % (sum(f1[i] for i in idx) / len(idx) if idx else float("nan")))
    print("%-22s %3d %7.4f %8.4f | %s" % (d, ep, acc, mf1, " ".join(cells)))

print()
print("spread across rows (max-min):")
print("  overall macroF1: %.4f" % (max(r[3] for r in rows) - min(r[3] for r in rows)))
for name, lo, hi in TIERS:
    vals = []
    for r in rows:
        idx = [i for i, s in enumerate(r[5]) if lo <= s <= hi]
        vals.append(sum(r[4][i] for i in idx) / len(idx) if idx else float("nan"))
    print("  tier %-8s : %.4f" % (name, max(vals) - min(vals)))
