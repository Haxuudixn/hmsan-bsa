"""Generate the paper figures for the path-3 clean rerun (v2).

Fig4 follows the D2 methodology exactly: pool the two same-config/same-seed rows
(`delivered` + `stage2_pilot_full_seed42`) per epoch before judging the peak,
because a single 66-doc val point is far too noisy to read a peak off directly.

Fig5 classifies every ablation delta per-metric against the measured run-to-run
noise floor (D7 thresholds), instead of collapsing the two metrics into one.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PULL = ROOT / "remote_pull_2026-09-29"
RUNS = PULL / "outputs" / "path3_clean"
DIAG = ROOT / "outputs" / "path3_diag"
OUT = ROOT / "thesis_assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

_TC = json.loads((DIAG / "test_eval_clean.json").read_text(encoding="utf-8"))
_TB = json.loads((DIAG / "test_eval_baselines.json").read_text(encoding="utf-8"))
TCS = {m["id"]: m for m in _TC["models"]}
TBS = {m["key"]: m for m in _TB["models"]}
EFF = {r["id"]: r for r in json.loads(
    (ROOT / "docs" / "paper_materials" / "efficiency_2026-10-04.json")
    .read_text(encoding="utf-8"))["rows"]}
BASE_KEY = {"b1_roberta_mlp": "b1|b1_roberta_mlp|last.pt|cap48",
            "b2_roberta_layout_mlp": "b2|b2_roberta_layout_mlp|last.pt|cap48",
            "b3_page_bigru": "b3|b3_page_bigru|last.pt|cap3000"}


def test_set(slug, split="test"):
    return TCS[f"{slug}/last"]["sets"][split] if slug in dict(MATRIX) else \
        TBS[BASE_KEY[slug]]["sets"][split]

CJK = next((n for n in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC")
            if n in {f.name for f in fm.fontManager.ttflist}), None)
if CJK:
    plt.rcParams["font.sans-serif"] = [CJK]
    plt.rcParams["axes.unicode_minus"] = False
def L(zh, en):
    return zh if CJK else en

TIERS = ("<100", "100-999", "1k-5k", "5k+")
MATRIX = [("delivered", "full (delivered)"), ("stage2_pilot_full_seed42", "full (pilot s42)"),
          ("A2_no_g2", "A2 no_g2"), ("A6_no_page_memory", "A6 no_page_memory"),
          ("A7_no_boundary_gate", "A7 no write-gate"), ("A10_dense", "A10 dense (=B4)"),
          ("A11_no_gates", "A11 no_gates")]
BASE = [("b1_roberta_mlp", "B1 roberta+MLP"), ("b2_roberta_layout_mlp", "B2 +layout"),
        ("b3_page_bigru", "B3 page-BiGRU")]
NOISE_MED, NOISE_MEAN = 1.04, 1.55      # pp, from D2's 32 aligned mid-epoch steps
COLORS = plt.get_cmap("tab10")


def hist(slug):
    return json.loads((RUNS / slug / "history.json").read_text(encoding="utf-8"))


def ends(slug):
    return [r for r in hist(slug) if r.get("phase") in (None, "epoch")]


def mids(slug):
    return [r for r in hist(slug) if r.get("phase") == "mid_epoch"]


def strat(val):
    f1 = [float(val.get(f"f1_{i}", 0.0)) for i in range(19)]
    sup = [int(val.get(f"support_{i}", 0)) for i in range(19)]
    return [sum(f1[i] for i, s in enumerate(sup) if lo <= s <= hi) /
            max(1, len([i for i, s in enumerate(sup) if lo <= s <= hi]))
            for _n, lo, hi in (("<100", 0, 99), ("100-999", 100, 999),
                               ("1k-5k", 1000, 4999), ("5k+", 5000, 10 ** 9))]


# ---------------- Fig 1: per-epoch curves ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for k, (slug, label) in enumerate(MATRIX + BASE):
    e = ends(slug)
    if not e:
        continue
    xs = [r["epoch"] for r in e]
    ls = "--" if slug in [b[0] for b in BASE] else "-"
    axes[0].plot(xs, [r["val"]["accuracy"] for r in e], ls, marker="o", ms=3.5, color=COLORS(k % 10), label=label)
    axes[1].plot(xs, [r["val"].get("macro_f1", 0) for r in e], ls, marker="o", ms=3.5, color=COLORS(k % 10), label=label)
for ax, t in zip(axes, ("Validation accuracy", "Validation macro F1")):
    ax.set_xlabel("epoch"); ax.set_ylabel(t); ax.grid(alpha=.3); ax.set_xticks(range(1, 9))
axes[0].legend(fontsize=7.5, ncol=2)
fig.suptitle(L("路径 3 干净矩阵：逐 epoch 验证曲线（实线=矩阵行，虚线=基线）",
               "Path-3 clean matrix: per-epoch validation curves (solid=matrix, dashed=baseline)"), fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "fig1_epoch_curves.png", dpi=200); plt.close(fig)

# ---------------- Fig 2: support-stratified macro F1, val | test ----------------
def tier_cells(d):
    return [d["tiers"]["macro_f1"][t] for t in TIERS]


rows = []
for slug, label in MATRIX + BASE:
    e = ends(slug)
    if not e:
        continue
    rows.append((label, strat(e[-1]["val"]), tier_cells(test_set(slug))))

x = np.arange(len(TIERS)); w = 0.8 / len(rows)
fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), sharey=True)
for k, (label, vcells, tcells) in enumerate(rows):
    for ax, cells in zip(axes, (vcells, tcells)):
        ax.bar(x + k * w - 0.4 + w / 2, cells, w, label=label)
for ax, name in zip(axes, (L("验证集（66 篇 / 64,788 块）", "validation (66 docs)"),
                           L("测试集（62 篇 / 79,851 块）", "test (62 docs)"))):
    ax.set_xticks(x); ax.set_xticklabels(TIERS)
    ax.set_xlabel(L("标签支持度档", "Label-support tier")); ax.grid(alpha=.3, axis="y")
    ax.set_ylim(0, 1.04); ax.set_title(name, fontsize=10)
axes[0].set_ylabel("macro F1")
axes[0].legend(fontsize=8, ncol=2, loc="lower left", framealpha=.9)
fig.suptitle(L("按支持度分层的 macro F1（第 8 epoch；左=验证集，右=测试集）\n"
               "读法：主模型的优势集中在中间三档，且测试集上的分层优势比验证集更明显",
               "Support-stratified macro F1 (epoch 8)"), fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "fig2_support_stratified.png", dpi=200); plt.close(fig)

# ---------------- Fig 3: accuracy vs macro F1 ----------------
b3 = json.loads((PULL / "b3_cap_eval.json").read_text(encoding="utf-8"))
fig, ax = plt.subplots(figsize=(7.6, 5.8))
for slug, label in MATRIX:
    e = ends(slug)
    if not e:
        continue
    v = e[-1]["val"]
    ax.scatter(v["accuracy"], v["macro_f1"], s=90, marker="o", zorder=3)
    ax.annotate(label, (v["accuracy"], v["macro_f1"]), textcoords="offset points", xytext=(7, 4), fontsize=8)
for slug, label in BASE:
    e = ends(slug)
    if not e:
        continue
    v = e[-1]["val"]
    ax.scatter(v["accuracy"], v["macro_f1"], s=90, marker="s", zorder=3)
    ax.annotate(label, (v["accuracy"], v["macro_f1"]), textcoords="offset points", xytext=(7, 4), fontsize=8)
if "3000" in b3:
    m = b3["3000"]
    ax.scatter(b3["48"]["accuracy"], b3["48"]["macro_f1"], s=90, marker="s", color="crimson", zorder=3)
    ax.scatter(m["accuracy"], m["macro_f1"], s=150, marker="*", color="crimson", zorder=4)
    ax.annotate("B3 (cap3000, matched denominator)", (m["accuracy"], m["macro_f1"]),
                textcoords="offset points", xytext=(9, -14), fontsize=8, color="crimson")
    ax.annotate("", xy=(m["accuracy"], m["macro_f1"]),
                xytext=(b3["48"]["accuracy"], b3["48"]["macro_f1"]),
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2, ls=":"))
ax.set_xlabel("Validation accuracy"); ax.set_ylabel("macro F1"); ax.grid(alpha=.3)
ax.set_title(L("准确率 vs macro F1（块级，第 8 epoch）\n圆=矩阵行，方=基线，星=B3 匹配分母",
               "Accuracy vs macro F1 (block level, epoch 8)\ncircles=matrix, squares=baselines, star=B3 matched"))
fig.tight_layout(); fig.savefig(OUT / "fig3_acc_vs_macrof1.png", dpi=200); plt.close(fig)

# ---------------- Fig 4: D2 evidence (pooled per-epoch + raw mid-epoch) ----------------
POOL = ("delivered", "stage2_pilot_full_seed42")
pooled = {}
for slug in POOL:
    for r in ends(slug):
        pooled.setdefault(r["epoch"], []).append(r["val"])
eps = sorted(pooled)
p_acc = [float(np.mean([v["accuracy"] for v in pooled[e]])) for e in eps]
p_mf1 = [float(np.mean([v.get("macro_f1", 0) for v in pooled[e]])) for e in eps]
acc_peak_ep = eps[int(np.argmax(p_acc))]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
ax = axes[0]
ax.plot(eps, p_acc, marker="o", label="val accuracy")
ax.plot(eps, p_mf1, marker="s", label="macro F1")
ax.axvline(acc_peak_ep, color="crimson", ls=":", lw=1.3)
ax.set_ylim(bottom=0.15, top=max(max(p_acc), max(p_mf1)) + 0.11)
ax.annotate(L(f"acc 峰值 ep{acc_peak_ep}", f"acc peak ep{acc_peak_ep}"),
            (acc_peak_ep, max(p_acc)), textcoords="offset points", xytext=(6, 8),
            fontsize=8.5, color="crimson")
ax.annotate(L(f"macro F1 到 ep{eps[-1]} 仍上升（+{100*(p_mf1[-1]-p_mf1[-3]):.2f} pp vs ep{eps[-3]}）",
              f"macro F1 still rising at ep{eps[-1]}"),
            (eps[-1], p_mf1[-1]), textcoords="offset points", xytext=(-150, -22), fontsize=8.5)
ax.set_xlabel("epoch"); ax.set_ylabel("score"); ax.grid(alpha=.3); ax.set_xticks(eps); ax.legend()
ax.set_title(L("两行同配置同 seed 的逐 epoch 平均（D2 判峰依据）",
               "Per-epoch mean of the two same-config rows (D2 peak rule)"))

ax = axes[1]
for k, slug in enumerate(POOL):
    m = mids(slug)
    ax.plot(range(len(m)), [r["val"]["accuracy"] for r in m], marker=".", ms=4,
            lw=1, color=COLORS(k), label=slug)
ax.set_xlabel(L("mid-epoch 检查点（每 75 step）", "mid-epoch check (every 75 steps)"))
ax.set_ylabel("val accuracy"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title(L("两行原始 mid-epoch 轨迹：单点噪声极大，不能直接判峰",
               "Raw mid-epoch traces: single points are far too noisy to read a peak"))
fig.tight_layout(); fig.savefig(OUT / "fig4_D2_epoch_budget.png", dpi=200); plt.close(fig)

# ---------------- Fig 5: per-metric ablation delta vs noise floor ----------------
# test split, so the figure matches the thesis main table
ref_t = TCS["delivered/last"]["sets"]["test"]
labels, dacc, dmf1 = [], [], []
for slug, label in MATRIX:
    if slug == "delivered":
        continue
    if slug not in TCS.get(f"{slug}/last", {}) and f"{slug}/last" not in TCS:
        continue
    t = TCS[f"{slug}/last"]["sets"]["test"]
    labels.append(label)
    dacc.append(100 * (t["accuracy"] - ref_t["accuracy"]))
    dmf1.append(100 * (t["macro_f1"] - ref_t["macro_f1"]))

ZONE_REAL, ZONE_GREY = 6.0, 3.0          # D7 revised ladder


def zone(d):
    a = abs(d)
    return "real" if a > ZONE_REAL else ("grey" if a >= ZONE_GREY else "tie")


ZC = {"real": "#1f77b4", "grey": "#ff7f0e", "tie": "#bfbfbf"}
y = np.arange(len(labels))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, data, name in zip(axes, (dacc, dmf1), ("Δ accuracy (pp)", "Δ macro F1 (pp)")):
    ax.axvspan(-ZONE_GREY, ZONE_GREY, color="gray", alpha=.18,
               label=L(f"平手区 ±{ZONE_GREY:.0f} pp", f"tie band ±{ZONE_GREY:.0f} pp"))
    ax.axvspan(-ZONE_REAL, -ZONE_GREY, color="gray", alpha=.09)
    ax.axvspan(ZONE_GREY, ZONE_REAL, color="gray", alpha=.09)
    ax.barh(y, data, 0.55, color=[ZC[zone(d)] for d in data])
    for yy, d in zip(y, data):
        ax.text(d + (0.12 if d >= 0 else -0.12), yy, f"{d:+.2f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=8)
    ax.axvline(0, color="black", lw=.8)
    lo = min(list(data) + [0.0]); hi = max(list(data) + [0.0])
    pad = 0.24 * (hi - lo) + 0.25
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(name); ax.grid(alpha=.3, axis="x"); ax.legend(fontsize=8, loc="lower right")
axes[0].set_yticks(y); axes[0].set_yticklabels(labels); axes[0].invert_yaxis()
handles = [plt.Rectangle((0, 0), 1, 1, color=ZC[k]) for k in ("real", "grey", "tie")]
axes[1].legend(handles + [plt.Rectangle((0, 0), 1, 1, color="gray", alpha=.18)],
               [L(">6 pp 可分辨", ">6 pp real"), L("3–6 pp 灰色区", "3-6 pp grey"),
                L("<3 pp 平手", "<3 pp tie"), L("平手区 / 灰色区底纹", "tie / grey bands")],
               fontsize=7.5, loc="lower right")
fig.suptitle(L("消融效应 vs 运行间噪声底（测试集，第 8 epoch；蓝=可分辨，橙=灰色区，灰=平手）",
               "Ablation delta vs run-to-run noise floor (test, epoch 8)"), fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "fig5_noise_floor.png", dpi=200); plt.close(fig)

# ---------------- Fig 6: efficiency / scale ----------------
POINTS = [("b1", "B1\nroberta+MLP"), ("b2", "B2\n+layout"),
          ("b3", "B3\npage-BiGRU"), ("main", "HMSAN-BSA\n(main)")]


def test_mf1(rid):
    if rid == "main":
        return TCS["delivered/last"]["sets"]["test"]["macro_f1"]
    if rid == "b3":
        return TBS["b3|b3_page_bigru|last.pt|cap3000"]["sets"]["test"]["macro_f1"]
    key = "b1|b1_roberta_mlp|last.pt|cap48" if rid == "b1" \
        else "b2|b2_roberta_layout_mlp|last.pt|cap48"
    return TBS[key]["sets"]["test"]["macro_f1"]


PARAM = [EFF[rid]["params_trainable"] / 1e6 for rid, _ in POINTS]
F1 = [test_mf1(rid) for rid, _ in POINTS]
HOURS = [EFF[rid]["train_hours_per_epoch"] for rid, _ in POINTS]
NAMES = [lab for _, lab in POINTS]
bar_colors = ["#9d9d9d", "#9d9d9d", "#7fbf7f", "#d62728"]

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9))
for ax, vals, title, fmt in (
        (axes[0], PARAM, L("可训练参数（百万）—— 四者几乎相同", "trainable params (M)"), "{:.1f}"),
        (axes[1], F1, L("测试集 macro F1 —— 差距全在这里", "test macro F1"), "{:.3f}"),
        (axes[2], HOURS, L("每轮训练时间（小时，同一台 RTX 3060）", "train h / epoch"), "{:.2f}")):
    xs = np.arange(len(vals))
    ax.bar(xs, vals, 0.62, color=bar_colors)
    for xx, vv in zip(xs, vals):
        ax.text(xx, vv, fmt.format(vv), ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(xs); ax.set_xticklabels(NAMES, fontsize=8.5)
    ax.set_title(title, fontsize=10); ax.grid(alpha=.3, axis="y")
    ax.set_ylim(0, max(vals) * 1.22)
axes[0].annotate(L("主模型只比 B3 多 1.8%", "only +1.8% vs B3"),
                 xy=(3, PARAM[3]), xytext=(1.35, PARAM[3] * 1.13), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", lw=1.0))
axes[1].annotate(L("+5.4 pp", "+5.4 pp"), xy=(3, F1[3]), xytext=(1.2, F1[3] * 0.99),
                 fontsize=9, color="#d62728",
                 arrowprops=dict(arrowstyle="->", lw=1.0, color="#d62728"))
fig.suptitle(L("效率与规模：主模型的增益不来自参数规模，每轮训练成本只比 B3 高约 6%\n"
               "（不要写\"更轻量 / 推理更快\"——主模型的总参数与推理时间都更大）",
               "Efficiency and scale"), fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "fig6_efficiency.png", dpi=200); plt.close(fig)

# ---------------- summary ----------------
print(f"font={CJK}")
print(f"\n[D2] pooled per-epoch (delivered + pilot):")
for e, a, m in zip(eps, p_acc, p_mf1):
    print(f"   ep{e}: acc={a:.4f}  macroF1={m:.4f}" + ("   <- acc peak" if e == acc_peak_ep else ""))
print(f"   acc peak at ep{acc_peak_ep}; macro F1 ep{eps[-3]}->ep{eps[-1]}: "
      f"{p_mf1[-3]:.4f}->{p_mf1[-1]:.4f} ({100*(p_mf1[-1]-p_mf1[-3]):+.2f} pp)")
print("\n[Ablation, last epoch vs delivered]")
for lb, da, dm in zip(labels, dacc, dmf1):
    print(f"   {lb:24s} dacc={da:+6.2f} [{zone(da):4s}]  dmF1={dm:+6.2f} [{zone(dm):4s}]")
print("\nwritten to", OUT)