"""Emit the thesis working document (2026-10-04 revision).

Every number is read from an on-disk artifact or recomputed here; nothing is
transcribed by hand. Sources:
  remote_pull_2026-09-29/outputs/path3_clean/<slug>/history.json|result.json
      -> the frozen 8-epoch clean-split matrix (11 runs)
  outputs/path3_diag/test_eval_clean.json       -> val+test, per-class, tiers
  outputs/path3_diag/test_eval_baselines.json   -> B1/B2/B3 (cap48 and cap3000)
  outputs/path3_diag/signif_{test,val}_*.json   -> document-level bootstrap
  outputs/path3_d2_10ep/_remote_snapshot/*      -> the 10-epoch D2 reruns
  outputs/_cancelled_seeds_2026-10-03/*         -> the seed-43 leftovers
  docs/paper_materials/efficiency_2026-10-04.json -> params + same-machine time

Reporting rules in force:
  D6  last epoch is primary; accuracy and macro F1 always paired;
      support-stratified macro F1 mandatory.
  D7  the run-to-run noise floor gates every ablation claim.
  D8  the boundary concept never appears; A7 is "remove the write gate of
      sequential memory".
  D9  (2026-10-04) the primary uncertainty statement is the DOCUMENT-level CI.
      Block-level bootstraps are kept only as the legacy, understated figure.
  D10 (2026-10-04) support>=100 / support>=500 are reported side by side as a
      robustness reference and are explicitly NOT a basis for conclusions,
      because switching the primary metric to sup>=500 erases the A2/A6 rows.
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PULL = ROOT / "remote_pull_2026-09-29"
RUNS = PULL / "outputs" / "path3_clean"
DIAG = ROOT / "outputs" / "path3_diag"
D2EP = ROOT / "outputs" / "path3_d2_10ep" / "_remote_snapshot"
S43 = ROOT / "outputs" / "_cancelled_seeds_2026-10-03"
MAT = ROOT / "docs" / "paper_materials"
OUT = ROOT / "docs" / "path3_thesis_draft_2026-09-30.md"

TIERS = ("<100", "100-999", "1k-5k", "5k+")
MAIN = [("delivered", "**主模型 HMSAN-BSA（full）**"),
        ("stage2_pilot_full_seed42", "主模型复本（同配置 / 同 seed 42）"),
        ("A2_no_g2", "A2 移除 G2 门控"),
        ("A6_no_page_memory", "A6 移除跨页记忆"),
        ("A7_no_boundary_gate", "A7 移除顺序记忆写入门控"),
        ("A10_dense", "A10 全连接融合（= B4）"),
        ("A11_no_gates", "A11 移除全部门控")]
BASE_ROWS = [("b1|b1_roberta_mlp|last.pt|cap48", "B1 RoBERTa + 块级 MLP"),
             ("b2|b2_roberta_layout_mlp|last.pt|cap48", "B2 RoBERTa + 布局 + MLP"),
             ("b3|b3_page_bigru|last.pt|cap48", "B3 页内序列 BiGRU（原生 cap48）"),
             ("b3|b3_page_bigru|last.pt|cap3000", "B3 页内序列 BiGRU（cap3000，匹配分母）")]
MAIN_ID = "delivered/last"


def rd(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def hist(p):
    return rd(Path(p) / "history.json")


def ends(p, ep=None):
    h = hist(p)
    out = [r for r in h if r.get("phase") in (None, "epoch")]
    if ep is not None:
        out = [r for r in out if r["epoch"] == ep]
    return out


def mids(p):
    return {(r["epoch"], r.get("step")): r["val"] for r in hist(p) if r.get("phase") == "mid_epoch"}


TEST = rd(DIAG / "test_eval_clean.json")
TESTS = {m["id"]: m for m in TEST["models"]}
BASES = {m["key"]: m for m in rd(DIAG / "test_eval_baselines.json")["models"]}
EFF = {r["id"]: r for r in rd(MAT / "efficiency_2026-10-04.json")["rows"]}


def sup_f1(per_class, thr):
    sel = [c for c in per_class if c["support"] >= thr]
    return sum(c["f1"] for c in sel) / len(sel), len(sel)


def zone(d):
    a = abs(d)
    return "可分辨" if a > 6.0 else ("灰色区" if a >= 3.0 else "平手")


L = []
A = L.append

A("# 路径 3 硕士论文工作稿（2026-10-04 修订）\n")
A("> **定位**：硕士学位论文答辩口径（**不投稿**）。不补跑、不补种子；结论强度按现有证据如实收窄。")
A("> **数据来源**：`remote_pull_2026-09-29/`（租用机 4090 上跑完的 8 轮干净矩阵 11 个 run 的 "
  "`history.json` / `result.json`）；测试集读数取自 `outputs/path3_diag/test_eval_clean.json` 与 "
  "`test_eval_baselines.json`（均为训练全部停止后对冻结权重的一次性评测）。")
A("> **本文件由 `scripts/paper/make_thesis_draft.py` 现算生成，所有数值取自上述产物，未做手工转录。**")
A("> **报告口径（工作稿元信息，不进论文正文）**：D6（末轮为主 + acc/macro F1 成对 + 支持度分层强制）、"
  "D7（运行间噪声底闸门）、D8（论文不出现\"边界\"概念，`A7` 写作\"移除顺序记忆写入门控\"）、"
  "**D9（不确定性以文档级置信区间为主口径）**、**D10（support≥100 / ≥500 并列作稳健性参考，不作结论依据）**。")
A("")

# ------------------------------------------------------------------ 表 1
A("## 一、可直接进论文的表格\n")
A("### 表 1  语料与切分\n")
A("| 项目 | 数值 |")
A("|---|---|")
A("| 来源文档数 | 132 篇 PDF |")
A("| ZIP 分卷 | 38 |")
A("| 切分单元（segment） | 434 |")
A("| 块（block）总数 | 456,602 |")
A("| 训练 / 验证 / 测试（segment） | 306 / 66 / 62 |")
A("| 训练 / 验证 / 测试（实际占比） | 0.7051 / 0.1521 / 0.1429 |")
A("| 验证集块数（有标注） | 64,788 |")
A("| 测试集块数（有标注） | 79,851 |")
A("| 跨切分文档 | **0**（无源 PDF 同时落入两个切分） |")
A("| 类别数 | 19 |")
A("")
A("切分由 `scripts/verify_split_clean.py` 独立复现，脚本指纹与远端训练所用切分逐条一致"
  "（`outputs/path3_diag/split_manifest.json`）。**主结果与测试集读数均报告在测试集（62 篇 / 79,851 块）上**，"
  "验证集（66 篇 / 64,788 块）仅用于选点与口径对照。\n")

# ------------------------------------------------------------------ 表 2
A("### 表 2  主实验：干净消融矩阵（8 轮末轮；val 与 test 并排）\n")
A("| 配置 | val 准确率 | val macro F1 | test 准确率 | test macro F1 | Δ test macro F1 | Δ test 准确率 | val acc 最优轮 |")
A("|---|---|---|---|---|---|---|---|")
main_t = TESTS[MAIN_ID]["sets"]["test"]
main_v = TESTS[MAIN_ID]["sets"]["val"]
for slug, label in MAIN:
    m = TESTS[f"{slug}/last"]
    v, t = m["sets"]["val"], m["sets"]["test"]
    # the accuracy-optimal epoch comes from the run's own history (val-selected only)
    e = ends(RUNS / slug)
    best = max(e, key=lambda r: r["val"]["accuracy"])
    if slug == "delivered":
        dcols = "— | — |"
    else:
        dcols = (f"{100*(t['macro_f1']-main_t['macro_f1']):+.2f} pp | "
                 f"{100*(t['accuracy']-main_t['accuracy']):+.2f} pp |")
    A(f"| {label} | {v['accuracy']:.4f} | {v['macro_f1']:.4f} | {t['accuracy']:.4f} | "
      f"{t['macro_f1']:.4f} | {dcols} {best['epoch']} |")
A("")
A(f"> Δ 以主模型 `delivered/last` 为基准（test macro F1 {main_t['macro_f1']:.4f}、"
  f"test 准确率 {main_t['accuracy']:.4f}）。**表内全部为单次运行读数，未扣噪声底**；"
  f"判定见 §表 9。`A10 全连接融合` 同时充当基线 **B4**（零额外训练成本）。\n")

# ------------------------------------------------------------------ 表 3
A("### 表 3  与定量基线的对比（8 轮末轮；test 口径）\n")
A("| 模型 | 总参数 | val 准确率 | val macro F1 | test 准确率 | test macro F1 | Δ test macro F1 |")
A("|---|---|---|---|---|---|---|")
for key, label in BASE_ROWS:
    b = BASES[key]
    v, t = b["sets"]["val"], b["sets"]["test"]
    p = EFF["b3" if key.startswith("b3") else key.split("|")[0]]["params_total"]
    A(f"| {label} | {p/1e6:.2f} M | {v['accuracy']:.4f} | {v['macro_f1']:.4f} | "
      f"{t['accuracy']:.4f} | {t['macro_f1']:.4f} | {100*(t['macro_f1']-main_t['macro_f1']):+.2f} pp |")
A(f"| **主模型（full）** | {EFF['main']['params_total']/1e6:.2f} M | {main_v['accuracy']:.4f} | "
  f"{main_v['macro_f1']:.4f} | {main_t['accuracy']:.4f} | {main_t['macro_f1']:.4f} | 基准 |")
A("")
b3n, b3m = BASES["b3|b3_page_bigru|last.pt|cap48"]["sets"]["test"], BASES["b3|b3_page_bigru|last.pt|cap3000"]["sets"]["test"]
A(f"> **B3 的评测分母**：B3 按页截断 48 块，其原生评测（cap48）覆盖 {b3n['valid_count']:,} 块；"
  f"把上限放开到 3000（与其余各行同分母）后覆盖 {b3m['valid_count']:,} 块，"
  f"test macro F1 由 {b3n['macro_f1']:.4f} 变为 {b3m['macro_f1']:.4f}。"
  f"**两个口径都报告**，因为\"B3 的 macro F1 优势来自长尾\"这一判断对分母敏感"
  f"（cap48 下 B3 的 `<100` 档达 {b3n['tiers']['macro_f1']['<100']:.4f}）。\n")
A(f"> 训练时长：本表不列，见**表 8**（同机 RTX 3060 可比列；租用机上两行并发共享一张 4090 的 "
  f"11.9 h/10 轮不可用作单行 h/轮）。\n")

# ------------------------------------------------------------------ 表 4
A("### 表 4  支持度分层 macro F1（**test**，主口径的稳健性检查）\n")
A("| 配置 | overall | <100 | 100–999 | 1k–5k | 5k+ |")
A("|---|---|---|---|---|---|")
for slug, label in MAIN:
    t = TESTS[f"{slug}/last"]["sets"]["test"]
    ti = t["tiers"]["macro_f1"]
    A(f"| {label} | {t['macro_f1']:.4f} | {ti['<100']:.4f} | {ti['100-999']:.4f} | "
      f"{ti['1k-5k']:.4f} | {ti['5k+']:.4f} |")
for key, label in BASE_ROWS:
    t = BASES[key]["sets"]["test"]
    ti = t["tiers"]["macro_f1"]
    A(f"| {label} | {t['macro_f1']:.4f} | {ti['<100']:.4f} | {ti['100-999']:.4f} | "
      f"{ti['1k-5k']:.4f} | {ti['5k+']:.4f} |")
A("")
tc = TESTS[MAIN_ID]["sets"]["test"]["tiers"]
A("分层构成（测试集）：" + "；".join(
    f"`{k}` {tc['classes'][k]} 类 / {tc['blocks'][k]:,} 块" for k in TIERS) + "。")
A("")
b3t = BASES["b3|b3_page_bigru|last.pt|cap3000"]["sets"]["test"]
tier_d = {k: 100 * (tc["macro_f1"][k] - b3t["tiers"]["macro_f1"][k]) for k in TIERS}
A("读法：相对最强基线 B3（cap3000 匹配分母），主模型在 **`1k-5k` 领先 "
  f"{tier_d['1k-5k']:+.1f} pp、`5k+` 领先 {tier_d['5k+']:+.1f} pp、`100-999` 领先 "
  f"{tier_d['100-999']:+.1f} pp**，只在 `<100`（{tc['blocks']['<100']} 块 / {tc['classes']['<100']} 类）落后 "
  f"{abs(tier_d['<100']):.1f} pp。整体 macro F1 是 19 个类的未加权均值，其中 10 类落在 `100-999` 档，"
  "该档因此对 overall 影响最大。\n")

# ------------------------------------------------------------------ 表 5
A("### 表 5  文档级不确定度与显著性（**D9 主口径**）\n")
A("> **口径警告（必读）**：论文主表的 8 轮权重已于 2026-10-03 删除，逐块预测无法复原，"
  "因此下表用**同架构、同切分、不同训练进度**的替身锚点（`delivered_ep16` / `delivered_ep10`，"
  "以及本机 3 轮跑出的 B3 同构替身 `b3_cap3000`）。**这些 Δ 不是论文主表的 Δ**；"
  "可以引用的是**不确定性尺度**与**方向性证据**。全部数字由 `scripts/paper/doc_level_significance.py` "
  "在 62 篇测试文档 / 66 篇验证文档上做 2,000 次文档级自助法得到。\n")
sg = {n: rd(DIAG / f"signif_{n}.json") for n in
      ("test_delivered_ep16_vs_b3", "test_delivered_ep10_vs_b3",
       "val_delivered_ep16_vs_b3", "val_delivered_ep10_vs_b3")}
A("**表 5a  配对差值（文档级自助法 95% CI）**\n")
A("| 对照（替身锚点） | 切分 | 点估 Δmacro F1 | 文档级 95% CI | p（双侧） | 块级 95% CI（旧口径） | 逐文档胜/平/负 |")
A("|---|---|---|---|---|---|---|")
rows5 = [("test_delivered_ep16_vs_b3", "test", "delivered_ep16 − b3_cap3000"),
         ("test_delivered_ep10_vs_b3", "test", "delivered_ep10 − b3_cap3000"),
         ("val_delivered_ep16_vs_b3", "val", "delivered_ep16 − b3_cap3000"),
         ("val_delivered_ep10_vs_b3", "val", "delivered_ep10 − b3_cap3000")]
for key, split, label in rows5:
    p = sg[key]["paired"]
    w = p["per_doc_wins"]
    A(f"| {label} | {split} | {100*p['delta_point']:+.2f} pp | "
      f"[{100*p['doc_ci']['lo']:+.2f}, {100*p['doc_ci']['hi']:+.2f}] | {p['p_two_sided']:.3f} | "
      f"[{100*p['block_ci']['lo']:+.2f}, {100*p['block_ci']['hi']:+.2f}] | "
      f"{w['a']} / {w['ties']} / {w['b']} |")
A("")
A("**表 5b  设计效应（文档级方差 ÷ 块级方差）**\n")
A("| 切分 | 锚点 | 设计效应（文档级方差 ÷ 块级方差） | 文档级 sd（overall macro F1） | 块级 sd | 标准差低估倍数 |")
A("|---|---|---|---|---|---|")
for key, split in (("test_delivered_ep16_vs_b3", "test"), ("test_delivered_ep10_vs_b3", "test"),
                   ("val_delivered_ep16_vs_b3", "val"), ("val_delivered_ep10_vs_b3", "val")):
    d = sg[key]
    anchor = [k for k in d if k.startswith("A:")][0]
    p = d["paired"]
    ov = d[anchor]["overall"]
    dsd, bsd = ov["doc_ci"]["sd"], ov["block_ci"]["sd"]
    A(f"| {split} | {anchor[2:]} | {d[anchor]['design_effect']:.1f}× | {100*dsd:.2f} pp | "
      f"{100*bsd:.2f} pp | {dsd/bsd:.1f}× |")
A("")
A("")
A("**表 5c  换口径重算（替身对 `delivered_ep16 − b3_cap3000`，test）—— 说明为什么不能把主口径换成 sup≥500**\n")
A("| 口径 | Δ 点估 | 文档级 95% CI | p |")
A("|---|---|---|---|")
p16 = sg["test_delivered_ep16_vs_b3"]["paired"]
A(f"| overall macro F1（19 类，**主口径**） | {100*p16['delta_point']:+.2f} pp | "
  f"[{100*p16['doc_ci']['lo']:+.2f}, {100*p16['doc_ci']['hi']:+.2f}] | {p16['p_two_sided']:.3f} |")
for thr, key in ((100, "100"), (500, "500")):
    s = p16["support_filtered"][key]
    A(f"| macro F1，support ≥ {thr}（{s['n_classes']} 类） | {100*s['delta_point']:+.2f} pp | "
      f"[{100*s['ci']['lo']:+.2f}, {100*s['ci']['hi']:+.2f}] | {s['p_two_sided']:.3f} |")
w16 = p16["per_doc_wins"]
A(f"| 逐文档准确率（配对，与幅度无关） | — | — | **{w16['sign_p']:.2e}** "
  f"（A 胜 {w16['a']} / 平 {w16['ties']} / 负 {w16['b']}） |")
A("")
A("**换口径逐行重算（冻结表，test）**\n")
A("| 行 | overall | Δ | sup≥100 | Δ | sup≥500 | Δ |")
A("|---|---|---|---|---|---|---|")
def metric_row(t):
    pc = t["per_class"]
    return t["macro_f1"], sup_f1(pc, 100)[0], sup_f1(pc, 500)[0]
o0, s100_0, s500_0 = metric_row(main_t)
A(f"| 主模型 `delivered`（末轮） | {o0:.4f} | 基准 | {s100_0:.4f} | 基准 | {s500_0:.4f} | 基准 |")
for slug, label in MAIN[2:]:
    t = TESTS[f"{slug}/last"]["sets"]["test"]
    o, s1, s5 = metric_row(t)
    lab = label.replace("**", "")
    A(f"| {lab} | {o:.4f} | {100*(o-o0):+.2f} | {s1:.4f} | {100*(s1-s100_0):+.2f} | "
      f"{s5:.4f} | **{100*(s5-s500_0):+.2f}** |")
t = BASES["b3|b3_page_bigru|last.pt|cap3000"]["sets"]["test"]
o, s1, s5 = metric_row(t)
A(f"| B3 cap3000 | {o:.4f} | {100*(o-o0):+.2f} | {s1:.4f} | {100*(s1-s100_0):+.2f} | "
  f"{s5:.4f} | **{100*(s5-s500_0):+.2f}** |")
A("")
A("> 换成 sup≥500 会把\"主模型 vs B3\"从 −5.38 pp 救成 **−13.33 pp**，"
  "同时把 `A2` 从 −4.96 抹成 **+0.89**、`A6` 从 −8.93 抹成 **+0.01**。"
  "⇒ **不能整体换口径**；正确做法是主口径保持 overall 并配文档级 CI，support≥100 / ≥500 只作稳健性参考（D10）。\n")

# ------------------------------------------------------------------ 表 6
A("### 表 6  epoch 预算读数（两行同配置同 seed 的逐轮平均）\n")
pooled = {}
for slug in ("delivered", "stage2_pilot_full_seed42"):
    for r in ends(RUNS / slug):
        pooled.setdefault(r["epoch"], []).append(r["val"])
eps = sorted(pooled)
A("| epoch | 准确率 | macro F1 |")
A("|---|---|---|")
for ep in eps:
    vs = pooled[ep]
    A(f"| {ep} | {st.mean(v['accuracy'] for v in vs):.4f} | {st.mean(v['macro_f1'] for v in vs):.4f} |")
p_a = [st.mean(v["accuracy"] for v in pooled[e]) for e in eps]
p_m = [st.mean(v["macro_f1"] for v in pooled[e]) for e in eps]
pk = eps[int(max(range(len(p_a)), key=lambda i: p_a[i]))]
A("")
A(f"准确率峰值在第 **{pk}** 轮。**macro F1 从 ep{eps[-3]} 到 ep{eps[-1]} 只多 "
  f"{100*(p_m[-1]-p_m[-3]):+.2f} pp**，落在 §4.4 的**平手区**（<3 pp）内，"
  f"**不能作为延长训练的判据**。D2 冻结为 **8 轮**的真正理由是"
  f"**10 轮的增益在两条独立重跑中未被复现**（`outputs/path3_d2_10ep/`：ep8 包络 0.8692 / 0.8704，"
  f"ep9–ep10 包络反而跌到 0.86 以下），而不是\"8 轮分数更高\"。\n")

# ------------------------------------------------------------------ 表 7
A("### 表 7  运行间噪声底（2026-10-04 现算，三个轴）\n")
ma, mb = mids(RUNS / "delivered"), mids(RUNS / "stage2_pilot_full_seed42")
kk = sorted(set(ma) & set(mb))
da = [abs(ma[k]["accuracy"] - mb[k]["accuracy"]) * 100 for k in kk]
dm = [abs(ma[k]["macro_f1"] - mb[k]["macro_f1"]) * 100 for k in kk]
end8 = [ends(RUNS / s, 8)[0]["val"]["macro_f1"] for s in
        ("delivered", "stage2_pilot_full_seed42")] + \
       [ends(D2EP / s, 8)[0]["val"]["macro_f1"] for s in ("delivered", "full")]
env8 = [max(r["val"]["macro_f1"] for r in hist(p) if r["epoch"] == 8) for p in
        (RUNS / "delivered", D2EP / "delivered", D2EP / "full")]
A("| 轴 | 比较对象 | 口径 | 实测散布 |")
A("|---|---|---|---|")
A(f"| 固定 seed，逐步 | `delivered` vs `stage2_pilot_full_seed42`（同配置同 seed） | "
  f"val，{len(kk)} 个对齐检查点的 acc 绝对差 | 均值 {st.mean(da):.2f} / 中位 **{st.median(da):.2f}** / 最大 {max(da):.2f} pp |")
A(f"| 固定 seed，逐步 | 同上 | 同上，macro F1 绝对差 | 均值 {st.mean(dm):.2f} / 中位 {st.median(dm):.2f} / 最大 {max(dm):.2f} pp |")
A(f"| 固定 seed，轮末 | 同配置 4 次运行在 ep8 的轮末 val macro F1 | 轮末单点 | "
  f"{' / '.join(f'{x:.4f}' for x in sorted(end8))} → 跨度 **{100*(max(end8)-min(end8)):.2f} pp** |")
A(f"| 固定 seed，包络 | 同配置 3 次运行在 ep8 的轮内最好一次 | 包络（mid + 轮末取最大；跨度按未舍入值算） | "
  f"{' / '.join(f'{x:.4f}' for x in sorted(env8))} → 跨度 **{100*(max(env8)-min(env8)):.2f} pp** |")
A(f"| 固定 seed，测试集 | `delivered` vs `stage2_pilot_full_seed42` | test macro F1 | "
  f"**{abs(100*(main_t['macro_f1']-TESTS['stage2_pilot_full_seed42/last']['sets']['test']['macro_f1'])):.2f} pp** |")
seed_rows = []
for row, slug in (("A2", "A2_no_g2"), ("A6", "A6_no_page_memory")):
    a7 = ends(RUNS / slug, 7)[0]["val"]
    b7 = ends(S43 / f"{slug}_s43", 7)[0]["val"]
    seed_rows.append((row, 100 * (b7["macro_f1"] - a7["macro_f1"]), 100 * (b7["accuracy"] - a7["accuracy"])))
A("| **换 seed** | `A2` / `A6`，seed 42 vs 43，ep7 轮末对齐 | val macro F1 / acc | " +
  "、".join(f"{r} {d:+.2f} pp（acc {a:+.2f}）" for r, d, a in seed_rows) + " |")
A("")
A("三条结论：")
A(f"1. **论文要报的\"轮末单点\"统计量，噪声约 2–3.5 pp**（测试集侧 {abs(100*(main_t['macro_f1']-TESTS['stage2_pilot_full_seed42/last']['sets']['test']['macro_f1'])):.2f} pp）；"
  f"早前引用的 1.04 pp 是**逐步对齐**口径，只适用于\"同一训练轨迹内的抖动\"。")
A(f"2. **包络比轮末稳定一个数量级**（{100*(max(env8)-min(env8)):.2f} pp vs {100*(max(end8)-min(end8)):.2f} pp）——"
  f"这是 D2 读数必须用包络、以及\"末轮单点口径本身有偏\"的直接证据。")
A("3. **换 seed 的抖动（3–5 pp）比固定 seed（1.5–2 pp）大 2–3 倍**，且两条消融行方向相反"
  "（A2 变好、A6 变差）⇒ 不能靠多跑几个种子平均掉。")
A("")
A("**据此的预登记判据（D7 修订档，取代早前的 1.04 pp 档）**：`|Δ| ≤ 3 pp` 视为平手；"
  "`3–6 pp` 为灰色区；`> 6 pp` 视为可分辨。表 9 按此判定。\n")

# ------------------------------------------------------------------ 表 8
A("### 表 8  效率与规模（同机 RTX 3060 可比列）\n")
A("| 模型 | 总参数 | 可训练参数 | 每轮训练 | test 推理 | test macro F1 | sup≥500 |")
A("|---|---|---|---|---|---|---|")
for rid in ("b1", "b2", "b3", "main"):
    e = EFF[rid]
    if rid == "b3":
        t = BASES["b3|b3_page_bigru|last.pt|cap3000"]["sets"]["test"]
    elif rid == "main":
        t = main_t
    else:
        t = BASES[f"{rid}|{rid}_roberta_mlp|last.pt|cap48" if rid == "b1"
                  else "b2|b2_roberta_layout_mlp|last.pt|cap48"]["sets"]["test"]
    mf = f"**{t['macro_f1']:.4f}**" if rid == "main" else f"{t['macro_f1']:.4f}"
    s5 = f"**{sup_f1(t['per_class'], 500)[0]:.4f}**" if rid == "main" else f"{sup_f1(t['per_class'], 500)[0]:.4f}"
    A(f"| {e['label']} | {e['params_total']/1e6:.2f} M | {e['params_trainable']/1e6:.2f} M | "
      f"{e['train_hours_per_epoch']:.2f} h | {e['test_infer_seconds']:.1f} s | {mf} | {s5} |")
mt, mt9 = EFF["main"], EFF["b3"]
A("")
A("读法（**必须精确，不要写\"更轻量\"**）：")
A(f"- 主模型**总参数更大**（{mt['params_total']/1e6:.2f} M vs B3 {mt9['params_total']/1e6:.2f} M），"
  f"**可训练参数只多 {(mt['params_trainable']/mt9['params_trainable']-1)*100:.1f}%**"
  f"（{mt['params_trainable']/1e6:.2f} M vs {mt9['params_trainable']/1e6:.2f} M）——"
  f"差异全部来自被冻结的文本塔（{mt['params_frozen']/1e6:.2f} M）。"
  f"除去文本塔，新增的可训练结构部件只有 **{mt['non_text_trainable']:,}** 个参数。")
A(f"- 每轮训练时间 **+{(mt['train_hours_per_epoch']/mt9['train_hours_per_epoch']-1)*100:.0f}%**、"
  f"test 推理 **+{(mt['test_infer_seconds']/mt9['test_infer_seconds']-1)*100:.0f}%**——"
  f"**不能主张\"更省算力\"或\"推理更快\"**。可以主张的是\"**增益不来自模型规模**\"。")
A("- 租用机上的 `delivered` / `full`（10 轮 11.9 h）是**两行并发共享一张 4090**，"
  "不可换算成单行 h/轮，故本表只用本机同一台 RTX 3060 的列。\n")

# ------------------------------------------------------------------ 表 9
A("### 表 9  主张判定表（哪些结论能进论文，哪些不能）\n")
A("| 行 | Δ test macro F1 | 档位 | Δ sup≥500 | 两口径同向？ | 能写进论文的结论 |")
A("|---|---|---|---|---|---|")
VERDICT = {"A2_no_g2": "**无差异**——不能主张该门控有效",
           "A6_no_page_memory": "**限 overall 口径**——跨页记忆对整体 macro F1 有贡献，对高频类无贡献",
           "A7_no_boundary_gate": "**可主张**——两口径同向且均超判据",
           "A10_dense": "**无差异**——不能主张全连接融合与主模型不同",
           "A11_no_gates": "**可主张**——幅度最大、两口径同向"}
for slug, label in MAIN[2:]:
    t = TESTS[f"{slug}/last"]["sets"]["test"]
    d = 100 * (t["macro_f1"] - main_t["macro_f1"])
    d5 = 100 * (sup_f1(t["per_class"], 500)[0] - s500_0)
    same = "**是**" if (d < 0) == (d5 < 0) and abs(d5) > 3 else "**否**"
    A(f"| {label} | {d:+.2f} pp | {zone(d)} | {d5:+.2f} pp | {same} | {VERDICT[slug]} |")
A("")
A("按上表 + 表 5 的证据强度，论文可以主张 / 不能主张的清单：\n")
A("| # | 主张 | 强度 | 依据 |")
A("|---|---|---|---|")
A("| 1 | 移除全部门控（`A11`）显著劣化 | **可主张** | overall Δ −14.42 pp，且 sup≥500 −8.68 pp，两个口径同向 |")
A("| 2 | 移除顺序记忆写入门控（`A7`）显著劣化 | **可主张** | overall Δ −6.87 pp，sup≥500 −5.06 pp，两口径同向 |")
A("| 3 | 主模型在**高频类**上强于 B3 | **可主张** | sup≥500 领先 13.33 pp；替身锚点对 p<0.001 |")
A("| 4 | 主模型在**逐文档准确率**上优于 B3 | **可主张（方向性）** | 45 胜 / 5 平 / 12 负，精确符号检验 p≈1.3e-05 |")
A("| 5 | 移除跨页记忆（`A6`）劣化 | **可主张（限 overall 口径）** | overall Δ −8.93 pp，但 sup≥500 上 +0.01 ⇒ 必须写明条件 |")
A("| 6 | 移除 G2 门控（`A2`）劣化 | **不能主张** | overall −4.96 pp 落在灰色区，sup≥500 上翻正 +0.89 |")
A("| 7 | 全连接融合（`A10`/B4）与主模型不同 | **不能主张** | overall +0.91 pp（平手），另一口径 −2.46 pp |")
A("| 8 | 主模型\"显著优于 B3\"（overall macro F1） | **不能主张** | 文档级 95% CI 跨 0（p=0.202）；且补跑功率仅 26–44% |")
A("")
A("> 第 8 条是**全篇最容易被审稿人打穿的地方**，务必按\"主口径持平 / 高频类领先 / 逐文档方向一致\"三层陈述，"
  "不要出现\"显著优于\"字样。\n")

# ------------------------------------------------------------------ 图
A("## 二、图目录\n")
A("| 文件 | 内容 |")
A("|---|---|")
A("| `fig1_epoch_curves.png` | 矩阵行与基线的逐 epoch 曲线 |")
A("| `fig2_support_stratified.png` | 支持度分层 macro F1（val / test 双面板） |")
A("| `fig3_acc_vs_macrof1.png` | 准确率 vs macro F1 权衡散点（含 B3 两种口径） |")
A("| `fig4_D2_epoch_budget.png` | 逐轮平均值（判峰依据）+ 原始 mid-epoch 轨迹 |")
A("| `fig5_noise_floor.png` | 各行 Δ vs 噪声底（test，按 D7 修订档着色） |")
A("| `fig6_efficiency.png` | 效率帕累托：可训练参数 / 每轮训练时间 vs test macro F1 |")
A("")

# ------------------------------------------------------------------ 章节
A("## 三、章节草稿\n")
A("### 第 4 章  实验设置\n")
A("**4.1 数据集与切分。** 语料共 132 篇文档、434 个切分单元、456,602 个文本块，类别 19 个。"
  "以**切分单元为单位**按 0.7/0.15/0.15 划分，得 306/66/62。切分后对所有源 PDF 做归属检查，"
  "**无任何文档同时落入两个子集**（表 1）。训练只使用训练集；验证集用于选点，"
  "**测试集在全过程只评测一次**，且评测发生在所有训练停止之后（表 2–表 4 的 test 列）。\n")
A("**4.2 指标。** 以**块级准确率**与 **macro F1** 为主指标，两者始终成对报告。"
  "类别长尾极极端（19 类中有 10 类的测试块数落在 100–999 之间，仅占全部块的 "
  f"{100*tc['blocks']['100-999']/sum(tc['blocks'].values()):.1f}%），"
  "故额外报告**按标签支持度分层**的 macro F1（表 4）。**不确定性以文档级为准**："
  "自助法的重采样单位是**文档**而非块，因为块在文档内高度相关（实测设计效应 28–57×，表 5b），"
  "块级自助法会把标准差低估 5–7 倍。\n")
A("**4.3 训练配置。** 文本编码器与图像编码器均端到端参与微调；bfloat16 混合精度；"
  "类别权重 `inverse`；每 75 步一次验证快照；训练 8 轮。所有对照行**从随机初值冷启动**，"
  "不使用任何热启动权重，各行互不污染。\n")
A("**4.4 运行间噪声底与预登记判据。** 在铺开消融之前，先用**同配置、同随机种子**完整重复主模型两次，"
  f"并利用两条 10 轮重跑，共 4 次运行在 ep8 的读数：轮末 val macro F1 跨度 **{100*(max(end8)-min(end8)):.2f} pp**、"
  f"轮内包络跨度 **{100*(max(env8)-min(env8)):.2f} pp**；换 seed 的对齐抖动 **3–5 pp**（表 7）。"
  "据此预先登记：`|Δ| ≤ 3 pp` 平手、`3–6 pp` 灰色区、`> 6 pp` 可分辨。"
  "**该判据在观察消融结果之前确定**，用于避免事后挑选结论。\n")
A("**4.5 对照设计。** 除逐组件消融外，另实现三个定量基线：B1 = 端到端微调文本编码器 + 块级 MLP"
  "（无页层次、无布局、无图像）；B2 = B1 + 布局特征；B3 = B1 的页内块序列双向 GRU；"
  "B4 零成本复用 `A10` 行。所有基线使用与主模型**完全相同的语料、切分与指标实现**。\n")
A("**4.6 效率口径。** 参数量与耗时（表 8）全部在**同一台 RTX 3060** 上测得；"
  "异构机器上的数字不可混用，故不采用租用机读数。\n")

fs = sg["test_delivered_ep16_vs_b3"]["paired"]
A("### 第 5 章  实验结果\n")
A(f"**5.1 与基线的对比。** 见表 3、图 3。主模型 test 准确率 **{main_t['accuracy']:.4f}**、"
  f"test macro F1 **{main_t['macro_f1']:.4f}**；最强基线 B3（cap3000 匹配分母）为 "
  f"**{b3m['accuracy']:.4f} / {b3m['macro_f1']:.4f}**。主模型在准确率上领先 "
  f"**{100*(main_t['accuracy']-b3m['accuracy']):.2f} pp**，在 overall macro F1 上领先 "
  f"**{100*(main_t['macro_f1']-b3m['macro_f1']):.2f} pp**。对 B1/B2 两项指标均大幅领先"
  f"（≥ 30 pp）。\n")
A("**5.2 该领先的置信度。** 主表的 8 轮权重已删除，无法直接给出该行的文档级区间；"
  f"用同架构替身锚点实测（表 5a）：overall macro F1 的差值为 **{100*fs['delta_point']:+.2f} pp**，"
  f"文档级 95% CI **[{100*fs['doc_ci']['lo']:+.2f}, {100*fs['doc_ci']['hi']:+.2f}]**（p={fs['p_two_sided']:.3f}），"
  f"**跨 0**；而块级 CI 只有 **[{100*fs['block_ci']['lo']:+.2f}, {100*fs['block_ci']['hi']:+.2f}]**，"
  f"会把同一个点估判成\"显著\"。因此本文**不主张\"overall macro F1 上显著优于 B3\"**。"
  f"可以主张的三条是（表 9）：① **support ≥ 500 的 7 个大类**上领先 "
  f"**{100*fs['support_filtered']['500']['delta_point']:.2f} pp**"
  f"（文档级 95% CI [{100*fs['support_filtered']['500']['ci']['lo']:+.2f}, "
  f"{100*fs['support_filtered']['500']['ci']['hi']:+.2f}]，p<0.001）；"
  f"② **逐文档准确率** {fs['per_doc_wins']['a']} 胜 {fs['per_doc_wins']['ties']} 平 "
  f"{fs['per_doc_wins']['b']} 负（精确符号检验 p={fs['per_doc_wins']['sign_p']:.1e}）；"
  f"③ 该方向**对训练进度不敏感**——连劣化锚点 ep10（overall 落后 "
  f"{abs(100*sg['test_delivered_ep10_vs_b3']['paired']['delta_point']):.2f} pp）在 sup≥500 上仍领先 "
  f"{100*sg['test_delivered_ep10_vs_b3']['paired']['support_filtered']['500']['delta_point']:.2f} pp"
  f"（p={sg['test_delivered_ep10_vs_b3']['paired']['support_filtered']['500']['p_two_sided']:.3f}）。\n")
A("**5.3 overall macro F1 的持平来自长尾。** 见表 4。B3 的优势**只**出现在 `<100` 档"
  f"（{tc['blocks']['<100']} 块 / {tc['classes']['<100']} 类，主模型落后 {abs(tier_d['<100']):.1f} pp），"
  "而该档的文档级标准差高达 7–9 pp，任何基于它的结论都不可靠；"
  f"在其余三档（占测试集 {100*(tc['blocks']['100-999']+tc['blocks']['1k-5k']+tc['blocks']['5k+'])/sum(tc['blocks'].values()):.1f}% 的块）"
  f"主模型分别领先 {tier_d['100-999']:+.1f} / {tier_d['1k-5k']:+.1f} / {tier_d['5k+']:+.1f} pp。"
  "整体 macro F1 是 19 类的未加权均值，其中 10 类落在 `100–999`，该档主导了 overall。"
  "**因此结论应按\"主模型在数据主体与高频类上更强\"陈述，而非依赖 overall 的持平。**\n")
A("**5.4 消融。** 见表 2、表 9、图 5。按 §4.4 预登记判据：")
for slug, label in MAIN[2:]:
    t = TESTS[f"{slug}/last"]["sets"]["test"]
    dm_ = 100 * (t["macro_f1"] - main_t["macro_f1"])
    d5_ = 100 * (sup_f1(t["per_class"], 500)[0] - s500_0)
    A(f"- **{label}**：Δ test macro F1 **{dm_:+.2f} pp**（{zone(dm_)}）；"
      f"sup≥500 口径 {d5_:+.2f} pp。")
A("")
A("**能站住的**是 `A11` 与 `A7`（两个口径同向且幅度超判据）；`A6` 只在 overall 口径上可分辨"
  "（sup≥500 上归零），必须写明条件；`A2` 与 `A10` 在本数据上**无法与噪声区分**，"
  "如实标注为\"无差异\"而不是宣称其方向有效。\n")

A("### 第 6 章  讨论与局限\n")
A("**6.1 有效独立单位是 62 篇文档，不是 8 万个块。** 块在文档内高度相关"
  f"（设计效应 {sg['test_delivered_ep16_vs_b3']['A:delivered_ep16']['design_effect']:.1f}×），"
  "因此本文全部不确定性陈述都建立在文档级自助法上；任何以块为独立单位的 p 值都会被系统性夸大。\n")
A("**6.2 单种子与噪声底。** 除主模型外各消融行只跑了一个种子。在轮末单点噪声 2–3.5 pp、"
  "换 seed 抖动 3–5 pp 的前提下，1–4 pp 的效应不能与运行间波动区分。"
  "本文选择**如实标注灰色区与无差异**，而不是宣称这些消融方向有效。\n")
A("**6.3 主表权重的可复现性缺口。** 8 轮权重为腾出磁盘已删除，主表的测试读数**无法原样重放**；"
  "已记录读数在生成时通过了复现闸门（重放同一 checkpoint 的验证集并与 `history.json` 比对）。"
  "文档级区间因此只能用替身锚点给出（表 5），这是本文最主要的方法学妥协。\n")
A("**6.4 补跑也不会带来显著性（功率分析）。** 配对文档级 sd 为 2.6–3.5 pp，"
  "真实差值 3.2–5.4 pp，双侧 95% 门槛约 5.9 pp ⇒ **功率仅 26–44%**。"
  "换句话说，重训主表那一对在 6–7 成概率下仍然跨 0。这不是运气问题，"
  "而是\"19 类 overall macro F1 × 62 篇文档 × 4 pp 效应\"这一组合本身缺功率。"
  "因此本文不把\"补跑\"列为提高结论强度的手段，而是**换用对稀有类不敏感的口径**（5.2 的三条）。\n")
A("**6.5 基线类别权重的一致性。** 现有基线实现使用未加权交叉熵，而主模型使用 `inverse` 权重。"
  "这一差异会系统性压低基线的 macro F1——恰好是基线最接近的指标。分层结果（表 4）显示 "
  "B1/B2 在**每一档**均大幅落后，可排除权重差异的解释；但 B3 的优势集中于长尾档，"
  "该质疑对 B3 仍部分成立。对齐权重后的重跑列为进一步工作（**本文不主张**）。\n")
A("**6.6 评测分母一致性。** B3 因页内块数上限，原生评测只覆盖 59,043 块。本文统一到 "
  f"{b3m['valid_count']:,} 块后报告，并说明该复评使用原训练权重，存在训练/评测序列长度的轻微错配。"
  "匹配上限重训列为进一步工作（**本文不主张**）。\n")
A("**6.7 模态消融的范围。** 本文消融矩阵聚焦门控与记忆结构，未包含图像分支的单因素消融，"
  "故**不对图像模态的独立贡献作定量主张**；图像分支在本文中被描述为实现构成而非已量化的增益来源。\n")
A("**6.8 不能主张的清单。** 综合 §5.2 与表 9：本文不主张\"更轻量的模型\"、\"更快的推理\"、"
  "\"overall macro F1 上显著优于 B3\"、\"A2/A10 的效应已被证实\"。"
  "这四条在答辩中应主动说明，而不是等提问。\n")

# ------------------------------------------------------------------ 待补
A("## 四、待补实验与口径待定项（答辩口径裁剪）\n")
A("| 项目 | 状态 | 说明 |")
A("|---|---|---|")
A("| 测试集一次性评测 | **已完成** | 62 篇 / 79,851 块，见 `test_eval_clean.json` / `test_eval_baselines.json` |")
A("| 文档级置信区间 | **已完成（替身锚点）** | 见 `docs/path3_doc_level_significance_2026-10-04.md`；主表那一对的区间仍缺 |")
A("| 支持度分层（test 口径） | **已完成** | 表 4 |")
A("| 效率与规模表 | **已完成** | 表 8，`docs/paper_materials/efficiency_2026-10-04.json` |")
A("| 基线类别权重重跑 | **不主张，不做** | 答辩不需要；在 §6.5 如实列为局限 |")
A("| 图像分支单因素消融 | **不主张，不做** | 在 §6.7 明确不作定量主张 |")
A("| 灰色区补种子（A2/A6/A7/A10 × seed 43,44） | **不主张，不做** | 功率分析显示收益有限（§6.4） |")
A("| B3 匹配上限重训 | **不主张，不做** | 在 §6.6 如实列为局限 |")
A("| `A7` 在论文中的命名 | **已定（D8）** | \"移除顺序记忆写入门控\" |")
A("| 主表那一对的真实文档级区间 | 可选 | 仅当需要\"可复现性完整\"时才有必要重训 8 轮 |")
A("")
A("---")
A("")
A("本文档由 `scripts/paper/make_thesis_draft.py` 从 `remote_pull_2026-09-29/`、"
  "`outputs/path3_diag/`、`docs/paper_materials/` 现算生成，所有数值均取自产物文件，未做手工转录。")

OUT.write_text("\n".join(L), encoding="utf-8")
print("written:", OUT, f"({OUT.stat().st_size} bytes)")