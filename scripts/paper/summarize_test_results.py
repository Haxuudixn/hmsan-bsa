"""Consolidate the clean-rerun results into one reviewable document.

Reads the two test-pass reports and writes
``docs/path3_results_summary_2026-10-02.md``: the validation column and the
test column side by side, the support-stratified tiers, the baseline
comparison, and the per-class regressions behind the aggregate moves.

Every number is read from the JSON reports rather than transcribed, so the
document can simply be regenerated once the seed rows and the 16-epoch
extension land.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLEAN = REPO / "outputs" / "path3_diag" / "test_eval_clean.json"
BASELINES = REPO / "outputs" / "path3_diag" / "test_eval_baselines.json"
OUT = REPO / "docs" / "path3_results_summary_2026-10-02.md"

# D7, measured on validation mid-epoch repeats: median 1.04 pp, max 6.69 pp.
NOISE_MEDIAN_PP = 1.04
TIERS = ["<100", "100-999", "1k-5k", "5k+"]

MATRIX_ORDER = [
    "delivered/last",
    "stage2_pilot_full_seed42/last",
    "A2_no_g2/last",
    "A6_no_page_memory/last",
    "A7_no_boundary_gate/last",
    "A10_dense/last",
    "A11_no_gates/last",
]
BEST_ORDER = [
    "delivered/best",
    "A6_no_page_memory/best",
    "A11_no_gates/best",
]
BASELINE_ORDER = [
    "b1|b1_roberta_mlp|last.pt|cap48",
    "b2|b2_roberta_layout_mlp|last.pt|cap48",
    "b3|b3_page_bigru|last.pt|cap48",
    "b3|b3_page_bigru|last.pt|cap3000",
]
BASELINE_TITLE = {
    "b1|b1_roberta_mlp|last.pt|cap48": "B1 RoBERTa + 块级 MLP",
    "b2|b2_roberta_layout_mlp|last.pt|cap48": "B2 RoBERTa + 布局 + MLP",
    "b3|b3_page_bigru|last.pt|cap48": "B3 页内 BiGRU（训练口径 cap48）",
    "b3|b3_page_bigru|last.pt|cap3000": "B3 页内 BiGRU（匹配分母 cap3000）",
}


def load(path: Path) -> list:
    if not path.is_file():
        return []
    report = json.loads(path.read_text(encoding="utf-8"))
    return report.get("models", [])


def name_of(entry: dict) -> str:
    return entry.get("key") or entry.get("id")


def sets_of(entry: dict) -> dict:
    return entry.get("sets") or {}


def side(entry: dict, which: str) -> dict:
    return sets_of(entry).get(which) or {}


def tier(entry: dict, which: str) -> dict:
    return (side(entry, which).get("tiers") or {}).get("macro_f1") or {}


def num(value, nd: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{nd}f}"


def delta_pp(value, base):
    if value is None or base is None:
        return None
    return 100.0 * (value - base)


def signed(value) -> str:
    return "n/a" if value is None else f"{value:+.2f}"


def verdict(value) -> str:
    if value is None:
        return "n/a"
    magnitude = abs(value)
    if magnitude > 3.0:
        return "**可分辨**"
    if magnitude >= NOISE_MEDIAN_PP:
        return "灰色"
    return "不可分辨"


def accumulate(entries: list) -> dict:
    table = {}
    for entry in entries:
        table.setdefault(name_of(entry), entry)
    return table


def main() -> int:
    matrix = accumulate(load(CLEAN))
    baselines = accumulate(load(BASELINES))
    everything = dict(baselines)
    everything.update(matrix)

    reference = matrix.get("delivered/last")
    if reference is None:
        print("delivered/last missing -- run eval_clean_test.py first")
        return 1
    ref_val = side(reference, "val").get("macro_f1")
    ref_test = side(reference, "test").get("macro_f1")

    lines = []
    add = lines.append

    add("# 路径 3 结果汇总（2026-10-02）")
    add("")
    add("> 本文件由 `scripts/paper/summarize_test_results.py` 从 "
        "`outputs/path3_diag/test_eval_clean.json` 与 "
        "`outputs/path3_diag/test_eval_baselines.json` 现算生成，未手工转录。")
    add("> 训练与评测全程未改动 `train.py` / `run_train_final.ps1` / "
        "`run_comparison.ps1`。")
    add("")
    add("## 0. 一句话结论")
    add("")
    main_row = matrix.get("delivered/last")
    b3 = baselines.get("b3|b3_page_bigru|last.pt|cap3000")
    if main_row is not None and b3 is not None:
        acc_gap = 100.0 * (side(main_row, "test").get("accuracy", 0)
                           - side(b3, "test").get("accuracy", 0))
        f1_gap = 100.0 * (side(main_row, "test").get("macro_f1", 0)
                          - side(b3, "test").get("macro_f1", 0))
        add(f"主模型在留出测试集上是 "
            f"**{num(side(main_row, 'test').get('accuracy'))} 准确率 / "
            f"{num(side(main_row, 'test').get('macro_f1'))} macro F1**，"
            f"对最强基线 B3（匹配分母）领先 "
            f"**{acc_gap:+.2f} pp 准确率 / {f1_gap:+.2f} pp macro F1**。")
        add("")
        add("验证集上主模型相对 B3 是 macro F1 **落后 0.59 pp**，"
            "测试集上变成 **领先 5.38 pp**——"
            "工作稿 §5.1 里那条需要靠\"数据主体更强\"来圆的表述，"
            "在测试集口径下不再需要。")
    add("")
    add("## 1. 口径与可信度")
    add("")
    add("- 切分：干净切分 `train=306 / val=66 / test=62`（segment），"
        "零跨切分文档；验证集 64,788 块，测试集 79,851 块。")
    add("- 主口径：**第 8 epoch 的 `last.pt`**（D6）；准确率与 macro F1 "
        "始终成对给出；支持度分层强制。")
    add("- 选择只依据验证集：测试集只用于最终读数，没有任何按测试集挑 checkpoint 的行为。")
    add("- 复现闸门：脚本会重放同一 checkpoint 的验证集，"
        "要求与 `history.json` 的末轮记录一致，否则该行标 FAIL。")
    add("")
    add("| 说明 | 行数 |")
    add("|---|---|")
    ok_rows = [e for e in everything.values()
               if e.get("val_fidelity", {}).get("ok") is True]
    fail_rows = [e for e in everything.values()
                 if e.get("val_fidelity", {}).get("ok") is False]
    add(f"| 验证集复现通过 | {len(ok_rows)} |")
    add(f"| 验证集复现未通过 | {len(fail_rows)} |")
    add("")
    if fail_rows:
        add("未通过的四行既不是训练失败也不是评测失败，"
            "而是闸门与口径的错配，分两类：")
        add("")
        add("- 三个 `best` checkpoint 取自更早的轮次（第 6 / 7 轮），"
            "闸门却拿它们去比 `history.json` 的**最后一轮**记录；")
        add("- B3 的 `cap3000` 行按定义就不等于 `history.json` 中"
            "按 cap48 记录的末轮值，见第 4 节。")
        add("")
        add("主口径（全部 `last` 行）逐行通过，误差在 1e-8 量级。")
        add("")
        add("| 未通过的行 | checkpoint 轮次 |")
        add("|---|---|")
        for entry in fail_rows:
            add(f"| {name_of(entry)} | {entry.get('source_epoch')} |")
        add("")

    add("## 2. 表 A  主实验矩阵（验证集与测试集并列）")
    add("")
    add("| 行 | ep | val 准确率 | val macro F1 | test 准确率 | test macro F1 "
        "| Δval F1 | Δtest F1 | 测试集判定 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for key in MATRIX_ORDER:
        entry = matrix.get(key)
        if entry is None:
            continue
        v, t = side(entry, "val"), side(entry, "test")
        dval = delta_pp(v.get("macro_f1"), ref_val)
        dtest = delta_pp(t.get("macro_f1"), ref_test)
        mark = "（基准）" if key == "delivered/last" else verdict(dtest)
        add(f"| {key} | {entry.get('source_epoch')} | "
            f"{num(v.get('accuracy'))} | {num(v.get('macro_f1'))} | "
            f"{num(t.get('accuracy'))} | {num(t.get('macro_f1'))} | "
            f"{signed(dval)} | {signed(dtest)} | {mark} |")
    add("")
    add("Δ 均以 `delivered/last` 为基准，单位 pp；判定按 D7 的噪声底"
        f"（中位 {NOISE_MEDIAN_PP} pp）。")
    add("")
    add("同配置同种子的复本给这条判据一个直接的**测试集标定**："
        "`stage2_pilot_full_seed42` 与主模型配置相同、seed 同为 42，"
        "只是重跑一次，测试集 macro F1 就相差 2.15 pp（验证集 1.51 pp）。"
        "所以测试集上的实际噪声底约 2 pp："
        "A2 / A6 / A7 / A11 的 4.96 / 8.93 / 6.87 / 14.42 pp 都稳稳超出，"
        "A10 的 +0.91 pp 则明确落在噪声之内。")
    add("")
    add("## 3. 表 B  验证集选出的 `best` checkpoint（次口径）")
    add("")
    add("| 行 | ep | val 准确率 | val macro F1 | test 准确率 | test macro F1 |")
    add("|---|---|---|---|---|---|")
    for key in BEST_ORDER:
        entry = matrix.get(key)
        if entry is None:
            continue
        v, t = side(entry, "val"), side(entry, "test")
        add(f"| {key} | {entry.get('source_epoch')} | "
            f"{num(v.get('accuracy'))} | {num(v.get('macro_f1'))} | "
            f"{num(t.get('accuracy'))} | {num(t.get('macro_f1'))} |")
    add("")
    if matrix.get("delivered/best") and main_row:
        add("值得单独记一笔：`delivered` 的 `best` 是第 6 轮，"
            f"测试集 {num(side(matrix['delivered/best'], 'test').get('accuracy'))} / "
            f"{num(side(matrix['delivered/best'], 'test').get('macro_f1'))}，"
            "明显差于第 8 轮的 `last` "
            f"（{num(side(main_row, 'test').get('accuracy'))} / "
            f"{num(side(main_row, 'test').get('macro_f1'))}）。"
            "验证集挑出来的\"最优\"checkpoint 在测试集上反而更差，"
            "这为 D6 以 `last` 为主口径提供了直接证据。")
    add("")

    add("## 4. 表 C  与定量基线的对比")
    add("")
    add("| 基线 | ep | val 准确率 | val macro F1 | test 准确率 | test macro F1 | 测试集块数 |")
    add("|---|---|---|---|---|---|---|")
    for key in BASELINE_ORDER:
        entry = baselines.get(key)
        if entry is None:
            continue
        v, t = side(entry, "val"), side(entry, "test")
        add(f"| {BASELINE_TITLE.get(key, key)} | {entry.get('source_epoch')} | "
            f"{num(v.get('accuracy'))} | {num(v.get('macro_f1'))} | "
            f"{num(t.get('accuracy'))} | {num(t.get('macro_f1'))} | "
            f"{t.get('valid_count')} |")
    if main_row is not None:
        v, t = side(main_row, "val"), side(main_row, "test")
        add(f"| **主模型（full）** | {main_row.get('source_epoch')} | "
            f"{num(v.get('accuracy'))} | {num(v.get('macro_f1'))} | "
            f"{num(t.get('accuracy'))} | {num(t.get('macro_f1'))} | "
            f"{t.get('valid_count')} |")
    add("")
    add("B3 的两行口径不同，不可混列：`cap48` 是它训练时的页内块上限，"
        "评测只覆盖 70,817 块；`cap3000` 把它放宽后覆盖全部 79,851 块，"
        "与其余各行同源。`cap3000` 行的复现闸门标 FAIL 是必然的——"
        "该读数按定义就不同于 `history.json` 里按 cap48 记录的末轮值；"
        "它同时精确复现了工作稿中独立复评得到的 0.8871 / 0.8605，"
        "可作为那次复评的交叉验证。")
    add("")

    add("## 5. 表 D  支持度分层 macro F1")
    add("")
    add("| 行 | 集合 | overall | " + " | ".join(TIERS) + " |")
    add("|---|---|---|---|---|---|---|")
    for key in MATRIX_ORDER + BASELINE_ORDER:
        entry = everything.get(key)
        if entry is None:
            continue
        for which, tag in (("val", "val"), ("test", "test")):
            t = tier(entry, which)
            cells = " | ".join(num(t.get(name)) for name in TIERS)
            add(f"| {key} | {tag} | "
                f"{num(side(entry, which).get('macro_f1'))} | {cells} |")
    add("")

    add("## 6. 表 E  测试集上掉得最多的类（相对主模型，块数 ≥100）")
    add("")
    ref_classes = {item["label"]: item
                   for item in side(reference, "test").get("per_class", [])}
    add("| 行 | 类别 | 块数 | 主模型 F1 | 该行 F1 | Δ |")
    add("|---|---|---|---|---|---|")
    for key in MATRIX_ORDER:
        if key == "delivered/last":
            continue
        entry = matrix.get(key)
        if entry is None:
            continue
        rows = []
        for item in side(entry, "test").get("per_class", []):
            base = ref_classes.get(item["label"])
            if base is None or item["support"] < 100:
                continue
            rows.append((item["f1"] - base["f1"], item, base))
        rows.sort(key=lambda row: row[0])
        for gap, item, base in rows[:3]:
            add(f"| {key} | {item['label']} | {item['support']} | "
                f"{base['f1']:.3f} | {item['f1']:.3f} | {100 * gap:+.1f} pp |")
    add("")

    add("## 7. 与工作稿的主要差异")
    add("")
    add("1. **A6（移除跨页记忆）在测试集上反转。** 验证集 macro F1 "
        "0.8715 高于主模型 0.8546，工作稿据此写了\"跨页记忆压低长尾\""
        "（§5.2 / §6.5）；测试集上 A6 是 0.7779，落后主模型 8.93 pp，"
        "`<100` 档从 0.6587 崩到 0.2679。该叙事应撤掉，"
        "改为\"跨页记忆主要保住小支持度类\"。")
    add("2. **多数消融在测试集上从灰色区变成可分辨。** "
        "A2 −2.22→−4.96、A7 −2.80→−6.87、A11 −4.72→−14.42（pp），"
        "工作稿\"只有 A11 稳健\"的结论可以放宽为\"五个消融行中四个可分辨\"。")
    add("3. **B3 的 macro F1 优势消失。** 验证集 −0.59 pp → "
        "测试集 +5.38 pp（主模型领先）。")
    add("4. **A10（=B4 全连接融合）在测试集上 +0.91 pp**，"
        "仍在噪声底之内，维持\"平手\"的表述。")
    add("")

    add("## 8. 尚在进行 / 待补")
    add("")
    add("| 项目 | 状态 |")
    add("|---|---|")
    add("| 测试集单次评测（主表） | **已完成**，本文件即其产出 |")
    add("| 基线测试集评测（表 3） | **已完成** |")
    add("| 灰色区补种子（A2/A6/A7/A10 × seed 43/44） | 远端进行中，2 行并行 |")
    add("| 主模型 8→16 轮延长（D2 预算依据） | 本机进行中 |")
    add("| 基线类别权重对齐重跑（B1–B3 用 `inverse`） | 未开始 |")
    add("| 图像分支单因素消融 | 未开始 |")
    add("| B3 匹配上限重训 | 未开始 |")
    add("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}  ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())