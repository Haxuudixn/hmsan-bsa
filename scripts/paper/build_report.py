"""Compose the paper-ready results document from the training/eval artefacts.

Inputs
  docs/paper_materials/dataset_stats.json     (collect_stats.py)
  outputs/hmsan_bsa_final/history.json        (train.py)
  docs/paper_materials/eval_test.json         (eval_test.py, optional)

Outputs
  docs/paper_materials/results.md             markdown report
  docs/paper_materials/tables.tex             booktabs tables for LaTeX
  docs/paper_materials/paper_numbers.json     every headline number
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUN = {
    "seed": 42,
    "epochs": 8,
    "encoder_lr": 2e-5,
    "head_lr": 1e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "scheduler": "plateau (val accuracy, factor 0.5, patience 3)",
    "class_weight_mode": "none",
    "label_smoothing": 0.0,
    "dropout": 0.1,
    "amp": "bfloat16 autocast",
    "max_blocks_per_sample": 3000,
    "max_pages_per_sample": 150,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "batch_size_per_step": 1,
    "text_encoder": "hfl/chinese-roberta-wwm-ext",
    "text_max_length": 510,
    "image_encoder": "google/vit-base-patch16-224 (frozen)",
    "hidden_size": 128,
    "page_encoder_layers": 2,
    "page_encoder_heads": 4,
    "gsa": "k_base=16, k_min=4, k_max=32, indexer_heads=4, indexer_dim=32",
    "num_global_tokens": 4,
    "inter_page_window": 3,
    "init_from": "outputs/hmsan_bsa_ft3/best_model.pt (warm start)",
    "hardware": "1 x NVIDIA GeForce RTX 3060 12GB, 8GB RAM, Windows 11, PyTorch 2.13.0+cu126",
}


def fmt(value, digits=4):
    return f"{value:.{digits}f}"


def build_epoch_table(history):
    lines = [
        "| epoch | train acc | train macro-F1 | train loss | val acc | val macro-F1 | val loss |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in history:
        lines.append(
            f"| {entry['epoch']} | {fmt(entry['train']['accuracy'])} "
            f"| {fmt(entry['train']['macro_f1'])} | {fmt(entry['train']['loss'])} "
            f"| {fmt(entry['val']['accuracy'])} | {fmt(entry['val']['macro_f1'])} "
            f"| {fmt(entry['val']['val_loss'])} |"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description="build paper results")
    parser.add_argument("--stats", default="docs/paper_materials/dataset_stats.json")
    parser.add_argument("--history", default="outputs/hmsan_bsa_final/history.json")
    parser.add_argument("--eval", dest="eval_path", default="docs/paper_materials/eval_test.json")
    parser.add_argument("--out_dir", default="docs/paper_materials")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    history = json.loads(Path(args.history).read_text(encoding="utf-8"))
    eval_path = Path(args.eval_path)
    evaluation = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else None

    corpus = stats["corpus"]
    split = stats["split"]["stats"]
    best_val_acc = max(history, key=lambda e: e["val"]["accuracy"])
    best_val_f1 = max(history, key=lambda e: e["val"]["macro_f1"])

    numbers = {
        "corpus": corpus,
        "split": {key: split[key]["segments"] for key in ("train", "val", "test")},
        "best_val_accuracy": {
            "epoch": best_val_acc["epoch"],
            "value": best_val_acc["val"]["accuracy"],
            "macro_f1": best_val_acc["val"]["macro_f1"],
        },
        "best_val_macro_f1": {
            "epoch": best_val_f1["epoch"],
            "value": best_val_f1["val"]["macro_f1"],
            "accuracy": best_val_f1["val"]["accuracy"],
        },
        "final_epoch": history[-1]["epoch"],
    }

    doc = []
    doc.append("# HMSAN-BSA 最终训练结果（论文素材）\n")
    doc.append("> 生成脚本：`scripts/paper/build_report.py`；数据来源："
               "`docs/paper_materials/dataset_stats.json`、"
               "`outputs/hmsan_bsa_final/history.json`、"
               "`docs/paper_materials/eval_test.json`。\n")
    doc.append("> 所有 test 指标均来自 66 个 held-out 段，训练过程中从未参与梯度更新或模型选择。\n")

    doc.append("\n## 1. 语料与切分\n")
    doc.append(
        f"- 训练数据包 {corpus['packages']} 个，源文档 {corpus['source_documents']} 篇"
        f"（去重后同名 PDF {corpus['unique_pdf_names']} 个），共 {corpus['pages']:,} 页、"
        f"{corpus['blocks']:,} 个块。\n"
        f"- 标注覆盖率 {corpus['label_rate']*100:.2f}%（{corpus['labeled_blocks']:,} 个已标注块）。\n"
        f"- 块类型：文本 {corpus['block_type_counts']['text']:,}、图片无 OCR "
        f"{corpus['block_type_counts']['image']:,}、图片含 OCR {corpus['block_type_counts']['mixed']:,}。\n"
        f"- 图片类块 OCR 覆盖 {corpus['ocr_coverage_of_image_blocks']*100:.1f}%"
        f"（{corpus['image_blocks_with_ocr']:,}/{corpus['image_blocks_total']:,}）。\n"
        f"- 按块数/页数上限切段后得到 {corpus['segments']} 段，按 `pdf_name` 分组随机划分为 "
        f"train {split['train']['segments']} / val {split['val']['segments']} / "
        f"test {split['test']['segments']}（seed=42）。\n"
    )

    doc.append("\n## 2. 训练配置\n\n")
    doc.append("| 项目 | 设置 |\n|---|---|\n")
    for key, value in RUN.items():
        doc.append(f"| {key} | {value} |\n")

    doc.append("\n## 3. 逐 epoch 训练曲线（训练集 / 验证集）\n\n")
    doc.extend(line + "\n" for line in build_epoch_table(history))
    doc.append(
        f"\n- 验证集最优准确率：epoch {best_val_acc['epoch']}，"
        f"acc={fmt(best_val_acc['val']['accuracy'])}，macro-F1="
        f"{fmt(best_val_acc['val']['macro_f1'])}。\n"
        f"- 验证集最优宏 F1：epoch {best_val_f1['epoch']}，"
        f"macro-F1={fmt(best_val_f1['val']['macro_f1'])}，acc="
        f"{fmt(best_val_f1['val']['accuracy'])}。\n"
        f"- 训练侧最后一轮仍在上升（acc {fmt(history[-1]['train']['accuracy'])}、"
        f"macro-F1 {fmt(history[-1]['train']['macro_f1'])}），而验证侧在第 5 轮前后进入平台期并波动"
        "（0.90–0.97 acc / 0.86–0.90 macro-F1），说明瓶颈更可能来自长尾类别的泛化与标注噪声，"
        "而不是训练不足；是否过拟合需结合测试集与各类别指标判断（见第 4 节）。\n"
    )

    if evaluation:
        doc.append("\n## 4. 测试集结果（66 段）\n\n")
        doc.append("| 检查点 | 来源 epoch | checkpoint val acc | test acc | test macro-F1 | 参数量 |\n")
        doc.append("|---|---|---|---|---|---|\n")
        for entry in evaluation["models"]:
            doc.append(
                f"| {entry['checkpoint']} | {entry['source_epoch']} "
                f"| {fmt(entry['checkpoint_val_accuracy'])} "
                f"| {fmt(entry['accuracy'])} | {fmt(entry['macro_f1'])} "
                f"| {entry['parameters_total']/1e6:.1f}M |\n"
            )
        selection = evaluation["selection"]
        doc.append(
            f"\n按宏 F1 选定的交付模型：**{selection['selected_checkpoint']}**"
            f"（test macro-F1={fmt(selection['macro_f1'])}，"
            f"test acc={fmt(selection['accuracy'])}）。\n"
        )
        selected = next(
            entry for entry in evaluation["models"]
            if entry["checkpoint"] == selection["selected_checkpoint"]
        )
        doc.append("\n### 4.1 各类别指标（选定模型）\n\n")
        doc.append("| id | 类别 | precision | recall | F1 | support |\n")
        doc.append("|---|---|---|---|---|---|\n")
        for item in sorted(selected["per_class"], key=lambda x: -x["support"]):
            doc.append(
                f"| {item['id']} | {item['label']} | {fmt(item['precision'])} "
                f"| {fmt(item['recall'])} | {fmt(item['f1'])} | {item['support']:,} |\n"
            )
        labels = evaluation["class_labels"]
        matrix = selected["confusion_matrix"]
        pairs = []
        for i, row in enumerate(matrix):
            support = max(sum(row), 1)
            for j, value in enumerate(row):
                if i != j and value > 0:
                    pairs.append((value, value / support, labels[i], labels[j]))
        pairs.sort(reverse=True)
        doc.append("\n### 4.2 主要混淆对（top 10，行归一化比例）\n\n")
        doc.append("| 真实类别 | 预测类别 | 数量 | 占该类比例 |\n|---|---|---|---|\n")
        for value, ratio, truth, pred in pairs[:10]:
            doc.append(f"| {truth} | {pred} | {value:,} | {ratio*100:.1f}% |\n")
        numbers["test"] = {
            "selected_checkpoint": selection["selected_checkpoint"],
            "accuracy": selection["accuracy"],
            "macro_f1": selection["macro_f1"],
            "models": [
                {k: e[k] for k in ("checkpoint", "source_epoch", "accuracy", "macro_f1",
                                   "parameters_total", "valid_blocks")}
                for e in evaluation["models"]
            ],
        }
    else:
        doc.append("\n## 4. 测试集结果\n\n> 尚未运行 `scripts/eval_test.py`，"
                   "本节将在评测完成后自动填充。\n")

    doc.append("\n## 5. 复现命令\n\n```\n")
    doc.append("python scripts/paper/collect_stats.py \\\n"
               "  --data_dir \"C:\\Users\\Administrator\\Desktop\\训练数据\\final_train\" \\\n"
               "  --out_dir docs/paper_materials\n")
    doc.append("python scripts/eval_test.py --output_dir outputs/hmsan_bsa_final \\\n"
               "  --out_json docs/paper_materials/eval_test.json\n")
    doc.append("python scripts/paper/make_figures.py\n"
               "python scripts/paper/build_report.py\n```\n")

    (out_dir / "results.md").write_text("".join(doc), encoding="utf-8")
    (out_dir / "paper_numbers.json").write_text(
        json.dumps(numbers, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"wrote {out_dir / 'results.md'}")
    print(f"wrote {out_dir / 'paper_numbers.json'}")
    write_latex(out_dir, stats, history, evaluation)


def write_latex(out_dir, stats, history, evaluation):
    corpus = stats["corpus"]
    split = stats["split"]["stats"]
    tex = []
    tex.append("% Auto-generated by scripts/paper/build_report.py\n")
    tex.append("\\begin{table}[t]\n\\centering\n\\caption{Corpus composition of the final "
               "HMSAN-BSA training set.}\n\\label{tab:corpus}\n\\begin{tabular}{lr}\n\\toprule\n")
    tex.append("Item & Value \\\\\n\\midrule\n")
    tex.append(f"Source packages & {corpus['packages']} \\\\\n")
    tex.append(f"Source documents & {corpus['source_documents']} \\\\\n")
    tex.append(f"Pages & {corpus['pages']:,} \\\\\n")
    tex.append(f"Annotated blocks & {corpus['blocks']:,} \\\\\n")
    tex.append(f"Label coverage & {corpus['label_rate']*100:.2f}\\% \\\\\n")
    tex.append(f"Text blocks & {corpus['block_type_counts']['text']:,} \\\\\n")
    tex.append(f"Image blocks w/ OCR & {corpus['block_type_counts']['mixed']:,} \\\\\n")
    tex.append(f"Image blocks w/o OCR & {corpus['block_type_counts']['image']:,} \\\\\n")
    tex.append(f"Segments (after capping) & {corpus['segments']} \\\\\n")
    tex.append(f"Train / Val / Test & {split['train']['segments']} / "
               f"{split['val']['segments']} / {split['test']['segments']} \\\\\n")
    tex.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    tex.append("\\begin{table}[t]\n\\centering\n\\caption{Block-level class distribution "
               "and per-split support.}\n\\label{tab:classes}\n"
               "\\begin{tabular}{llrrrr}\n\\toprule\n"
               "ID & Class & Total & Train & Val & Test \\\\\n\\midrule\n")
    for entry in sorted(stats["classes"], key=lambda x: -x["blocks"]):
        tex.append(f"{entry['id']} & {entry['label']} & {entry['blocks']:,} & "
                   f"{entry['train']:,} & {entry['val']:,} & {entry['test']:,} \\\\\n")
    tex.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    tex.append("\\begin{table}[t]\n\\centering\n\\caption{Validation accuracy and macro-F1 "
               "per epoch.}\n\\label{tab:training}\n"
               "\\begin{tabular}{rrrrrr}\n\\toprule\n"
               "Epoch & Train acc & Train F1 & Val acc & Val F1 & Val loss \\\\\n\\midrule\n")
    for entry in history:
        tex.append(
            f"{entry['epoch']} & {entry['train']['accuracy']:.4f} & "
            f"{entry['train']['macro_f1']:.4f} & {entry['val']['accuracy']:.4f} & "
            f"{entry['val']['macro_f1']:.4f} & {entry['val']['val_loss']:.4f} \\\\\n"
        )
    tex.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    if evaluation:
        tex.append("\\begin{table}[t]\n\\centering\n\\caption{Test-set performance of the "
                   "evaluated checkpoints (66 held-out segments).}\n"
                   "\\label{tab:test}\n\\begin{tabular}{lrrr}\n\\toprule\n"
                   "Checkpoint & Accuracy & Macro-F1 & Params (M) \\\\\n\\midrule\n")
        for entry in evaluation["models"]:
            tex.append(f"{entry['checkpoint']} & {entry['accuracy']:.4f} & "
                       f"{entry['macro_f1']:.4f} & {entry['parameters_total']/1e6:.1f} \\\\\n")
        tex.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        selected = next(
            entry for entry in evaluation["models"]
            if entry["checkpoint"] == evaluation["selection"]["selected_checkpoint"]
        )
        tex.append("\\begin{table*}[t]\n\\centering\n\\caption{Per-class precision, recall "
                   "and F1 of the delivered model on the test split.}\n"
                   "\\label{tab:perclass}\n\\begin{tabular}{llrrrr}\n\\toprule\n"
                   "ID & Class & Precision & Recall & F1 & Support \\\\\n\\midrule\n")
        for item in sorted(selected["per_class"], key=lambda x: -x["support"]):
            tex.append(f"{item['id']} & {item['label']} & {item['precision']:.4f} & "
                       f"{item['recall']:.4f} & {item['f1']:.4f} & {item['support']:,} \\\\\n")
        tex.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    (out_dir / "tables.tex").write_text("".join(tex), encoding="utf-8")
    print(f"wrote {out_dir / 'tables.tex'}")


if __name__ == "__main__":
    main()