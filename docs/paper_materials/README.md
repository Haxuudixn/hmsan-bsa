# 论文素材目录（HMSAN-BSA 最终训练）

本目录把最终训练所涉及的全部数据、方法、结果整合为**可直接用于论文**的素材。
所有数字均由脚本从原始产物生成，不使用手工抄录。

## 文件索引

| 文件 | 内容 | 状态 |
|---|---|---|
| `dataset_card.md` | 数据卡：来源、规模、标注体系、类别分布、切段与划分、清洗与 OCR 流程、局限、合规、**未来用途** | 已完成 |
| `paper_sections_zh.md` | 可直接粘贴进论文的中文段落（4.1.1、4.1.5、4.2、4.3、**4.4**、5.2、5.3、结论修订建议） | 已完成 |
| `results.md` | 最终结果：语料统计、训练配置、逐 epoch 曲线、测试集指标、各类别指标、主要混淆对 | 自动生成 |
| `experiment_log.md` | 运行记录：环境硬件、时间线、5 次早期中断、OOM/重试计数、显存模型、产物清单、复现命令 | 已完成 |
| `params.json` | 四种冻结设定下的参数量、以及双冻结设定的逐模块参数量（由 `scripts/paper/count_params.py` 生成，无需 GPU） | 自动生成 |
| `dataset_stats.json` | 机器可读的语料统计（供脚本二次引用） | 自动生成 |
| `paper_numbers.json` | 论文标题级数字汇总（最优 epoch、测试指标、选定模型） | 自动生成 |
| `package_stats.csv` | 逐包明细（文档数、页数、块数、OCR 数） | 自动生成 |
| `class_distribution.csv` | 19 类的总量与 train/val/test 分布 | 自动生成 |
| `per_class_metrics.csv` | 各类别 × 各候选权重的 P/R/F1/support | 自动生成 |
| `tables.tex` | LaTeX（booktabs）表格：语料构成、类别分布、逐 epoch、测试结果、各类别 | 自动生成 |
| `figures/` | 训练曲线、类别分布、切段规模、OCR 覆盖、混淆矩阵（PNG） | 自动生成 |

## 一键再生成

```powershell
cd outputs\hmsan-bsa

# 0) 参数量表（无需 GPU，约 1 分钟）
python scripts\paper\count_params.py

# 1) 语料统计（纯标准库，8 秒左右）
python scripts\paper\collect_stats.py `
  --data_dir "C:\Users\Administrator\Desktop\训练数据\final_train" `
  --out_dir docs\paper_materials

# 2) 测试集评测（需要项目 venv：C:\hmsan\.venv\Scripts\python.exe）
python scripts\eval_test.py --output_dir outputs\hmsan_bsa_final `
  --checkpoints best_model.pt last.pt checkpoint_epoch5.pt `
  --out_json docs\paper_materials\eval_test.json

# 3) 图表与报告
python scripts\paper\make_figures.py
python scripts\paper\build_report.py
```

## 论文写作对接清单

- **表 1（语料构成）** → `tables.tex` 第 1 个表 / `dataset_card.md` 第 1 节
- **表 2（19 类分布）** → `tables.tex` 第 2 个表 / `dataset_card.md` 第 3 节
- **表 3（逐 epoch）** → `tables.tex` 第 3 个表 / `results.md` 第 3 节
- **表 4（测试集结果，候选权重对比）** → `tables.tex` 第 4 个表
- **表 4（参数效率 / 冻结设定）** → `params.json`、`scripts/paper/count_params.py`
- **表 5（交付模型各类别指标）** → `tables.tex` 第 5 个表
- **图：训练曲线** → `figures/fig_training_curves.png`
- **图：类别分布** → `figures/fig_class_distribution.png`
- **图：切段规模（论证显存约束）** → `figures/fig_segment_size.png`
- **图：OCR 覆盖** → `figures/fig_ocr_coverage.png`
- **图：混淆矩阵（交付模型）** → `figures/fig_confusion_matrix_last.png`（对照：`fig_confusion_matrix_best_model.png`）
- **第 4.1.1、4.1.5、4.2、4.3 节正文** → `paper_sections_zh.md`
- **第 4.4 节（测试集结果与交付模型）** → `paper_sections_zh.md`
- **第 5.2、5.3 节（含数据集后续用途）** → `paper_sections_zh.md`
- **实验环境、可复现性与测试集评测** → `experiment_log.md`（第 7 节）

## 需要人工确认的两点

1. **测试集的是否作为"性能声明"**：数据为真实业务文件，测试集与训练集同分布（同一批招标场景），
   因此指标反映的是**同分布泛化**，不代表跨机构、跨行业泛化能力；论文中应明确这一边界。
2. **OCR 质量**：`dataset_card.md` 第 7 节列出的局限（无逐张人工校对、空白率抽检）需要在论文
   局限性中如实写出，避免把"未识别的 47,140 个图片块"表述为识别失败。

## 交付模型

- `outputs\hmsan_bsa_final\last.pt` —— **交付模型**（第 8 轮，测试集 acc 0.9167 / 宏 F1 0.8854）
- `outputs\hmsan_bsa_final\best_model_by_macro_f1.pt` —— 上述权重的副本，文件名即选型依据
- `outputs\hmsan_bsa_final\best_model.pt` —— 训练脚本按验证准确率保存（第 2 轮），仅作对照
- `outputs\hmsan_bsa_final\checkpoint_epoch5.pt` —— 第 5 轮快照，仅作对照
- 评测日志：`outputs\hmsan_bsa_final\eval.log`；原始报告：`eval_test.json`
