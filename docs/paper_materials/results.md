# HMSAN-BSA 最终训练结果（论文素材）
> 生成脚本：`scripts/paper/build_report.py`；数据来源：`docs/paper_materials/dataset_stats.json`、`outputs/hmsan_bsa_final/history.json`、`docs/paper_materials/eval_test.json`。
> 所有 test 指标均来自 66 个 held-out 段，训练过程中从未参与梯度更新或模型选择。

## 1. 语料与切分
- 训练数据包 38 个，源文档 135 篇（去重后同名 PDF 132 个），共 52,344 页、456,602 个块。
- 标注覆盖率 99.98%（456,509 个已标注块）。
- 块类型：文本 371,796、图片无 OCR 47,140、图片含 OCR 37,666。
- 图片类块 OCR 覆盖 44.4%（37,666/84,806）。
- 按块数/页数上限切段后得到 434 段，按 `pdf_name` 分组随机划分为 train 303 / val 65 / test 66（seed=42）。

## 2. 训练配置

| 项目 | 设置 |
|---|---|
| seed | 42 |
| epochs | 8 |
| encoder_lr | 2e-05 |
| head_lr | 0.0001 |
| weight_decay | 0.0001 |
| grad_clip | 1.0 |
| scheduler | plateau (val accuracy, factor 0.5, patience 3) |
| class_weight_mode | none |
| label_smoothing | 0.0 |
| dropout | 0.1 |
| amp | bfloat16 autocast |
| max_blocks_per_sample | 3000 |
| max_pages_per_sample | 150 |
| train_ratio | 0.7 |
| val_ratio | 0.15 |
| batch_size_per_step | 1 |
| text_encoder | hfl/chinese-roberta-wwm-ext |
| text_max_length | 510 |
| image_encoder | google/vit-base-patch16-224 (frozen) |
| hidden_size | 128 |
| page_encoder_layers | 2 |
| page_encoder_heads | 4 |
| gsa | k_base=16, k_min=4, k_max=32, indexer_heads=4, indexer_dim=32 |
| num_global_tokens | 4 |
| inter_page_window | 3 |
| init_from | outputs/hmsan_bsa_ft3/best_model.pt (warm start) |
| hardware | 1 x NVIDIA GeForce RTX 3060 12GB, 8GB RAM, Windows 11, PyTorch 2.13.0+cu126 |

## 3. 逐 epoch 训练曲线（训练集 / 验证集）

| epoch | train acc | train macro-F1 | train loss | val acc | val macro-F1 | val loss |
|---|---|---|---|---|---|---|
| 1 | 0.8744 | 0.7273 | 1.2418 | 0.9621 | 0.8347 | 0.5232 |
| 2 | 0.9277 | 0.8117 | 0.7889 | 0.9736 | 0.8669 | 0.3601 |
| 3 | 0.9332 | 0.8493 | 0.6267 | 0.9037 | 0.8451 | 1.0190 |
| 4 | 0.9592 | 0.8921 | 0.4759 | 0.9643 | 0.8974 | 0.5337 |
| 5 | 0.9641 | 0.9084 | 0.3573 | 0.9717 | 0.9043 | 0.2969 |
| 6 | 0.9761 | 0.9195 | 0.2188 | 0.9192 | 0.8634 | 0.4758 |
| 7 | 0.9788 | 0.9362 | 0.1851 | 0.9725 | 0.8988 | 0.3943 |
| 8 | 0.9914 | 0.9508 | 0.0964 | 0.9656 | 0.8872 | 0.4246 |

- 验证集最优准确率：epoch 2，acc=0.9736，macro-F1=0.8669。
- 验证集最优宏 F1：epoch 5，macro-F1=0.9043，acc=0.9717。
- 训练侧最后一轮仍在上升（acc 0.9914、macro-F1 0.9508），而验证侧在第 5 轮前后进入平台期并波动（0.90–0.97 acc / 0.86–0.90 macro-F1），说明瓶颈更可能来自长尾类别的泛化与标注噪声，而不是训练不足；是否过拟合需结合测试集与各类别指标判断（见第 4 节）。

## 4. 测试集结果（66 段）

| 检查点 | 来源 epoch | checkpoint val acc | test acc | test macro-F1 | 参数量 |
|---|---|---|---|---|---|
| best_model.pt | 2 | 0.9736 | 0.8863 | 0.8183 | 191.2M |
| last.pt | 8 | 0.9656 | 0.9167 | 0.8854 | 191.2M |
| checkpoint_epoch5.pt | 5 | 0.9717 | 0.9178 | 0.8717 | 191.2M |

按宏 F1 选定的交付模型：**last.pt**（test macro-F1=0.8854，test acc=0.9167）。

### 4.1 各类别指标（选定模型）

| id | 类别 | precision | recall | F1 | support |
|---|---|---|---|---|---|
| 12 | 评分支撑材料 | 0.9823 | 0.8964 | 0.9374 | 48,456 |
| 9 | 财务状况 | 0.8707 | 0.9358 | 0.9021 | 12,652 |
| 8 | 资格证明文件 | 0.9084 | 0.9863 | 0.9457 | 10,442 |
| 17 | 其他 | 0.6477 | 0.9421 | 0.7676 | 4,315 |
| 5 | 基本情况表 | 0.9545 | 0.7444 | 0.8365 | 2,312 |
| 3 | 投标保证金 | 0.9326 | 0.9967 | 0.9636 | 2,098 |
| 1 | 目录 | 0.9361 | 0.9834 | 0.9592 | 1,445 |
| 13 | 名称变更 | 0.4047 | 0.9759 | 0.5721 | 498 |
| 4 | 关系说明 | 0.9863 | 0.9945 | 0.9904 | 363 |
| 18 | 法定代表人授权委托书 | 0.8805 | 0.9085 | 0.8943 | 284 |
| 2 | 商务偏差表 | 1.0000 | 0.9961 | 0.9980 | 255 |
| 15 | 十不准 | 0.9868 | 0.9073 | 0.9454 | 248 |
| 6 | 营业执照 | 0.9247 | 0.6308 | 0.7500 | 214 |
| 0 | 封面页 | 0.8327 | 1.0000 | 0.9087 | 204 |
| 10 | 财务凭证单 | 0.8757 | 0.9548 | 0.9136 | 155 |
| 16 | 公章授权书 | 0.9364 | 0.9626 | 0.9493 | 107 |
| 14 | 一致性承诺函 | 0.8487 | 0.9806 | 0.9099 | 103 |
| 7 | 税务证明 | 0.7111 | 0.9505 | 0.8136 | 101 |
| 11 | 资质业绩凭证单 | 0.9333 | 0.8077 | 0.8660 | 52 |

### 4.2 主要混淆对（top 10，行归一化比例）

| 真实类别 | 预测类别 | 数量 | 占该类比例 |
|---|---|---|---|
| 评分支撑材料 | 其他 | 2,161 | 4.5% |
| 评分支撑材料 | 财务状况 | 1,595 | 3.3% |
| 评分支撑材料 | 名称变更 | 705 | 1.5% |
| 财务状况 | 评分支撑材料 | 640 | 5.1% |
| 基本情况表 | 资格证明文件 | 500 | 21.6% |
| 评分支撑材料 | 资格证明文件 | 418 | 0.9% |
| 其他 | 财务状况 | 163 | 3.8% |
| 财务状况 | 资格证明文件 | 105 | 0.8% |
| 评分支撑材料 | 目录 | 73 | 0.2% |
| 营业执照 | 基本情况表 | 70 | 32.7% |

## 5. 复现命令

```
python scripts/paper/collect_stats.py \
  --data_dir "C:\Users\Administrator\Desktop\训练数据\final_train" \
  --out_dir docs/paper_materials
python scripts/eval_test.py --output_dir outputs/hmsan_bsa_final \
  --out_json docs/paper_materials/eval_test.json
python scripts/paper/make_figures.py
python scripts/paper/build_report.py
```
