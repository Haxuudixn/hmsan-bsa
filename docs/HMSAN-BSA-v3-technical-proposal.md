# HMSAN-BSA v3 技术方案

## 基于门控稀疏注意力的层次化记忆网络投标文件切片分类

---

## 1. 问题定义

### 1.1 任务描述

投标文件是高度结构化的多页文档，包含封面、目录、商务偏离表、资质证明、财务报表等十余种章节类型。本方案针对**多模态投标文件切片分类**任务，将 PDF 文档按 `Document → Page → Block` 三层结构解析后，对每个 Block 进行细粒度类别标注。

### 1.2 技术挑战

| 挑战 | 描述 |
|---|---|
| **长文档建模** | 投标文件可达 150+ 页，传统 Transformer 受限于 O(N²) 复杂度 |
| **跨页上下文** | "财务报表"章节可能跨 40+ 页，需要记忆远距离章节边界 |
| **多模态融合** | Block 分为纯文本/纯图像/图文混合三种，需自适应融合策略 |
| **稀疏标注** | 当前已标注 907 个 Block（来自 1 个 PDF），未标注 Block 不可作负样本 |

### 1.3 数据概况

| 指标 | 数值 |
|---|---|
| 总 Block 数 | 85,304 |
| 文档数 (PDF) | 24 |
| 页面数 | 8,750 |
| Block 类型 | text (68,617) / image |
| 已标注 Block | 907 |
| 类别数 | 19 |
| 文本 Block 中位数字符数 | 17 |
| 文本 Block 最大字符数 | 1,910 |

---

## 2. 整体架构

### 2.1 数据流

```
                     ┌─────────────────────────────────────────┐
  all_blocks.xlsx ──→│ Dataset (Document/Page/Block hierarchy) │
                     └─────────────────┬───────────────────────┘
                                       │
  ┌────────────────────────────────────┼──────────────────────────────┐
  │                        HMSAN-BSA v3                               │
  │                                                                    │
  │  ┌──────────────────────────────────────────────────────────────┐ │
  │  │  ① Gated LiLT Block Encoder  ← 特征提取（Gated Sparse #1）   │ │
  │  │     RoBERTa + ViT + Layout ──→ GatedFusionGate ──→ block_vec │ │
  │  └──────────────────────────────┬───────────────────────────────┘ │
  │                                 │                                  │
  │  ┌──────────────────────────────┼───────────────────────────────┐ │
  │  │  ② Dilated Gated Page Encoder ← 页内聚合（Gated Sparse #2）  │ │
  │  │     dil=1,2,4,8 + top-k gate → page_repr + encoded_blocks    │ │
  │  └──────────────────────────────┬───────────────────────────────┘ │
  │                                 │                                  │
  │  ┌──────────────────────────────┼───────────────────────────────┐ │
  │  │  ③ Gated Infini Memory ← 页间传播（Gated Sparse #3）         │ │
  │  │     delta-rule压缩 + gate_w写入 + gate_r读取 + β融合         │ │
  │  └──────────────────────────────┬───────────────────────────────┘ │
  │                                 │                                  │
  │  ┌──────────────────────────────┼───────────────────────────────┐ │
  │  │  ④ Heads: BoundaryHead / ClassificationHead / SectionHead    │ │
  │  └──────────────────────────────────────────────────────────────┘ │
  └────────────────────────────────────────────────────────────────────┘
```

### 2.2 模型规格

| 参数 | 值 | 说明 |
|---|---|---|
| `hidden_size` | 128 | 统一隐层维度 |
| `num_classes` | 19 | 投标文件章节类型 |
| `num_boundaries` | 4 | B/I/E/O 边界标签 |
| `dropout` | 0.1 | 全局 dropout |

---

## 3. 模块一：Gated LiLT 块编码器

### 3.1 设计动机

传统多模态融合采用 `concat(RoBERTa, ViT, Layout) → Linear` 的固定策略，忽略了不同 Block 类型对模态的依赖差异：
- 纯文本 Block：应主要依赖 RoBERTa 编码
- 纯图像 Block（公章、签名）：应主要依赖 ViT 编码
- 图文混合 Block：需 OCR 文本 + 截图的联合编码

### 3.2 架构设计

```
                    ┌──────────────────────────────────┐
  text_content ────→│ RoBERTa-wwm-ext ──→ text_emb     │
                    │                    (768d)        │
  block_image ─────→│ ViT-B/16 ────────→ image_emb    │
                    │                    (768d)        │
  bbox/font/ ──────→│ LayoutEncoder ───→ layout_emb   │
  color/type        │                    (64d)         │
                    │                                  │
                    │  ┌───────────────────────────┐   │
                    │  │    GatedFusionGate        │   │
                    │  │                           │   │
                    │  │  layout_emb ──→ MLP ──→   │   │
                    │  │  softmax(α_text,          │   │
                    │  │          α_layout,        │   │
                    │  │          α_image)         │   │
                    │  │                           │   │
                    │  │  out = Σ α_i · emb_i      │   │
                    │  └───────────────────────────┘   │
                    │              │                   │
                    │              ▼                   │
                    │  LiLT Layout Bias (residual)     │
                    │              │                   │
                    │              ▼                   │
                    │         block_vec (128d)         │
                    └──────────────────────────────────┘
```

### 3.3 Gate 机制

布局特征（bbox 位置、字体大小、是否加粗、字体颜色、block_type）通过 MLP 预测三模态权重：

```
α = softmax(W₂ · ReLU(W₁ · layout_emb + b₁) + b₂)
```

直觉：字体大小异常大的 Block 往往是标题（文本更重要）；image 类型的 Block 图像权重自然更高；居中的 bbox 可能是公章（图像/混合权重更高）。

### 3.4 LiLT 布局偏置

从 bbox 提取 5 维空间特征 `[cx, cy, w, h, area]`，投影为 attention bias 注入编码，使空间位置相近的 Block 获得隐式交互。

### 3.5 关键参数

| 参数 | 值 |
|---|---|
| 文本编码器 | `hfl/chinese-roberta-wwm-ext` |
| 最大文本长度 | 510 tokens（超长首尾截断） |
| 图像编码器 | `google/vit-base-patch16-224` |
| 布局输出维 | 64 |
| Gate 模态数 | 3 (text, layout, image) |

---

## 4. 模块二：Dilated Gated 页编码器

### 4.1 设计动机

投标文件单页可能包含 10~50 个 Block。全连接自注意力 O(N²) 过高，但固定小窗口（如 w=5）无法捕捉跨块的长程依赖（如页首标题与页尾表格的语义关联）。

LongNet 的扩张注意力通过逐层倍增 dilation 实现指数增长的感受野，但固定的扩张窗口内仍然包含不相关的 token。引入 Gated Sparse 在扩张窗口内进一步做 top-k 筛选。

### 4.2 扩张 + 门控机制

```
Layer 0 (dil=1):  ████████████████████         每个token看 ±5 邻域, gate选 top-16
                  ↓ gate top-k

Layer 1 (dil=2):  ██░█░█░█░█░█░█░█░█░██       每个token看间隔=2的 ±5 邻域, gate选 top-16
                  ↓ gate top-k                    有效覆盖 ≈ 10 blocks

Layer 2 (dil=4):  █░░░█░░░█░░░█░░░█░░░█         有效覆盖 ≈ 20 blocks
                  ↓ gate top-k

Layer 3 (dil=8):  █░░░░░░░█░░░░░░░█░░░░░░░█     有效覆盖 ≈ 40 blocks
                  ↓ gate top-k                    可覆盖整页
```

### 4.3 Gate 公式

对每一层、每个 head，在扩张候选集内：

```
g_ij = σ(W · [Q_i; K_j] / τ)      // 逐对门控得分
mask  = top-k(g)                    // 保留 top-k 个连接
attn  = softmax(QKᵀ/√d + mask)     // 只在门控选中的连接上计算注意力
```

训练时使用 **Straight-Through Estimator**：前向用离散 mask，反向用连续 gate 梯度。

### 4.4 全局 Token

除扩张窗口外，每层保留 4 个可学习全局 token。全局 token 与所有 Block token 全连接，作为页级信息聚合的"总线"。

### 4.5 关键参数

| 参数 | 值 |
|---|---|
| 层数 | 4 |
| 头数 | 4 |
| 基础窗口 | 5 |
| Dilation | [1, 2, 4, 8] |
| 全局 token | 4 |
| Gate top-k | 16 |
| 第 4 层有效覆盖 | ~40 blocks |

---

## 5. 模块三：Gated Infini 压缩记忆

### 5.1 设计动机

投标文件的章节通常跨越多页（如"财务报表"跨 40+ 页）。v1 的固定 K=4 槽位 GRU 记忆存在两个问题：

1. **容量瓶颈**：4 个槽位 × 128d = 512d 总容量，难以压缩 40 页的语义信息
2. **更新刚性**：GRU 门控对所有维度等权更新，无法选择性保留关键信息

Infini-attention 用压缩记忆矩阵实现"无限"容量，但原始 Infini 对所有维度等价更新。引入 **Gated Sparse** 实现逐维选择性读写。

### 5.2 记忆结构

```
Memory Matrix: M ∈ R^(K × d_key × d_value)
  K  = 4  (槽位数，对应不同上下文类型)
  d_key = 64
  d_value = 128
  Total capacity: 4 × 64 × 128 = 32,768 维
```

### 5.3 门控写入

```
k = ELU(W_k · page_repr) + 1       // 正值key（稳定delta-rule）
v = W_v · page_repr                 // value
gate_w = σ(W_gw · page_repr)       // 逐维写入门控 [d_value]

retrieval = σ(k)ᵀ @ M              // 当前记忆检索
error     = v - retrieval           // 预测误差
delta     = gate_w ⊙ (k ⊗ error)   // 门控增量
M_new     = M + delta               // 选择性更新
```

**直觉**：gate_w 学习哪些语义维度与当前页相关，只更新相关维度，避免无关信息污染记忆。

### 5.4 门控读取

```
q = ELU(W_q · page_repr) + 1
retrieval = σ(q) @ M / (σ(q) @ z + ε)     // 归一化检索
gate_r = σ(W_gr · [page_repr; retrieval])  // 逐维读取门控
output = gate_r ⊙ retrieval                // 选择性读取
```

### 5.5 β 融合

```
β = σ(W_β · [local_out; memory_out])
fused = β · local_out + (1-β) · memory_out
```

可学习的 β 门控决定当前页更多依赖局部注意力还是跨页记忆。

### 5.6 与非门控 Infini 的对比

| 维度 | 原始 Infini | Gated Infini |
|---|---|---|
| 写入 | M += σ(K)ᵀ(V - σ(K)M) | M += gate_w ⊙ σ(K)ᵀ(V - σ(K)M) |
| 读取 | σ(Q)M / σ(Q)z | gate_r ⊙ σ(Q)M / σ(Q)z |
| 门控参数 | 无 | 2 个 MLP (write_gate + read_gate) |
| 额外计算 | — | O(d²) per gate |
| 选择性 | 无（全维等权） | 逐维自适应 |

### 5.7 关键参数

| 参数 | 值 |
|---|---|
| 槽位 K | 4 |
| d_key | 64 |
| d_value | 128 |
| 总容量/槽 | 8,192 维 |
| PageMemory + SectionMemory | 双路独立 Infini |

---

## 6. 训练策略

### 6.1 损失函数

多任务联合损失：

```
L_total = L_cls + α_boundary · L_boundary + β_section · L_section
```

| 损失 | 公式 | ignore_index |
|---|---|---|
| L_cls | CrossEntropy(block_logits, y_cls) | -100（未标注 Block） |
| L_boundary | CrossEntropy(boundary_logits, y_boundary) | -100（未标注页） |
| L_section | CrossEntropy(section_logits, y_section) | -100（未标注页） |

### 6.2 训练阶段

| 阶段 | 内容 | 目的 |
|---|---|---|
| Stage 1 | 冻结 RoBERTa + ViT，训练 GatedFusionGate + LayoutEncoder | 验证门控融合有效性 |
| Stage 2 | 解冻 RoBERTa + ViT 顶层，端到端微调 BlockEncoder | 提升特征质量 |
| Stage 3 | 单页分类训练（无记忆传播） | 建立 Block 级 baseline |
| Stage 4 | 加入 DilatedGatedPageEncoder | 验证扩张+门控注意力 |
| Stage 5 | 加入 GatedInfiniMemory，滑动窗口训练 | 验证跨页记忆 |
| Stage 6 | 全文档端到端训练 (Joint) | 最终模型 |

### 6.3 训练超参数

| 参数 | 值 |
|---|---|
| batch_size | 1 (按文档) |
| learning_rate | 1e-3 (Gate/Head) / 2e-5 (RoBERTa/ViT) |
| weight_decay | 1e-4 |
| α_boundary | 0.5 |
| β_section | 0.5 |
| 滑动窗口 | 10 页 |

---

## 7. 评估方案

### 7.1 Block 级指标

- Accuracy（仅已标注 Block）
- Macro-F1 / Weighted-F1
- 每类别 Precision / Recall / F1

### 7.2 边界指标

- Boundary Precision / Recall / F1
- Section-Start Detection Accuracy

### 7.3 章节级指标

- Segment Accuracy
- Section IoU
- Over-segmentation / Under-segmentation 计数

### 7.4 消融实验设计

| 实验 | 配置 | 验证目标 |
|---|---|---|
| Baseline | concat 融合 + 固定窗口 + GRU 记忆 | v1 baseline |
| +Gated Fusion | 加入 GatedFusionGate | 门控融合增益 |
| +Dilated Gate | 加入扩张+门控注意力 | 稀疏注意力增益 |
| +Infini | 加入 Gated Infini 记忆 | 压缩记忆增益 |
| Full v3 | 全部开启 | 整体增益 |

---

## 8. 文件结构

```
src/bid_slicing/
├── features/
│   ├── layout_features.py          # 布局编码 (bbox/font/color/type → 64d)
│   ├── text_features.py            # RoBERTa-wwm-ext 封装
│   ├── image_features.py           # ViT-B/16 封装
│   ├── gated_lilt_encoder.py       # Gated LiLT Block Encoder  ← 新增
│   └── __init__.py
├── models/
│   ├── block_encoder.py            # v1 BlockMultiModalEncoder (保留)
│   ├── page_encoder.py             # v1 PageEncoder + InterPageAttn (保留)
│   ├── memory.py                   # v1 PageMemory + SectionMemory (保留)
│   ├── boundary_head.py            # BoundaryHead + ClassificationHead + SectionHead
│   ├── gated_sparse.py             # GatedSparsifier + GatedMHA + GatedFusionGate ← 新增
│   ├── dilated_gated_attention.py  # DilatedGatedPageEncoder  ← 新增
│   ├── gated_infini_memory.py      # GatedInfiniMemory  ← 新增
│   ├── hmsan_bsa.py                # v1 完整模型 (保留)
│   ├── hmsan_bsa_v3.py             # v3 完整模型  ← 新增
│   └── __init__.py
├── data/                           # 数据加载、标注 schema
├── training/                       # 训练循环、loss、metrics (待实现)
├── inference/                      # 推理、解码、导出 (待实现)
└── utils/                          # 工具函数
```

---

## 9. 参考文献

| 论文 | 年份 | arXiv ID | 关联模块 |
|---|---|---|---|
| Leave No Context Behind: Infini-attention | 2024 | `2404.07143` | 压缩记忆 |
| Gated Sparse Attention | 2026 | `2601.15305` | 全部三个模块 |
| MKA: Memory-Keyed Attention | 2026 | `2603.20586` | 压缩记忆 |
| LongNet: Scaling to 1B Tokens | 2023 | `2307.02486` | 扩张注意力 |
| LiLT: Language-Independent Layout Transformer | 2022 | `2202.13669` | 布局编码 |
| LayoutLMv3: Pre-training for Document AI | 2022 | `2204.08387` | 布局编码 |
| LayoutMask: Enhance Text-Layout Interaction | 2023 | `2305.18721` | 布局编码 |
| Long-Range Transformer for Doc Understanding | 2023 | `2309.05503` | 长文档 |
| Ring Attention with Blockwise Transformers | 2023 | `2310.01889` | 批量处理 |
| Block Sparse Flash Attention | 2025 | `2512.07011` | 稀疏加速 |
| Multimodal RAG for Document Understanding (Survey) | 2025 | `2510.15253` | 检索增强 |
---

## 10. 后续方向

| 方向 | 描述 | 优先级 |
|---|---|---|
| Mamba-2 页编码 | 用 SSM 替代 Dilated Gate Transformer，线性复杂度 | ⭐⭐ |
| CRF 边界解码 | Linear-chain CRF 约束 B→I→E 转移合法性 | ⭐⭐ |
| 课程学习 | 单页→多页→全文档渐进训练 | ⭐ |
| RAG 增强 | 检索相似章节标注辅助分类 | ⭐⭐⭐ |
| 无标注 Block 伪标签 | 用已训练模型生成伪标签扩充训练集 | ⭐⭐ |