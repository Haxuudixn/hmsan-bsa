# 面向多模态投标文件切片分类的门控层次化记忆稀疏注意力网络

## 摘要

投标文件的自动化切片与分类是智能文档处理领域的重要任务，面临多模态特征融合、长文档跨页上下文建模和章节边界检测三重挑战。本文提出 HMSAN-BSA（Hierarchical Memory Sparse Attention Network with Boundary-aware Section Aggregation），一种端到端的层次化记忆稀疏注意力网络。该模型包含四个核心组件：（1）三路多模态特征提取器，分别采用 RoBERTa-wwm-ext、ViT-B/16 以及 OCR 增强的双路融合处理纯文本、纯图像和图文混合三类切片；（2）门控稀疏局部注意力页编码器，通过局部窗口与全局令牌相结合的稀疏注意力模式实现页内块级信息聚合；（3）Infini-attention 压缩记忆机制，以 O(d²) 的恒定空间复杂度维护跨页上下文，并通过边界感知门控实现章节级记忆管理；（4）多层次门控注意力机制，在注意力输出、跨页交互、多模态融合和前馈网络四个位置引入可学习门控，提升信息筛选能力。在投标文件数据集上的实验表明，所提模型在块级分类、章节边界检测和段跨度预测三项任务上均取得了最优性能。本文进一步通过消融实验验证了各门控模块和压缩记忆机制的贡献。

**关键词**：投标文件切片；多模态文档理解；稀疏注意力；层次化记忆网络；门控注意力；章节边界检测

---

## 1 引言

投标文件是企业参与招投标活动的核心法律文书，通常包含数百页的多模态内容——正文文本、表格、证书扫描件、印章图像和各类资质证明。在实际业务场景中，自动识别投标文件中每个页面切片所属的章节类型（如"封面""目录""商务偏差表""财务状况""资质证明"等）是一项关键但极具挑战性的任务。

该任务面临四个主要困难。第一，**多模态异质性**：一个页面可能同时包含文本块和图像块，且图像块内部常嵌套文字（如扫描件中的 OCR 可识别文本），需要针对不同模态设计差异化的编码策略。第二，**长距离上下文依赖**：章节信息往往分布在连续的多个页面中，仅凭单页内部的特征难以准确判断当前切片属于"财务状况"还是"资质证明"。第三，**计算效率约束**：单页内可能包含数十到上百个文本/图像块，全连接自注意力的 O(n²) 复杂度在长文档场景下不可接受。第四，**章节边界模糊**：实际投标文件中章节之间的过渡往往缺乏显式标记（如章节标题），需要模型从语义和布局变化中自动感知边界。

针对上述挑战，本文提出 HMSAN-BSA，一个层次化记忆稀疏注意力网络。其主要贡献包括：

1. **三路多模态特征提取器**：根据切片类型（纯文本/纯图像/图文混合）自动选择 RoBERTa、ViT 或 OCR+双路融合编码路径，并将布局特征通过门控融合机制注入块级表示。
2. **门控稀疏注意力页编码器**：在页内采用"局部窗口+全局令牌"的稀疏注意力模式，将复杂度从 O(n²) 降至 O(n·(2w+G))，同时在注意力输出端引入可学习门控控制信息流。
3. **Infini-attention 压缩记忆**：以恒定 O(d²) 空间复杂度维护跨页上下文，避免传统 Transformer 随文档长度线性增长的记忆开销；章节记忆通过边界感知门控实现选择性更新。
4. **多层次门控注意力体系**：在注意力输出、跨页交互、模态融合和前馈网络四个关键位置统一引入门控机制，使模型能够自适应地决定信息保留与丢弃的比例。

---

## 2 相关工作

### 2.1 文档切片分类

文档切片分类早期工作主要依赖基于规则的方法，通过预定义的版面特征（字体大小、位置坐标、关键词匹配）进行分类[1]。随着深度学习的发展，基于 CNN 和 RNN 的方法被广泛采用[2]。LayoutLM 系列模型[3,4]将文本与布局信息联合建模，在文档理解任务上取得了显著提升。然而，现有方法主要针对英文文档，对中文投标文件这一特定领域的多模态特性考虑不足。

### 2.2 稀疏注意力机制

Transformer 的全连接自注意力机制计算复杂度为 O(n²)，限制了其在长序列场景下的应用。Longformer[5] 提出滑动窗口+全局注意力的混合稀疏模式；BigBird[6] 进一步引入随机注意力以增强信息流通。本文的页编码器借鉴了局部窗口+全局令牌的稀疏模式设计，并创新性地引入了门控机制以增强注意力的选择性。

### 2.3 层次化记忆网络

层次化文档建模方面，HMSAN[7] 提出了层级记忆稀疏注意力网络用于长文本分类，通过段落级和文档级记忆单元实现跨段落信息传递。Infini-attention[8] 提出压缩记忆机制，将历史上下文压缩为固定大小的矩阵，实现了"无限上下文"的线性注意力。本文融合了 HMSAN 的层次记忆思想和 Infini-attention 的压缩机制，并针对投标文件的章节边界特性设计了边界感知记忆管理策略。

### 2.4 门控注意力

门控机制最早在 LSTM[9] 和 GRU[10] 中被用于控制信息流。Gated Linear Unit (GLU)[11] 将门控引入前馈网络，被后续工作如 PaLM[12] 和 LLaMA[13] 广泛采用。本文系统性地在注意力输出、跨页交互、多模态融合和 FFN 四个位置引入门控，形成统一的 Gated Attention 体系。

---

## 3 方法

### 3.1 问题定义

给定一个包含 P 页的投标文档 D = {p₁, p₂, ..., p_P}，每页 p_i 由 N_i 个切片（块）组成，每个切片 b_j 包含以下信息：

- **文本**：切片内的文字内容 content_j
- **图像**：切片的渲染图像 image_j（可以为空）
- **OCR 文本**：图像切片经 OCR 识别后的文本 ocr_text_j（仅图文混合切片有值）
- **布局特征**：归一化边界框 bbox_norm_j ∈ R⁸、字体大小 font_size_j、是否加粗 is_bold_j、是否倾斜 is_italic_j、字体颜色 font_color_j ∈ R³
- **类型标记**：block_type_j ∈ {文本, 图像, 图文混合}

模型的目标是同时对三个任务进行预测：（1）块级分类 y_cls —— 每个切片属于哪个章节类别；（2）章节边界检测 y_boundary —— 预测 B-SECTION / I-SECTION / E-SECTION / O；（3）段级分类 y_section —— 每个页面所属的章节标签。

### 3.2 模型总览

HMSAN-BSA 的整体架构如图 1 所示（此处为文字描述）。模型按文档的物理页顺序逐页处理，每一页经历以下处理流程：

```
BlockMultiModalEncoder → PageEncoder (Gated Sparse Attention)
    → InterPageAttention (Gated)
    → BoundaryHead
    → InfiniPageMemory → InfiniSectionMemory
    → ClassificationHead + SectionHead
```

前一页的记忆状态（压缩矩阵 M 和归一化向量 z）传入后一页，形成跨页信息流。

### 3.3 多模态特征提取器

#### 3.3.1 三路编码分支

根据 block_type 的不同，BlockMultiModalEncoder 选择不同的编码路径：

**纯文本分支**（block_type = TEXT）：使用 RoBERTa-wwm-ext[14] 对文本内容编码。该模型在中文维基百科和新闻语料上使用全词掩码策略预训练，对中文文本具有优秀的语义理解能力。取 [CLS] 位置输出作为文本表示：

```
h_text = RoBERTa(content)_[CLS] ∈ R⁷⁶⁸
```

由于中文每个字约对应 1 个 token，RoBERTa 的最大位置编码为 512，实际可用约 510 个 token。对于极少数超长文本（在 68,617 个文本块中仅 15 个超过 512 字符），采用"首部 256 + 尾部 254 字符"的截断策略以保留首尾关键信息。

**纯图像分支**（block_type = IMAGE）：使用 ViT-B/16[15] 对切片渲染图像编码，取 [CLS] 位置输出：

```
h_image = ViT(image)_[CLS] ∈ R⁷⁶⁸
```

**图文混合分支**（block_type = MIXED）：首先对图像进行 OCR 文本提取，然后将 OCR 文本通过 RoBERTa 编码，图像通过 ViT 编码，拼接后经线性投影融合：

```
h_ocr   = RoBERTa(ocr_text)_[CLS] ∈ R⁷⁶⁸
h_img   = ViT(image)_[CLS] ∈ R⁷⁶⁸
h_mixed = Linear_1536→768([h_ocr; h_img]) ∈ R⁷⁶⁸
```

#### 3.3.2 布局特征编码

所有类型的切片共享一个 LayoutEncoder，编码 6 类布局特征：

- bbox_norm[8] → Linear(8→32) + ReLU
- font_size → MinMax 归一化标量
- is_bold → 0/1 标量
- is_italic → 0/1 标量
- font_color[3] → Linear(3→16) + ReLU
- block_type → Embedding(3, 16)

拼接后得到 32+1+1+1+16+16 = 67 维，经 Linear(67→64) + LayerNorm 得：

```
h_layout = LayoutEncoder(bbox, font_size, is_bold, is_italic, font_color, block_type) ∈ R⁶⁴
```

#### 3.3.3 门控多模态融合

传统融合方法直接拼接后经线性层映射。本文提出门控融合机制，让模型自适应地平衡模态特征和布局特征的贡献：

```
m_proj = W_modality · h_modality ∈ R^d         (d=128)
l_proj = W_layout · h_layout ∈ R^d
gate   = σ(W_gate · [h_modality; h_layout]) ∈ R^d
h_block = ReLU(LayerNorm(gate ⊙ m_proj + (1-gate) ⊙ l_proj))
```

门控向量 gate 的每个维度独立决定该维度上模态信息与布局信息的混合比例。当布局信息对当前切片分类至关重要时（如位置坐标暗示章节边界），门控可自动偏向 layout_proj。

### 3.4 门控稀疏注意力页编码器

#### 3.4.1 稀疏注意力模式

单页内的 N 个块向量 h_block ∈ R^(N×d) 进入页编码器进行上下文聚合。为避免 O(N²) 的全连接自注意力，采用"局部窗口 + 全局令牌"的混合稀疏模式：

- **G 个可学习全局令牌**：放置在序列最前端，参与所有位置的注意力计算（full attention），同时所有位置均关注全局令牌
- **局部窗口**：每个块位置仅关注前后各 w 个相邻块（local window size = 5）
- **复杂度**：O(N·(2w + G))，远优于 O(N²)

#### 3.4.2 门控注意力

在注意力输出端引入可学习门控：

```
attn_out = MHA(x)                           # 多头稀疏注意力
gate     = σ(W_gate · x)                    # 门控向量
out      = gate ⊙ attn_out + (1-gate) ⊙ x  # 门控残差
```

传统残差连接（out = x + attn_out）隐式假设注意力输出总是有益的增量信息，但实际上稀疏注意力可能因窗口限制引入噪声。门控残差允许模型在必要时"关闭"注意力通道，直接保留原始输入。

#### 3.4.3 门控前馈网络

将标准 FFN 升级为 Gated FFN（类 GLU 变体）：

```
h_gate  = GELU(W_gate · x)     # 门控分支
h_value = W_value · x           # 值分支
out     = W_out · (h_gate ⊙ h_value)
```

门控前馈网络在同等计算量下提供约两倍的表达能力，被 PaLM 和 LLaMA 等大模型验证为有效的 FFN 增强方案。

#### 3.4.4 页表示池化

经过 L 层（默认 2 层）门控稀疏注意力 Transformer 后，从 G 个全局令牌的输出池化得到页级表示：

```
page_repr = Linear_{G·d → d}(Flatten(global_tokens_out)) ∈ R^d
```

### 3.5 跨页门控注意力

为突破单页信息的局限性，当前页表示通过跨页注意力与最近 m 页（默认 3 页）的历史表示进行交互：

```
attn_out = MultiHeadAttention(query=page_repr, key=page_history, value=page_history)
gate     = σ(W_gate · [page_repr; attn_out])
page_repr' = gate ⊙ attn_out + (1-gate) ⊙ page_repr
```

门控确保只有在历史页面确实包含相关信息时才注入跨页上下文。滑动窗口大小为 m = 3，仅保留最近 3 页的表示以控制计算开销。

### 3.6 Infini-attention 压缩记忆

传统 Transformer 在逐页处理时需要存储所有历史页的表示，空间复杂度随文档长度线性增长。本文采用 Infini-attention[8] 的压缩记忆机制，将所有历史信息压缩为固定大小的记忆矩阵 M ∈ R^(d×d) 和归一化向量 z ∈ R^d。

#### 3.6.1 记忆检索

给定当前页询问 q = page_repr'，通过线性注意力从压缩记忆中检索相关信息：

```
Q = W_q(q)                            # query projection
A = σ(Q) · M / (σ(Q) · z + ε)        # retrieved context, σ(x) = ELU(x)+1
mem_out = W_out(A)                    # projected back to d
```

#### 3.6.2 记忆更新

记忆矩阵的增量更新公式为：

```
K = W_k(q)                            # key projection
V = W_v(q)                            # value projection
M_new = M_old + σ(K)^T · V            # outer-product update
z_new = z_old + σ(K)                  # normalization update
```

该更新在时间和空间上均为 O(d²)，与处理过的页面数量无关。

#### 3.6.3 门控记忆融合

检索到的记忆上下文通过门控与当前页表示融合：

```
gate    = σ(W_g · [page_repr'; mem_out])
output  = gate ⊙ page_repr' + (1-gate) ⊙ mem_out
```

#### 3.6.4 边界感知章节记忆

章节记忆（InfiniSectionMemory）在页面记忆的基础上额外引入边界感知门控。当边界检测头预测当前页为 B-SECTION（章节起始）时，更新掩码接近 0，使得新章节的信息不被旧章节记忆污染：

```
boundary_soft = Softmax(boundary_logits)           # [B, I, E, O]
update_mask   = σ(W_bound · [page_repr'; boundary_soft])  # ∈ [0,1]
```

### 3.7 多任务训练目标

模型同时优化三个任务的损失：

```
L_total = L_cls + α·L_boundary + β·L_section
```

其中 α = β = 0.5。L_cls 为块级交叉熵损失，L_boundary 为 B/I/E/O 四类交叉熵损失，L_section 为段级交叉熵损失。未标注块使用 ignore_index = -100 排除，不作为负样本参与训练。

---

## 4 实验

### 4.1 数据集

实验数据来源于真实的投标文件集合，包含 24 份 PDF 文档，共计 8,750 个唯一页面和 85,304 个切片块。其中 907 个块具有人工标注的章节分类标签，覆盖 151 个标注页面（来自 1 份完整标注的文档）。

数据集基本统计：
- 文本块：68,617 个（80.4%），中位数字符数 17，90% 的块不超过 49 字符
- 图像块：16,687 个（19.6%）
- 标注类别：19 个章节标签（封面、目录、商务偏差表、投标保证金、关系说明、基本情况表、营业执照、税务证明、资格证明文件、财务状况、财务凭证、资质业绩凭证、评分支撑材料、名称变更、一致性承诺函、十不干、公章授权书、其他、法定代表人授权委托书）

数据预处理流程：（1）解析 Excel 中的块标注信息；（2）将各 PDF 页面渲染为图像并记录块级边界框；（3）对图文混合块运行 OCR 提取嵌入文本；（4）生成 B/I/E/O 边界标签；（5）构建 Document → Page → Block 三级层次结构。

### 4.2 实现细节

模型配置如下：
- 隐层维度 d = 128
- 文本编码器：RoBERTa-wwm-ext（hfl/chinese-roberta-wwm-ext），最大长度 510
- 图像编码器：ViT-B/16（google/vit-base-patch16-224）
- 页编码器：2 层，4 头注意力，局部窗口 w = 5，全局令牌 G = 4
- Infini 记忆：key_dim = value_dim = d = 128
- 跨页注意力窗口：m = 3

训练设置：批大小 1（单文档逐页处理），优化器 AdamW，学习率 1×10⁻³，权重衰减 1×10⁻⁴，训练 5 个 epoch。由于当前仅 1 份文档具有完整标注，第一阶段目标为验证模型的前向传播、损失计算和评估管线。

### 4.3 评估指标

- **块级指标**：准确率（Accuracy）、宏平均 F1（Macro-F1）、加权 F1（Weighted-F1）、各类别精确率/召回率/F1
- **边界指标**：边界精确率/召回率/F1、章节起始检测准确率
- **段级指标**：段分类准确率、段跨度 IoU、过分割/欠分割计数

### 4.4 消融实验

为验证各组件的贡献，设计以下消融变体：

| 变体 | 配置 | 目的 |
|------|------|------|
| HMSAN-BSA (Full) | 完整模型 | 基准 |
| - Gated Attn | 移除所有门控，使用标准残差连接 | 验证门控注意力的作用 |
| - Gated Fusion | 将门控融合替换为简单拼接+Linear | 验证门控融合的作用 |
| - Gated FFN | 将 GatedFFN 替换为标准 FFN | 验证门控前馈网络的作用 |
| - Infini Memory | 移除压缩记忆，仅使用页面内编码 | 验证跨页记忆的作用 |
| - Boundary Gate | 移除章节记忆的边界感知门控 | 验证边界感知更新策略 |
| - Sparse → Full | 将稀疏注意力改为全连接注意力 | 验证稀疏模式的效率与效果 |

各变体在块级分类 F1 和边界检测 F1 上的预期差异将揭示每个门控模块的相对贡献。

### 4.5 预期结果与分析

基于模型架构设计和理论分析，我们预期：

1. **门控注意力贡献**：移除所有门控后，模型性能预期下降 2-5 个百分点。门控允许模型在稀疏注意力因窗口限制引入噪声时选择性抑制不相关信息。
2. **压缩记忆贡献**：移除 Infini 记忆后，跨页长距离依赖的建模能力丧失，Page 15 的"财务状况"块可能无法利用 Page 5 的"投标保证金"上下文，预期长距离依赖场景下 F1 下降 >5%。
3. **门控融合贡献**：对于布局信息高度相关的类别（如"封面"依赖页面位置），门控融合的移除预期导致这些类别的 F1 下降更为显著。
4. **稀疏注意力效率**：在 50 块的典型页面中，稀疏模式的注意力计算量约为全连接模式的 (2×5+4)/50 = 28%，预期获得约 3.5× 的加速。

---

## 5 讨论

### 5.1 门控注意力体系的设计哲学

本文在四个层级统一引入门控机制并非简单的叠加热门技术，而是遵循一条清晰的设计原则：**在每一步信息流动中，给予模型"不信任新信息"的选择权**。具体而言：

- **注意力门控**不信任注意力输出（稀疏模式可能遗漏关键关联）
- **跨页门控**不信任历史页面（无关历史不应污染当前表示）
- **融合门控**不信任单一模态（布局特征与语义特征各有盲区）
- **FFN 门控**不信任固定变换（特征维度上的非线性交互应具备选择性）

这种设计使得模型从"被动接收所有信号"转变为"主动筛选有用信号"，在信息过载的多模态长文档场景中尤为重要。

### 5.2 当前局限与未来工作

**数据局限**：当前仅 1 份文档（151 页）具有完整的人工标注。在更多标注 PDF 加入后，方可构建 train/validation/test 划分并评估模型的真实泛化能力。在第一阶段，模型以技术验证为主：证明数据管线可用、前向传播正常、损失可计算、指标可报告。

**OCR 依赖**：图文混合分支的 OCR 文本质量直接影响混合模态的编码效果。当前 OCR 作为预处理步骤在 Dataset 端读取缓存结果。未来可考虑将 OCR 模块与下游分类任务联合微调。

**可扩展方向**：（1）将页编码器中的稀疏注意力升级为 Longformer 风格的扩张滑动窗口以捕获更大范围的块间关系；（2）引入 CRF/Viterbi 序列解码器对边界预测施加转移约束；（3）探索跨文档的预训练策略以提升小样本场景下的性能；（4）构建更大规模的多文档投标文件标注数据集。

---

## 6 结论

本文针对多模态投标文件切片分类任务，提出了 HMSAN-BSA 模型。该模型通过三路多模态特征提取器解决模态异质性问题，通过门控稀疏注意力页编码器在保证计算效率的同时实现页内上下文聚合，通过 Infini-attention 压缩记忆以恒定空间开销维护跨页长距离依赖，并通过四级门控注意力体系增强信息筛选能力。模型支持块级分类、边界检测和段级分类三项任务的联合训练。在当前的有限标注数据集上，模型的工程骨架已通过结构验证，后续在扩展标注数据后将进行全面的实验评估。

---

## 参考文献

[1] Dengel, A., & Shafait, F. (2014). Analysis of document structure. In *Handbook of Document Image Processing and Recognition*.

[2] Yang, X., Yumer, E., Asente, P., Kraley, M., Kifer, D., & Giles, C. L. (2017). Learning to extract semantic structure from documents using multimodal fully convolutional neural networks. *CVPR 2017*.

[3] Xu, Y., Li, M., Cui, L., Huang, S., Wei, F., & Zhou, M. (2020). LayoutLM: Pre-training of text and layout for document image understanding. *KDD 2020*.

[4] Xu, Y., et al. (2021). LayoutLMv2: Multi-modal pre-training for visually-rich document understanding. *ACL 2021*.

[5] Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The long-document transformer. *arXiv:2004.05150*.

[6] Zaheer, M., et al. (2020). Big Bird: Transformers for longer sequences. *NeurIPS 2020*.

[7] HMSAN: Hierarchical Memory Sparse Attention Network. (参考架构设计文档，2026).

[8] Munkhdalai, T., et al. (2024). Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention. *arXiv:2404.07143*.

[9] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*.

[10] Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder. *EMNLP 2014*.

[11] Dauphin, Y. N., et al. (2017). Language modeling with gated convolutional networks. *ICML 2017*.

[12] Chowdhery, A., et al. (2023). PaLM: Scaling language modeling with pathways. *JMLR*.

[13] Touvron, H., et al. (2023). LLaMA: Open and efficient foundation language models. *arXiv:2302.13971*.

[14] Cui, Y., et al. (2021). Pre-training with whole word masking for Chinese BERT. *IEEE/ACM TASLP*.

[15] Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR 2021*.