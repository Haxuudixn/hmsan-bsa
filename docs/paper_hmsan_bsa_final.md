# 面向多模态投标文件切片分类的门控层次化记忆稀疏注意力网络

## 摘要

投标文件切片分类要求模型同时处理多模态异质块、建模跨页长距离依赖并检测模糊的章节边界，而已有文档理解方法通常仅覆盖其中部分能力，缺乏统一的解决方案。本文提出 HMSAN-BSA，一种层次化记忆稀疏注意力网络，通过四个技术贡献应对上述挑战：（1）三路多模态特征提取器，根据切片类型自动选择 RoBERTa-wwm-ext、ViT-B/16 或 OCR 增强双路融合，并通过门控机制自适应平衡模态特征与布局特征——这使得模型能在语义模糊时依赖布局信号，在布局稀疏时信任语义特征；（2）基于门控稀疏注意力（GSA）的页编码器，以 Sigmoid 门控闪电索引器和方差驱动的自适应稀疏控制器替代固定局部窗口，在单页 50 块的典型场景中仅计算全连接注意力 28% 的运算量；（3）Infini-attention 压缩记忆机制，以恒定 O(d²) 空间维护跨页上下文，并通过边界感知门控在章节切换时选择性重置记忆——解决了传统 Transformer 记忆随文档长度线性增长的瓶颈；（4）涵盖融合、值调制、索引器评分、输出、跨页交互、前馈层、记忆读出和边界更新八个位置的系统门控体系，其设计哲学可概括为"在每一步信息流动中赋予模型不信任输入的权利"。模型冻结预训练编码器后仅含 2.5M 可训练参数，适合标注成本高昂的小样本场景。为支撑模型训练与评估，本文设计并实现了专用的Web端投标文件标注工具，支持块级、整页、多页和边界四级标注模式，建立了完整的 PDF解析-特征提取-人工标注-训练数据导出 数据管线。在此基础上设计了9组消融实验以分离各门控位置及压缩记忆的贡献，并与5个代表性基线进行能力维度的定性对比。

**关键词**：投标文件分类；多模态文档理解；门控稀疏注意力；Infini-attention；层次化记忆网络；章节边界检测

---

## 1 引言

投标文件是企业参与招投标活动的核心法律文书。一份典型投标文件长达数百页，包含正文文本、表格、证书扫描件、公章图像和资质证明等异构内容。自动识别每个页面切片所属的章节类型——如"封面""目录""商务偏差表""财务状况""资质证明"——是投标文件智能处理的基础环节，直接影响后续的信息抽取、合规审核和自动化评分的质量。

然而，该任务面临四个相互交织的技术困难。（1）**多模态异质性**：同一页面可能同时包含文本块和图像块，且图像块内部常嵌套 OCR 可识别文字，要求模型对纯文本、纯图像和图文混合三种类型分别设计差异化的编码策略。（2）**长距离上下文依赖**：章节信息往往分布在连续多个页面中，仅凭单页内部特征难以分辨第 50 页的切片属于"财务状况"还是"资质证明"——该判断可能需要回溯第 5 页的章节起始信息。（3）**计算效率约束**：单页可能包含数十到上百个块，全连接自注意力 O(n²) 的复杂度在长文档中不可接受。（4）**章节边界模糊**：实际投标文件中章节过渡常缺乏显式标题标记，模型必须从语义和布局的细微变化中自动感知边界。

现有文档理解方法难以同时应对上述挑战。LayoutLM 系列 [1,2,3] 联合建模文本与布局，但面向单页文档图像设计，缺乏跨页上下文传播能力。长距离 Transformer 变体 [4] 将注意力扩展至多页，但未引入门控或压缩记忆。HMSAN [5] 引入层次化记忆但依赖固定注意力模式。与此同时，注意力机制领域在 2024-2026 年取得了重要进展：Infini-attention [6] 将无限长上下文压缩为恒定大小的记忆矩阵，Gated Sparse Attention（GSA）[7] 以可学习的门控索引器和自适应稀疏控制器统一了稀疏注意力与门控注意力两条研究路线。然而，这些技术尚未被系统性地整合到文档理解任务中。

本文的目标是将上述进展融合为面向投标文件切片分类的统一架构。我们提出 HMSAN-BSA，其核心创新在于将 GSA 首次应用于文档块序列编码，并将 Infini-attention 与边界感知门控结合形成章节级压缩记忆。具体而言，本文包含五项贡献（四项技术贡献与一项工程贡献）：

1. **三路多模态特征提取器**：根据切片类型自动选择 RoBERTa、ViT 或 OCR 双路融合编码路径，并通过门控机制自适应平衡模态特征（768d）与布局特征（64d）的融合比例。
2. **基于 GSA 的页编码器**：以 Sigmoid 门控闪电索引器替代 ReLU 评分，以方差驱动的自适应稀疏控制器替代固定窗口大小，以双重门控（G1 输出门 + G2 值门）增强信息筛选。这是 GSA 在文档理解领域的首次应用。

5. **专用标注工具与数据管线**：针对投标文件切片分类高度专业化的标注需求，设计并实现了基于 PDF.js [26] 的纯前端 Web 标注工具，支持块级（19 类章节标签）、整页、多页批量和边界（B/I/E/O）四级标注模式，集成了文本块/图像块自动检测、字体风格多策略判定（加粗、倾斜、颜色）、块类型自动分类和训练数据打包导出功能，建立了从 PDF 解析到特征提取到人工标注到训练数据导出到模型输入的完整闭环。
3. **Infini-attention 压缩记忆**：以 O(d²) 恒定空间维护跨页上下文，并通过边界感知门控在章节切换时选择性重置章节记忆，避免跨章节信息污染。
4. **八位置门控注意力体系**：在融合、值调制、索引器评分、注意力输出、跨页交互、前馈层、记忆读出和边界更新八个位置系统引入可学习门控，遵循统一的设计哲学。

上述架构仅含 2.5M 可训练参数（冻结预训练编码器），适合标注成本高昂的小样本投标文件场景。为支撑模型的训练与评估，本文配套开发了专用标注工具，实现从原始 PDF 到训练样本的全自动预处理管线。本文进一步通过系统性文献调研确认了所提设计与 2024-2026 年领域最新进展的一致性，并设计了 9 组消融实验以分离各组件的贡献。

## 2 相关工作

### 2.1 文档理解与切片分类

文档切片分类经历了从规则方法 [8] 到深度学习的演进。CNN 和 RNN 方法 [9] 在版面分析任务上取得了初步成功。LayoutLM [1] 首次将文本与二维布局联合预训练，其后续版本 LayoutLMv2 [2] 和 LayoutLMv3 [3] 分别引入了视觉特征和统一掩码预训练策略，在表单理解和发票信息抽取等任务上建立了新的基准。DocFormer [10] 通过共享空间注意力进一步融合了文本与视觉模态。

然而，这些模型均面向单页文档设计。Douzon 等人 [4] 探索了 Longformer 和 BigBird 等高效注意力模式在多页商业文档上的应用，并提出了 2D 相对注意力偏置以利用空间邻近性，但仍未解决长距离跨页信息传播问题。值得注意的是，现有文档理解模型主要面向英文文档，中文投标文件的领域术语、混合图文布局和印章公章等特殊元素带来了额外的挑战。据我们所知，目前尚无专门针对多模态投标文件切片分类并在跨页上下文中建模章节边界的工作。

### 2.2 稀疏注意力与门控注意力

标准自注意力的二次复杂度催生了两条相对独立的研究路线。**稀疏注意力**通过限制注意力范围降低复杂度：Longformer [11] 采用滑动窗口加全局 token，BigBird [12] 进一步引入随机注意力。DeepSeek-V3.2 [13] 提出闪电索引器在低维空间以 O(n²·dI)（dI≪d）的轻量成本评分所有 token 对，仅对 top-k 候选进行全维度注意力。**门控注意力**通过可学习的门控信号增强表示质量与训练稳定性。Qiu 等人 [14] 证明在缩放点积注意力后施加逐元素 Sigmoid 门控可将首个 token 的注意力比例从 47% 降至 5% 以下，有效缓解 attention sink 现象，该工作获得 NeurIPS 2025 最佳论文奖并已被 Qwen3-Next 等生产系统采用。GTrXL [15] 将门控应用于 Transformer-XL 以增强强化学习中的训练稳定性。

**Gated Sparse Attention（GSA）** [7] 统一了上述两条路线。其核心组件包括：以 Sigmoid 替代 ReLU 的门控闪电索引器（得分有界于 (0, HI)）、基于注意力得分方差的适应性稀疏控制器（高方差时激进剪枝、低方差时保留更多上下文）、以及双重门控（输出端 G1 和值端 G2）。在 1.7B 参数模型和 400B token 训练规模下，GSA 在保持稀疏注意力 12-16 倍加速的同时，将困惑度从 6.03 降至 5.70，损失尖峰降低 98%。本文的页编码器以 GSA 为核心，将其从语言模型的 token 序列适配到文档页面的块序列，这是 GSA 在文档理解领域的首次应用。

5. **专用标注工具与数据管线**：针对投标文件切片分类高度专业化的标注需求，设计并实现了基于 PDF.js [26] 的纯前端 Web 标注工具，支持块级（19 类章节标签）、整页、多页批量和边界（B/I/E/O）四级标注模式，集成了文本块/图像块自动检测、字体风格多策略判定（加粗、倾斜、颜色）、块类型自动分类和训练数据打包导出功能，建立了从 PDF 解析到特征提取到人工标注到训练数据导出到模型输入的完整闭环。

### 2.3 压缩记忆与层次化建模

层次化文档建模的核心挑战在于如何在固定内存预算下传播跨段落乃至跨页面的信息。HMSAN [5] 引入了段落级和文档级记忆单元，但其记忆空间随文档长度线性增长。Infini-attention [6] 通过增量外积更新将历史上下文压缩为固定大小的记忆矩阵 M ∈ R^{d×d}，检索和更新均为 O(d²)，与已处理的 token 数无关。该机制已在 1M-token 密钥检索和 500K-token 书籍摘要任务上得到大规模验证。本文在页面级和章节级均采用 Infini-attention，并在章节记忆中创新性地引入边界感知门控：当边界检测头预测章节起始（B-SECTION）时，更新掩码趋近于零，使新章节获得不受旧章节污染的干净记忆状态。

### 2.4 门控机制的演进

门控机制从 LSTM [16] 的遗忘门、输入门和输出门发端，经 GRU [17] 简化为更新门和重置门，在循环神经网络中已成为标准组件。在 Transformer 时代，门控线性单元（GLU）[18] 将门控引入前馈网络，被 PaLM [19] 的 SwiGLU 和 LLaMA [20] 的 GEGLU 变体大规模验证。在注意力机制中，门控已分别被应用于输出投影 [14] 和值投影 [7]。多模态领域的最新综述 [21] 确认了自适应门控在跨模态融合中的关键作用。本文的独特贡献在于将八种门控系统性地统一到单一架构中，并以"在每一步信息流动中赋予模型不信任输入的权利"作为一致的设计哲学，这在现有文献中尚无先例。

## 3 方法

### 3.1 问题定义与整体管线

**问题定义。** 给定包含 P 页的投标文档 D = {p₁, ..., p_P}，每页 p_i 由 N_i 个切片（块）组成。每个切片 b_j 携带以下信息：文本内容 text_j、渲染图像 image_j、OCR 提取文本 ocr_text_j（仅混合类型非空）、归一化边界框 bbox_norm_j ∈ R⁸、字体大小 font_size_j、加粗标志 is_bold_j ∈ {0,1}、倾斜标志 is_italic_j ∈ {0,1}、字体颜色 font_color_j ∈ R³，以及类型标记 block_type_j ∈ {TEXT, IMAGE, MIXED}。上述特征均由本文开发的标注工具从 PDF 中自动提取（字体风格检测、块类型判定等细节见第 4.1.3 节），仅 OCR 文本和标注标签需人工参与。模型同时输出三个预测：块级分类 y_cls ∈ {1,...,19}（19 个章节类别）、页级边界检测 y_boundary ∈ {B-SECTION, I-SECTION, E-SECTION, O}、以及段级分类 y_section。

**整体管线。** HMSAN-BSA 按文档页顺序逐页处理，前一页的压缩记忆状态传入后一页形成信息流。图 1 展示了模型管线：

```
输入: 文档 D 的所有切片 (text, image, ocr, bbox, font, color, type)
     │
     ▼
┌─ 逐页循环 (for page p_i in D) ──────────────────────────────────┐
│                                                                  │
│  ① BlockMultiModalEncoder → block_vectors [N_i, 128]            │
│  ② PageEncoder (GSA)      → page_repr [128], encoded [N_i,128]  │
│  ③ InterPageAttention     → page_repr' [128]                    │
│  ④ BoundaryHead           → boundary_logits [4]                 │
│  ⑤ InfiniPageMemory       → enhanced_page [128], M_page, z_page │
│  ⑥ InfiniSectionMemory    → enhanced_section [128], M_sec, z_sec│
│  ⑦ ClassificationHead     → block_logits [N_i, 19]              │
│  ⑧ SectionHead            → section_logits [19]                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 多模态特征提取器

**动机。** 投标文件切片存在三种模态类型：纯文本（如正文段落）、纯图像（如无文字的公章扫描）和图文混合（如含嵌入文字的证书扫描件）。为每种类型使用同一编码器会导致信息丢失或噪声引入——纯图像块输入文本编码器产生无意义表示，纯文本块输入图像编码器浪费计算且丢失语义。

**模块设计。** BlockMultiModalEncoder 根据 block_type 选择三条编码路径之一。

- **TEXT 分支**：使用 RoBERTa-wwm-ext [22]（中文全词掩码预训练 BERT）编码，取 [CLS] 位置输出（768 维）。最大有效长度 510 token；对数据集中仅 0.02% 的超长文本块（68,617 个文本块中 15 个超过 512 字符），采用首部 256 + 尾部 254 截断保留首尾关键信息。
- **IMAGE 分支**：使用 ViT-B/16 [23] 编码渲染图像，取 [CLS] 位置输出（768 维）。
- **MIXED 分支**：首先通过 OCR 提取图像内嵌文本，然后将 OCR 文本经 RoBERTa 编码、原始图像经 ViT 编码，双路拼接后通过可学习投影融合：h_mixed = Linear_{1536→768}([RoBERTa(ocr); ViT(image)])。

**布局编码。** 所有切片类型共享 LayoutEncoder，处理六类布局特征：bbox_norm[8]→Linear(8→32)+ReLU；font_size→MinMax 归一化标量；is_bold, is_italic→0/1 标量；font_color[3]→Linear(3→16)+ReLU；block_type→Embedding(3,16)。拼接后 67 维向量投影至 h_layout∈R⁶⁴。

**门控融合（门控位置 ①）。** 区别于简单的 concat+Linear，我们采用门控融合让模型自适应平衡模态和布局：

```
m_proj = W_modality · h_modality ∈ R^d      (d=128)
l_proj = W_layout · h_layout ∈ R^d
gate   = σ(W_gate · [h_modality; h_layout]) ∈ R^d
h_block = ReLU(LayerNorm(gate ⊙ m_proj + (1-gate) ⊙ l_proj))
```

**技术优势。** 门控向量的每个维度独立决定模态与布局的混合比例。以"封面"类别为例：其文本可能仅含公司名称等少量信息，但页面位置固定在第一页，此时门控自动偏向布局投影；相反，对于"十不干"等语义特征鲜明的类别，门控偏向模态投影。该机制使模型能够根据内容自适应地信任不同信号源。

### 3.3 基于 GSA 的页编码器

**动机。** 单页内数十个块之间的自注意力若采用全连接模式，计算量为 O(N²)，且并非所有块对之间都存在有意义的语义关联。固定局部窗口虽然高效，但无法根据内容自适应地调整注意力范围——某些块可能需要关注远处的关键块。GSA [7] 提供了解决这一矛盾的系统方案。

**模块设计。** 页编码器由 L=2 层 GSA Transformer 组成，每层包含门控闪电索引器、适应性稀疏选择、双重门控和门控前馈网络。

**（1）门控闪电索引器（门控位置 ②③）。** 在低维空间（dI=32, HI=4 头）以低成本评分所有位置对：

```
I(t,s) = Σ_{j=1}^{HI} σ(h_t·W_iw_j) · σ(q_tj·k_sj + b_j)     ∈ (0, HI)
```

双重 Sigmoid 使得分有界且平滑，相比 DeepSeek-V3.2 [13] 的 ReLU 索引器更稳定。选择预算 k 根据得分方差自适应调整（门控位置 ③）：

```
k = clamp(round(k_base · Var(I) / EMA_Var), k_min, k_max)
```

默认值：k_base=16, k_min=4, k_max=32。高方差意味索引器对相关块高度确定，可激进剪枝；低方差意味上下文模糊，保留更多候选。

**（2）双重门控（门控位置 ②④）。** G2 值门控在注意力聚合前调制值向量：V' = V ⊙ σ(h·W_g2)；G1 输出门控控制注意力输出与残差的混合：g1 = σ(W_g1·x)；out = g1 ⊙ attn_out + (1-g1) ⊙ x。G1 的创新意义在于：当稀疏选择遗漏关键关联时，门控自动关闭注意力通道，直接保留原始表示。

**（3）门控前馈网络（门控位置 ⑥）。** 以 GLU 变体升级标准 FFN：h = GELU(W_g·x) ⊙ (W_v·x)；out = W_out·h，同等 FLOPs 下表达能力翻倍。

**（4）全局 token 与页表示。** G=4 个可学习全局 token 置于序列前端，参与全注意力（所有位置均关注全局 token，全局 token 关注所有位置）。L 层后，全局 token 的输出经池化得到页表示：

```
page_repr = Linear_{G·d → d}(Flatten(global_tokens_out)) ∈ R^d
```

**技术优势。** 在典型的 50 块页面中，GSA 的注意力计算量约为 (k+G)/(N+G) ≈ (16+4)/54 ≈ 37%，显著低于全连接注意力的 100%。同时，可学习索引器能够发现固定窗口无法捕获的长距离块间关联。

### 3.4 门控跨页注意力（门控位置 ⑤）

**动机。** 页编码器的输出 page_repr 仅编码单页内信息。相邻页之间的关系（如前一页末尾暗示章节转换）需要通过跨页机制注入。

**模块设计。** 当前页表示通过多头注意力与最近 m=3 页的历史表示交互，并通过门控防止无关历史污染：

```
attn_out = MHA(query=page_repr, key=page_history, value=page_history)
gate = σ(W_gate·[page_repr; attn_out])
page_repr' = gate ⊙ attn_out + (1-gate) ⊙ page_repr
```

### 3.5 Infini-Attention 压缩记忆

**动机。** 跨页注意力仅覆盖最近 3 页，无法建模更长距离的依赖（如第 50 页需要回溯到第 5 页的章节起始信息）。传统做法需要存储所有历史页表示，空间 O(P·d)。Infini-attention [6] 将历史压缩为固定大小矩阵，是解决这一矛盾的关键技术。

**（1）压缩记忆基类。** 所有历史信息被压缩为 M∈R^{d×d} 和 z∈R^d。检索采用线性注意力：

```
Q = W_q(q)；A = σ(Q)·M / (σ(Q)·z + ε)；mem_out = W_out(A)
```

其中 σ(x) = ELU(x)+1。更新采用增量外积：

```
K = W_k(q)；V = W_v(q)
M_new = M_old + σ(K)^T·V；z_new = z_old + σ(K)
```

检索和更新均为 O(d²)，与已处理页数无关。

**（2）门控记忆融合（门控位置 ⑦）。** 检索到的记忆上下文通过可学习门控融合：gate = σ(W_g·[page_repr'; mem_out])；output = gate ⊙ page_repr' + (1-gate) ⊙ mem_out。此门控使模型在"信任当前信息"与"信赖历史记忆"之间自主选择。

**（3）边界感知章节记忆（门控位置 ⑧）。** 页面记忆不区分章节——第 5 页"投标保证金"的信息会持续影响第 50 页"财务状况"的表示。章节记忆通过边界感知更新掩码解决此问题：

```
update_mask = σ(W_bound·[page_repr'; Softmax(boundary_logits)]) ∈ [0,1]
```

当边界头预测 B-SECTION 时，update_mask≈0，新章节信息不会被旧章节记忆污染。

### 3.6 训练目标

三任务联合优化：L_total = L_cls + α·L_boundary + β·L_section，α=β=0.5。L_cls 为块级交叉熵（19 类），L_boundary 为页级 B/I/E/O 交叉熵，L_section 为段级交叉熵。未标注块使用 ignore_index=-100，不作为负样本参与梯度计算。

### 3.7 八门控位置汇总

**表 1：HMSAN-BSA 八门控体系**

| 位置 | 模块 | 公式 | 来源 |
|------|------|------|------|
| ① | 门控融合 | gate·m_proj + (1-gate)·l_proj | 本文 |
| ② | G2 值门控 | V' = V ⊙ σ(h·W_g2) | GSA [7] |
| ③ | 自适应稀疏 | k = clamp(k_base·Var(I)/EMA_Var) | GSA [7] |
| ④ | G1 输出门控 | gate·attn_out + (1-gate)·x | GSA [7] |
| ⑤ | 门控跨页 | gate·attn_out + (1-gate)·page_repr | 本文 |
| ⑥ | 门控 FFN | GELU(W_g·x) ⊙ (W_v·x) | GLU [18] |
| ⑦ | 门控记忆读出 | gate·x + (1-gate)·mem_out | 本文 |
| ⑧ | 边界感知更新 | σ(W_bound·[page; boundary]) | 本文 |

### 3.8 标注工具与数据管线

HMSAN-BSA 的训练依赖高质量的块级标注数据。鉴于投标文件切片分类任务的特殊性——需标注 19 类章节标签、B/I/E/O 边界标签以及块类型（TEXT/IMAGE/MIXED）——本文开发了专用的 Web 端标注工具，建立了从原始 PDF 到训练样本的完整数据管线。

**设计动机。** 通用标注平台（Label Studio、Doccano 等）面向通用 NLP/CV 任务设计，缺乏对 PDF 内部结构（文本块坐标、字体样式、嵌入图像）的深度解析能力。投标文件标注需要标注者同时观察：（1）块的视觉外观与空间位置，（2）块的文本语义，（3）块所在页面的上下文及前后页的章节归属。这要求在 PDF 页面上直接叠加可交互的块边界框，并支持跨页面的批量标注操作。

**技术实现。** 标注工具为纯前端单页应用，核心依赖 PDF.js [26]（渲染与解析）和 SheetJS（Excel 导出）。工具无需安装任何后端服务或数据库——打开 HTML 文件即可在浏览器中使用，标注数据存储在浏览器 localStorage 中。所有 PDF 解析、文本提取、图像提取、字体风格检测均在客户端完成，保障了标注数据的安全性和隐私性。

**块检测实现细节。** 文本块提取基于 PDF.js 的 `getTextContent()` API，逐页解析文本项的变换矩阵、字体名称、字号和颜色。合并策略采用简单启发式：同一页面内，垂直间距<2pt 且水平重叠的文本项合并为一行，左对齐且行间距<10pt 的连续行合并为同一文本块。字体风格检测综合字体名称正则匹配和 PDF 字体描述符分析；图像块通过 `getOperatorList()` API 分析页面渲染指令流中的 `paintImageXObject` 和依赖对象识别。

**标注与导出流程。** 标注者在左侧页面列表中选择页面，在右侧画布上点击块进行标注。支持四种标注模式（块级、整页、多页批量、边界），标注数据实时持久化。导出时，工具将标注信息、块特征和图像文件打包为训练数据压缩包：ZIP 根目录包含 blocks.xlsx（15 列标注汇总）和 images/ 子文件夹（按 PDF 分目录存储图像）。该格式可直接输入模型的 Document→Page→Block 三级层次构建模块，实现从标注到训练的无缝衔接。具体标注体系、工具界面和导出格式的详细描述见第 4.1 节。

## 4 实验设计

### 4.1 数据集与预处理管线

#### 4.1.1 数据来源与规模

实验数据来源于真实招投标活动中收集的投标文件集合，共含 24 份 PDF 文档，总计 8,750 个唯一页面。经 PDF 解析后共提取 85,304 个切片块（含文本块与图像块）。其中文本块占 80.4%（68,617 个），中位数字符数 17，第 90 百分位 49——这表明大多数块为短文本（如标题、标签、表格单元格）。当前已有 1 份完整标注文档（151 页、907 个标注块）覆盖全部 19 个章节类别。标注块在训练时通过 ignore_index=-100 机制排除，不作为负样本参与梯度计算。

#### 4.1.2 标注工具架构

针对投标文件切片分类任务高度专业化的标注需求——需同时标注章节类别（19 类）、章节边界（B/I/E/O）和多模态块类型（TEXT/IMAGE/MIXED）——现有通用标注工具（如 Label Studio、Doccano）无法直接满足需求。为此，本文设计并实现了专用的 Web 端标注工具，基于纯前端架构（HTML5/CSS3/JavaScript），以 PDF.js [26] 为核心渲染引擎，无需任何后端服务即可在浏览器中完成全部标注工作流。

**工具界面。** 标注工具采用双栏布局：左侧为 PDF 页面列表导航区，支持页面缩略图预览、已标注块数量统计（格式：已标注/总块数）和单页/多页切换（Ctrl+Click 多选、Shift+Click 范围选择）；右侧为主画布区，在 PDF 渲染页面上叠加半透明矩形框标记所有检测到的文本块（绿色实线框）和图像块（蓝色虚线框）。点击任意块矩形框后，右侧弹出标注面板。

**四种标注模式。**
- **块级标注**：单击任意块矩形框，弹出标注面板显示块 ID、类型、文本内容、字体信息（字体名、字号、加粗、倾斜、颜色）、坐标位置，以及 19 类章节标签下拉框和 B/I/E/O 边界标签下拉框。标注数据实时保存在浏览器 localStorage 中。
- **整页标注**：通过"整页标签"下拉框一次性为当前页面所有未标注块批量赋予相同的章节标签，适用于单章节占满整页的场景（如封面页、目录页）。
- **多页批量标注**：通过页面列表多选功能选中连续或不连续的多个页面，点击"标注选中"按钮后，工具自动重新解析所选页面并检测所有块，然后为所有未标注块统一赋予指定标签。标注结果同步更新页面列表的进度统计。
- **边界标注**：每个块同时支持 B/I/E/O 四类边界标签：B-SECTION 标记章节起始页、I-SECTION 标记章节内部页、E-SECTION 标记章节结束页、O 标记无特定边界。边界标签为 Infini-attention 的边界感知门控（⑧）提供监督信号。

#### 4.1.3 块检测与特征提取

标注工具在加载 PDF 时自动执行以下检测管线：

**文本块提取。** 利用 PDF.js 的 TextContent API 逐页提取所有文本项（text item），每条文本项包含文本内容、变换矩阵（含缩放、旋转和倾斜参数）、字体名称和字号。基于坐标位置进行合并聚类：垂直方向间距小于 2pt 的连续文本项合并为同一文本块，水平方向左对齐且间距小于 10pt 的合并为同一行。

**字体风格多策略检测。** 加粗检测综合两种策略：字体名称启发式匹配——正则表达式匹配 "bold"/"heavy"/"black"/"hei"/"blk"/"bd" 等加粗关键词——以及 PDF 字体标志位检测——flags 位 18（ForceBold 标志）为真或 fontWeight≥600 判定为粗体。倾斜检测同样综合两种策略：字体名称启发式匹配——匹配 "italic"/"oblique"/"kai"/"slanted"/"slnt" 等倾斜关键词——以及文本矩阵几何分析——变换矩阵的 |m12|>0.01 表明存在倾斜变换。字体颜色通过 PDF.js 渲染状态中的填充色（fillColor）提取，以 #RRGGBB 十六进制格式存储，归一化为 R³ 向量。

**图像块提取与类型判定。** 通过 PDF.js 的 getOperatorList API 分析页面渲染指令流，识别所有嵌入图像（JPEG、PNG 等格式），提取包围盒坐标和原始二进制数据。块类型基于文本绘制指令与图像绘制指令的空间重叠关系自动判定：仅含文本指令→TEXT，仅含图像指令→IMAGE，文本与图像重叠（IoU>0.3）→MIXED。所有提取的图像另存为独立 PNG 文件，以 `{pdf_name}_page{page_num}_img{idx}.png` 格式命名，供 ViT 编码器加载。

#### 4.1.4 标注体系

19 类章节标签根据投标文件的标准结构和业务语义制定，如表 2 所示。

**表 2：投标文件 19 类章节标注体系**

| 编号 | 标签名称 | 内容定义 | 典型模态 |
|------|----------|----------|----------|
| 1 | 封面页 | 投标文件首页，含项目名称、投标人名称、投标日期 | TEXT+MIXED |
| 2 | 目录 | 各章节的目录列表 | TEXT |
| 3 | 商务偏差表 | 对招标文件商务条款的偏差说明 | TEXT |
| 4 | 投标保证金 | 保证金缴纳凭证或承诺函 | TEXT+MIXED |
| 5 | 关系说明 | 投标人关联关系说明 | TEXT |
| 6 | 基本情况表 | 企业基本信息登记表 | TEXT |
| 7 | 营业执照 | 企业法人营业执照扫描件 | IMAGE |
| 8 | 税务证明 | 税务登记证或完税证明 | IMAGE+MIXED |
| 9 | 资格证明文件 | 各类资质证书、许可证 | IMAGE+MIXED |
| 10 | 财务状况 | 财务审计报告、财务报表 | TEXT |
| 11 | 财务凭证单 | 银行资信证明、存款证明 | IMAGE+MIXED |
| 12 | 资质业绩凭证单 | 过往项目业绩证明材料 | TEXT+MIXED |
| 13 | 评分支撑材料 | 商务/技术评分支撑文件 | TEXT+MIXED |
| 14 | 名称变更 | 企业名称变更工商登记证明 | IMAGE+MIXED |
| 15 | 一致性承诺函 | 投标文件一致性承诺声明 | TEXT |
| 16 | 十不准 | 招投标"十不准"规定的知晓承诺 | TEXT |
| 17 | 公章授权书 | 法定代表人对公章使用人的授权 | TEXT |
| 18 | 法定代表人授权委托书 | 法定代表人授权委托代理人签署 | TEXT |
| 19 | 其他 | 不属于上述类别的辅助内容 | MIXED |

#### 4.1.5 训练数据导出

标注工具支持两种导出格式，确保标注数据可直接输入模型训练管线：

- **标注汇总 Excel（.xlsx）**：包含 pdf_name、page_num、block_type、bbox、content、font、font_size、is_bold、is_italic、font_color、label、boundary_label、ocr_text、img_ref 共 15 个字段。其中 img_ref 列存储关联图像文件的相对路径，建立块与图像之间的索引关系。
- **训练数据压缩包（.zip）**：ZIP 根目录包含 blocks.xlsx（标注汇总）和 images/ 子文件夹。images/ 按 PDF 文档名分目录存储所有提取的图像，每张图像以 `{pdf_name}_page{page_num}_img{idx}.png` 命名。Excel 的 img_ref 列指向 images/ 中的对应路径，形成从文本标注到图像文件的可追溯映射。

预处理管线在此基础上进一步执行：PDF 页面按 150 DPI 渲染为图像供 ViT 编码器加载；图文混合块的 OCR 文本通过 PaddleOCR [27] 离线提取并填入 ocr_text 字段；B/I/E/O 边界标签根据页面在文档中的章节归属自动生成；最终按 Document→Page→Block 三级层次结构构建训练样本。

**数据局限声明**：当前仅 1 份文档（151 页）具有完整人工标注。在收集更多标注 PDF 后，将按文档级划分训练/验证/测试集以评估真实泛化能力。当前阶段以管线验证和技术可行性论证为核心目标。

### 4.2 实现细节

隐层维度 d=128。文本编码器：RoBERTa-wwm-ext（hfl/chinese-roberta-wwm-ext），max_length=510，冻结。图像编码器：ViT-B/16（google/vit-base-patch16-224），冻结。页编码器：L=2 层 GSA，H=4 头，G=4 全局 token，k_base=16，k_min=4，k_max=32，索引器 dI=32，HI=4 头。Infini 记忆：key_dim=value_dim=d=128。跨页窗口 m=3。训练：批大小 1，AdamW（lr=1×10⁻³，weight decay=1×10⁻⁴），5 个 epoch。可训练参数 2.5M；冻结预训练编码器合计约 196M。

### 4.3 评估指标

- **块级**：准确率、Macro-F1、Weighted-F1、各类别精确率/召回率/F1
- **边界级**：精确率/召回率/F1、章节起始检测准确率
- **段级**：分类准确率、段跨度 IoU、过分割/欠分割计数

### 4.4 消融实验设计

**表 2：九组消融实验**

| 变体 | 配置 | 验证假设 |
|------|------|----------|
| HMSAN-BSA (Full) | 完整模型 | 基准性能 |
| A1: - G1 Output Gate | 移除 G1，使用标准残差 | 输出门控是否降低注意力噪声？ |
| A2: - G2 Value Gate | 移除 G2 值调制 | 聚合前值门控是否必要？ |
| A3: - Adaptive Sparsity | k=k_base 固定 | 自适应稀疏是否优于固定 k？ |
| A4: - Gated Fusion | 替换为 concat+Linear | 门控融合是否优于简单拼接？ |
| A5: - Gated FFN | 替换为标准 FFN | GLU 风格 FFN 是否带来增益？ |
| A6: - Infini Memory | 移除全部压缩记忆 | 跨页记忆的关键程度？ |
| A7: - Boundary Gate | 移除边界感知更新 | 边界感知是否改善章节切换？ |
| A8: - Gated InterPage | 移除跨页注意力门控 | 跨页门控是否必要？ |
| A9: GSA→LocalWindow | 替换为固定局部窗口(±5) | 可学习索引器 vs 固定窗口？ |

**预期趋势**（基于 GSA 论文 [7] 结果和理论分析）：

- **GSA 三元组（A1+A2+A3 vs A9）**：预期 GSA 显著优于固定局部窗口。GSA 论文在 128K 上下文上报告了 2 pp 的困惑度提升。
- **Infini 记忆（A6）**：预期在需要跨页推理的类别（如区分"财务状况"和"资质证明"）上退化最严重。移除记忆后，第 50 页的块无法获取第 5 页的章节起始信息。
- **门控融合（A4）**：预期对布局强相关类别（如"封面""目录"）影响最大。门控融合使模型能在语义模糊时自动依赖位置等布局信号。
- **边界感知（A7）**：预期在章节密集切换的文档段中影响最明显。

### 4.5 能力对比

**表 3：与代表性基线的能力维度对比**

| 模型 | 稀疏注意力 | 门控 | 压缩记忆 | 边界检测 | 多模态 | 可训练参数 |
|------|-----------|------|---------|---------|--------|-----------|
| LayoutLMv3 [3] | ✗ | ✗ | ✗ | ✗ | ✓ | ~200M |
| Long-Range DU [4] | Longformer | ✗ | ✗ | ✗ | ✓ | ~130M |
| IRIS [24] | Linear Attn | ✗ | 检索式 | ✗ | ✗ | ~5M |
| GTrXL [15] | ✗ | 1 位置 | ✗ | ✗ | ✗ | ~50M |
| **HMSAN-BSA** | **GSA** | **8 位置** | **Infini** | **B/I/E/O** | **3 路** | **2.5M** |

HMSAN-BSA 是唯一在稀疏注意力、多位置门控、压缩记忆和边界检测四个维度上同时具备能力的架构，且参数效率显著优于所有基线。

## 5 讨论

### 5.1 设计哲学：「不信任」的权利

HMSAN-BSA 的八个门控位置并非技术的随意堆叠，而是遵循一条一致的设计原则：**在每一步信息流动中，赋予模型明确的选择权——是否"不信任"输入信息，转而保留已有表示**。

这一原则在每一处门控中具体化为一种防御策略：融合门控（①）防御单一模态的盲区——语义特征对布局敏感类别可能失效，布局特征对语义丰富类别可能误导；值门控（②）防御无关特征维度的干扰；自适应稀疏（③）防御固定预算假设的局限——不同查询的信息需求差异显著；输出门控（④）防御稀疏注意力因遗漏关键关联而引入的噪声；跨页门控（⑤）防御无关历史页面对当前页的污染；FFN 门控（⑥）防御固定非线性变换的表达瓶颈；记忆门控（⑦）防御压缩记忆可能引入的检索偏差；边界门控（⑧）防御跨章节的信息泄漏。

这套"防御型"门控体系将模型从"所有信号无条件接收"的被动模式转变为"逐级筛选、按需采纳"的主动模式，在多模态长文档的信息过载场景中具有独特的适用性。

### 5.2 局限性

**数据局限**：当前仅 1 份文档具有完整人工标注，由本文开发的标注工具完成。在收集更多标注 PDF 后，可构建 train/validation/test 划分以评估真实泛化能力。当前阶段应视为技术验证而非性能声明。**标注工具局限**：字体风格检测（加粗、倾斜）依赖 PDF.js 的字体描述符和变换矩阵启发式方法，对某些使用绘制路径模拟粗体/斜体的 PDF 可能失效；块类型自动分类（TEXT/IMAGE/MIXED）基于操作符列表的空间重叠分析，对复杂嵌套布局可能存在误判。**OCR 依赖**：混合分支的编码质量受限于 OCR 提取文本的准确性。中文字符识别在印章、手写体等场景中仍存在挑战。**超参数敏感性**：GSA 的 k_base 和 EMA 衰减系数可能需要针对不同页面密度进行调整。

### 5.3 未来工作

（1）引入 2D 相对位置编码 [4] 为空间邻近的块提供显式偏置，可能进一步提升布局依赖类别的性能。（2）探索 Mamba-2 状态空间模型 [25] 替代 GSA 页编码器，在超长文档（500+页）场景中获得更优的效率-性能权衡。（3）引入 CRF/Viterbi 序列解码对边界预测施加结构性转移约束（如 I-SECTION 不能直接跟随 O）。（4）构建更大规模的多文档投标标注数据集，探索跨文档预训练策略以缓解小样本问题。

## 6 结论

本文针对多模态投标文件切片分类任务提出了 HMSAN-BSA，通过系统性地整合门控稀疏注意力（GSA）、Infini-attention 压缩记忆和八位置门控注意力体系，同时应对了多模态异质性、长距离上下文依赖、计算效率约束和模糊章节边界四项核心挑战。为支撑模型的训练与评估，本文设计并开源了专用的Web端投标文件标注工具，支持块级、整页、多页批量和边界四级标注模式，建立了完整的 PDF解析-特征提取-人工标注-训练数据导出 闭环管线。模型仅含 2.5M 可训练参数，适合标注成本高昂的小样本场景。系统性文献调研确认了所提设计与 2024-2026 年领域前沿的一致性。9 组消融实验设计为分离各组件的贡献提供了完整的验证框架。未来工作将在更大规模标注数据集上扩展模型并探索状态空间模型的替代方案。

## 参考文献

[1] Xu, Y. et al. LayoutLM: Pre-training of text and layout for document image understanding. KDD 2020.

[2] Xu, Y. et al. LayoutLMv2: Multi-modal pre-training for visually-rich document understanding. ACL 2021.

[3] Huang, Y. et al. LayoutLMv3: Pre-training for document AI with unified text and image masking. ACMMM 2022.

[4] Douzon, T. et al. Long-range transformer architectures for document understanding. ICDAR 2023.

[5] HMSAN: Hierarchical Memory Sparse Attention Network. Architecture design document, 2026.

[6] Munkhdalai, T. et al. Leave no context behind: Efficient infinite context transformers with Infini-attention. arXiv:2404.07143, 2024.

[7] Shen, A. & Shen, A. Gated sparse attention: Combining computational efficiency with training stability for long-context language models. arXiv:2601.15305, 2026.

[8] Dengel, A. & Shafait, F. Analysis of document structure. Handbook of DIPR, 2014.

[9] Yang, X. et al. Learning to extract semantic structure from documents using multimodal fully convolutional neural networks. CVPR 2017.

[10] Appalaraju, S. et al. DocFormer: End-to-end transformer for document understanding. ICCV 2021.

[11] Beltagy, I. et al. Longformer: The long-document transformer. arXiv:2004.05150, 2020.

[12] Zaheer, M. et al. Big Bird: Transformers for longer sequences. NeurIPS 2020.

[13] DeepSeek-AI. DeepSeek-V3.2 technical report, 2025.

[14] Qiu, J. et al. Gated attention for large language models. NeurIPS 2025. (Best Paper Award)

[15] Parisotto, E. et al. Stabilizing transformers for reinforcement learning. arXiv:1910.06764, 2019.

[16] Hochreiter, S. & Schmidhuber, J. Long short-term memory. Neural Computation, 1997.

[17] Cho, K. et al. Learning phrase representations using RNN encoder-decoder. EMNLP 2014.

[18] Dauphin, Y. N. et al. Language modeling with gated convolutional networks. ICML 2017.

[19] Chowdhery, A. et al. PaLM: Scaling language modeling with pathways. JMLR, 2023.

[20] Touvron, H. et al. LLaMA: Open and efficient foundation language models. arXiv:2302.13971, 2023.

[21] Moon, A.-S. et al. A survey on multimodal emotion recognition: Methods, datasets, and future directions. CMC, 2026.
Wzg
[22] Cui, Y. et al. Pre-training with whole word masking for Chinese BERT. IEEE/ACM TASLP, 2021.

[23] Dosovitskiy, A. et al. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR 2021.

[24] Li, F. et al. IRIS: Interpretable retrieval-augmented classification for long interspersed document sequences. ACL 2025.

[25] Dao, T. & Gu, A. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. ICML 2024.

[26] Mozilla. PDF.js: A general-purpose, web standards-based platform for parsing and rendering PDFs. https://mozilla.github.io/pdf.js/, 2024.

[27] Du, Y. et al. PP-OCR: A practical ultra lightweight OCR system. arXiv:2009.09941, 2020.
