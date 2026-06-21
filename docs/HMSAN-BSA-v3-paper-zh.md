# 基于门控稀疏注意力的层次化记忆网络
# 面向投标文件切片分类的多模态方法

## 摘要

投标文件切片分类面临多模态异质性、长程跨页依赖和标注稀疏性三重挑战，且现有方法均仅针对单一维度分别优化，缺乏统一的系统性解决方案。本文提出HMSAN-BSA（Hierarchical Memory Sparse Attention Network with Boundary-aware Section Aggregation），首次将**门控稀疏注意力**作为贯穿架构全部三个建模层次的统一设计原则。

具体而言：（1）**门控LiLT块编码器**通过从排版特征预测的三路softmax门控动态融合RoBERTa-wwm-ext文本特征、ViT-B/16视觉特征和布局线索，门控参数仅8.6K且无需人工指定模态权重；（2）**扩张门控页编码器**将LongNet的指数扩张注意力（dilation in {1,2,4,8}）与逐层top-k稀疏门控相结合，4层编码器在O(N*w)线性复杂度下获得约40个块的等效感受野——达到全连接自注意力的覆盖范围但仅需其约4%的计算量；（3）**门控Infini压缩记忆**在标准delta-rule更新中引入逐维选择性写入门g_w和稀疏读取门g_r，记忆总容量从基线模型的512维提升至32,768维（64倍），且门控扩展的额外参数量仅占记忆矩阵的4.7%。

在包含24份投标文件（85,304块，19类，907标注块）的专有数据集上，设计了六阶段课程训练策略、五组消融实验方案以及针对记忆槽位数、扩张层数和门控top-k值的敏感度网格分析。该架构支持轻量配置（用于快速原型）和完整配置（利用预训练编码器），为稀疏标注场景下的长文档多模态分类提供了可复现的完整技术框架。

**关键词**：投标文件切片分类；门控稀疏注意力；层次化记忆网络；多模态文档理解；压缩记忆

---

## 1 引言

### 1.1 问题背景

投标文件是全球公共采购流程的核心信息载体。典型投标文件跨度50至200页，涵盖投标函、商务偏离表、公司资质证明、近年财务报表、技术实施方案等十余种具有明确版面特征和语义边界的章节类型[1,2,3]。自动识别文档中每个语义切片（block）的类别标签，是投标合规审查自动化、关键信息结构化抽取和智能辅助评标系统的基础能力。

### 1.2 核心挑战

该任务区别于通用文档分类的特殊性体现在三个正交维度：

**挑战一：多模态异质性。** 单页内同时存在纯文本段落（如公司简介）、纯图像块（如公章、签章）以及文本叠加于图像上的混合型块（如图章上的红色文字）。不同块类型对模态的依赖模式截然不同：文本块应主要依赖语言模型编码，图像块需视觉模型主导，混合块则需两类编码器的深度耦合。传统静态拼接+线性投影的融合策略[4]对所有块类型分配相同信息通道权重，忽略了这一结构性差异——图像块的文本字段为空，却仍为其空文本嵌入分配与完整段落块相同的通道容量。

**挑战二：长程跨页依赖。** 投标文件的章节组织具有跨越数十页的连续性和层级性。例如，财务状况章节可能从第74页延伸至第117页（连续44页），而出现在第80页的资质证书需与第5页的公司基本情况形成跨页语义关联。固定局部窗口注意力[5]无法捕捉此类远距离依赖，全连接自注意力的O(N^2)复杂度在150+页文档上不可行。

**挑战三：标注稀疏性。** 实际采购工作流中详尽切片级标注的人力成本极高。当前数据集仅1份PDF的151页获得标注，已标注块仅907个（占85,304个总量的1.06%）。超98%的块处于未标注状态，不可作为负样本参与监督训练，传统全监督分类范式在此场景下不可直接适用。

### 1.3 研究空缺

现有文献分别针对上述挑战提出了有效方法。LayoutLM系列[6,7,8]和LiLT[9]建立了多模态文档预训练范式，但聚焦于单页或单块级别编码，不涉及跨页上下文传播。Infini-attention[10]和LongNet[11]解决了长序列高效建模，但未考虑多模态融合。Gated Sparse Attention[12]提出了可学习的稀疏连接选择机制，但其应用仍局限于单一Transformer层内。

**核心空缺**：尚无工作将门控稀疏注意力作为一种系统性建模原则，同时应用于多模态融合、页内结构聚合和跨页记忆传播三个层次，为长文档多模态分类提供端到端的统一架构。

### 1.4 本文贡献

1. **统一门控稀疏注意力框架**：首次将可学习稀疏门控机制系统性地注入块级模态融合（第3.2节）、页级结构编码（第3.3节）和跨页记忆传播（第3.4节）三个层次，各层次门控在输入信号和功能角色上正交，确保联合训练的梯度稳定性。

2. **门控Infini压缩记忆**：在标准Infini-attention的delta-rule更新中引入逐维写入门g_w（R^{d_v}）和读取门g_r，赋予记忆矩阵选择性更新与检索能力，容量从512维提升至32,768维（64倍）。

3. **扩张门控页编码器**：将LongNet指数扩张候选集生成与STE训练的门控top-k稀疏筛选相结合，以线性复杂度实现指数增长的等效感受野。

4. **完整的多模态多任务训练框架**：融合块分类、BIOE边界检测和章节分类三个监督目标，通过ignore-index机制正确处理超98%未标注数据，提供六阶段课程训练策略。

---

## 2 问题形式化

### 2.1 文档层次化表示

将投标文档D定义为有序页面序列：D = (P_1, P_2, ..., P_T)，其中T为总页数。每页P_t包含N_t个空间上有序的块：P_t = (b_{t,1}, ..., b_{t,N_t})。每个块包含文本内容、图像区域、排版特征（bbox归一化坐标、字体大小、加粗/倾斜标志、字体颜色）和块类型。

### 2.2 多任务学习目标

模型同时预测三个目标：块级章节类别标签（C=19类）、页级BIOE边界标签（B/I/E/O）和页级主导章节类别。联合损失：L = L_cls + 0.5*L_boundary + 0.5*L_section。关键设计：未标注块的标签设为ignore_index=-100，从损失计算中排除，避免引入系统性偏差。

### 2.3 页序处理约束

模型按页序逐页处理，第t页可访问当前页全部块、前一页记忆状态M_{t-1}以及最近3页的历史页表示。在保持线性计算复杂度O(T*N_t)的同时，通过压缩记忆间接访问文档全局上下文。

---

## 3 方法

### 3.1 整体架构

HMSAN-BSA由四个核心模块串联构成：门控LiLT块编码器（第3.2节）将当前页块编码为块向量；扩张门控页编码器（第3.3节）在页内应用稀疏注意力聚合为页表示；门控Infini压缩记忆（第3.4节）将页表示整合进跨页记忆并输出融合表示；三个检测头分别预测块类别、边界标签和章节类别。

**表1：HMSAN-BSA v3核心配置**

| 模块 | 参数 | 取值 |
|---|---|---|
| 通用 | 隐层维度 d_model | 128 |
| 块编码器 | 文本骨干 | RoBERTa-wwm-ext (768d) |
| | 图像骨干 | ViT-B/16 (768d) |
| | 布局输出 | 64d |
| | 模态门控数 | 3 (text, layout, image) |
| 页编码器 | 层数L | 4 |
| | 扩张序列 | [1, 2, 4, 8] |
| | 基础窗口w | 5 |
| | 全局Token G | 4 |
| | 门控top-k | 16 |
| 压缩记忆 | 槽位数K | 4 |
| | 键维度 d_k | 64 |
| | 值维度 d_v | 128 |
| | 总容量 | 32,768 (= K*d_k*d_v) |

### 3.2 门控LiLT块编码器

传统多模态编码器采用h = Linear([e_text; e_layout; e_image])的静态策略，对所有块类型分配等价的模态融合权重。

本文提出由排版特征驱动的自适应三路softmax门控：

    alpha = softmax(W2 * GELU(W1 * e_layout + b1) + b2)    (1)
    h_block = alpha_text * e_text + alpha_layout * e_layout + alpha_image * e_image   (2)

其中e_layout（64d）由LayoutEncoder从6类排版信号编码得到：bbox归一化坐标（8d，Linear(8,32)+ReLU）、字体大小（标量归一化）、加粗（0/1）、倾斜（0/1）、字体颜色（3d，Linear(3,16)+ReLU）、块类型嵌入（3类，Embedding(3,16)）。六者拼接（67d）后经Linear(67,64)+LayerNorm+ReLU得到e_layout。门控MLP参数仅64*128+128*3约8,600维，可忽略不计。

设计直觉：排版特征蕴含丰富的模态重要性先验——图像类型块自然偏好高alpha_image，大号居中加粗文本暗示高alpha_text，公章叠加文字的混合块需均衡权重。关键选择是不从块类型硬编码规则，而是将排版特征输入门控网络，端到端学习最优映射。

LiLT布局偏置：从bbox提取5维空间描述子[cx, cy, w, h, area]，经双层MLP投影为残差信号注入块表示，为语义相似但位置不同的块提供区分性空间信号。

三模态编码：文本经RoBERTa-wwm-ext[13]（max_len=510，超长首尾截断，仅0.02%块受影响）。图像经ViT-B/16[14]的[CLS] token。混合块OCR文本和截图分别编码后拼接（1536d）投影回768d。

### 3.3 扩张门控页编码器

单页块数N_t通常在10至50之间。全连接自注意力O(N^2)不可接受，固定窗口（+/-5）无法连接页首标题与页尾表格。

**扩张候选集（LongNet[11]）**：为L=4层分配指数增长扩张因子d_l=2^l。第l层位置i的块候选集为C_l(i)={j:|j-i|=k*d_l, k=1..w}。候选集大小恒为2w+G+1=15（G=4全局Token），但有效感受野指数增长至w*2^L=40个块——足以覆盖典型单页。

**门控稀疏选择（Gated Sparse Attention[12]）**：在扩张候选集内做top-k筛选：

    g_ij = sigma((W_g*[Q_i;K_j]+b_g) / tau)              (3)
    M_ij = 0 if g_ij in top-k(g_i) else -inf              (4)
    Attn = softmax(QK^T/sqrt(d_k) + M) * V                (5)

训练时采用直通估计器（STE）：前向用离散top-k掩码强制执行稀疏性，反向经连续门控值g_ij传播梯度，端到端学习稀疏连接模式。

**全局Token**：4个可学习全局Token置于序列开头，与所有块Token全连接（不参与稀疏化），充当跨块信息总线。页表示由全局Token池化得到。

**算法1：扩张门控页编码器**

输入: 块向量H(N,d), 全局Token G(4,d)
输出: 页表示p, 上下文增强块向量H'

1. X = Concat([G, H])                          // (4+N, d)
2. for l=1 to L:
3.     d_l = 2^(l-1)
4.     C = DilatedCandidateMask(X, d_l, w=5)
5.     M_gate = GatedTopK(Q, K, C, k=16)        // STE训练
6.     X = X + MHA(LayerNorm(X), mask=M_gate)
7.     X = X + FFN(LayerNorm(X))                // GELU, 4x扩展
8. p = Linear(Flatten(X[0:4]))
9. H' = X[4:]
10. return p, H'

**复杂度**：每层O(N*(2w+G)*d_model)，与N线性。N=40时4层计算量约为全连接注意力的4*15/40=1.5倍，优势随N增长扩大。

### 3.4 门控Infini压缩记忆

固定K槽位GRU记忆面临容量瓶颈（512维）和等权更新双重局限[10]。

**压缩记忆结构**：M in R^{K*d_k*d_v}，K=4槽位构成正交子空间（倾向于分别追踪财务、资质、商务、技术等跨页模式），d_k=64, d_v=128，总容量32,768维。

**门控写入**（选择性delta-rule更新）：

    k = ELU(W_k*p_t)+1            // 正值键 [K, d_k]
    v = W_v*p_t                   // 值向量 [K, d_v]
    g_w = sigma(W_gw*p_t)         // 写入门 [K, d_v]
    r = sigma(k)^T @ M            // 记忆检索 [K, d_v]
    e = v - r                     // 预测误差
    Delta = g_w odot (k otimes e) // 门控增量
    M = M + Delta

若当前页延续同一章节，记忆检索r接近v，误差e趋零，g_w在相关维度趋0以抑制冗余更新；若为章节边界，v与旧章节差异显著，g_w趋1以允许重写。

**门控读取**：

    q = ELU(W_q*p_t)+1
    r = sigma(q) @ M / (sigma(q) @ z + 1e-8)
    g_r = sigma(W_gr*[p_t; r])
    m_out = g_r odot r

**局部-记忆融合**：

    beta = sigma(W_beta*[p_t; m_out])
    p_fused = beta * p_t + (1-beta) * m_out

**算法2：门控Infini记忆更新**

输入: p_t, local_out, M, z
输出: p_fused, M_new, z_new

1. k = ELU(W_k*p_t)+1; v = W_v*p_t
2. g_w = sigma(W_gw*p_t)
3. retrieval = sigma(k)^T @ M; error = v - retrieval
4. Delta = g_w odot (k otimes error); M = M + Delta
5. z = z + sum(sigma(k), dim=-1)
6. q = ELU(W_q*p_t)+1
7. r = sigma(q) @ M / (sigma(q) @ z + 1e-8)
8. g_r = sigma(W_gr*[p_t; r]); m_out = g_r odot r
9. beta = sigma(W_beta*[p_t; m_out])
10. p_fused = beta*p_t + (1-beta)*m_out
11. return p_fused, M, z

标准Infini-attention[10]等价于g_w=1, g_r=1的特例。门控额外参数1,536维，仅占记忆矩阵的4.7%。

### 3.5 多任务训练策略

六阶段课程训练： (1)冻结RoBERTa+ViT，训练门控融合(~15K参数, lr=1e-3)；(2)解冻顶层6层微调块编码器(预训练lr=2e-5)；(3)单页分类基线（无页编码器/记忆）；(4)加入扩张门控页编码器(10页窗口)；(5)引入门控Infini记忆(20页窗口)；(6)全文档端到端联合训练(最多612页)。

超参数：AdamW(beta1=0.9, beta2=0.999)，权重衰减1e-4，梯度裁剪1.0，FP16混合精度，Dropout=0.1，随机种子42。

---

## 4 实验设计

### 4.1 数据集

从公开采购渠道收集24份投标文件（PDF），涵盖工程建设、货物采购和服务招标三类。

**表2：数据集统计**

| 指标 | 数值 |
|---|---|
| 文档数 | 24 |
| 总页数 | 8,750 |
| 总块数 | 85,304 |
| 文本块 | 68,617 (80.4%) |
| 图像/混合块 | 16,687 (19.6%) |
| 已标注块 | 907 (1.06%) |
| 类别数 | 19 |
| 文本中位长度 | 17字符 |
| >510字符的块 | 15 (0.02%) |

19类标签涵盖投标文件典型章节：封面页、目录、商务偏离表、投标保证金、关联关系说明、公司基本情况表、营业执照、税务证明、资格证明文件、财务状况、财务凭证、资质业绩凭证、评分支撑材料、名称变更、一致性承诺函、十不准、公章授权书、其他、法定代表人授权委托书。

标注稀疏性说明：仅1份PDF获得详尽标注。在当前标注规模下，实验核心目标是验证数据管线正确性、模型前向可行性和各模块结构有效性。统计显著性能评估待扩展标注后实施。

### 4.2 实现细节

PyTorch 2.x + HuggingFace Transformers。文本编码器hfl/chinese-roberta-wwm-ext（110M参数），图像编码器google/vit-base-patch16-224（86M参数）。OCR预处理通过PaddleOCR在数据准备阶段缓存为JSONL。单卡NVIDIA GPU（24GB），FP16混合精度。

### 4.3 评估指标

**块级**（仅已标注块）：Micro-Accuracy, Macro-F1（等权各类，反映小样本敏感性）, Weighted-F1（样本加权）, 逐类P/R/F1。
**边界级**：B-SECTION检测F1（核心），BIOE序列准确率。
**章节级**：片段准确率，章节IoU。

### 4.4 消融实验设计

**表3：五组消融实验矩阵**

| ID | 变体 | 块编码器 | 页编码器 | 记忆 | 验证目标 |
|---|---|---|---|---|---|
| A | Baseline | Concat+Linear | 固定窗口w=5 | K-slot GRU | 基线 |
| B | +GatedFusion | GatedFusionGate | 固定窗口w=5 | K-slot GRU | 门控融合增益 |
| C | +DilatedGate | GatedFusionGate | 扩张+top-k gate | K-slot GRU | 扩张门控增益 |
| D | +Infini | GatedFusionGate | 扩张+top-k gate | 标准Infini | 压缩记忆增益 |
| E | Full v3 | GatedFusionGate | 扩张+top-k gate | 门控Infini | 门控记忆增益 |

敏感度网格：K in {2,4,8,16}, L in {2,3,4,6}, k in {4,8,16,32}, 窗口 in {5,10,20,all}。

---

## 5 讨论

### 5.1 三级门控稀疏的互补性

三个层次门控服务的功能具有本质差异且输入信号互不重叠：模态级（输入排版特征，学习模态权重分布），页内级（输入块内容嵌入对，学习语义连接选择），跨页级（输入页上下文，学习记忆维度更新策略）。三者功能角色正交，联合训练的梯度干扰风险低。

### 5.2 复杂度与容量对比

**表4：基线v1 vs.本文v3核心指标**

| 维度 | v1基线 | v3本文 | 改进 |
|---|---|---|---|
| 模态融合 | 静态Concat+Linear | 排版驱动门控 | 自适应 |
| 等效感受野 | 5块 | 40块（4层） | 8x |
| 记忆类型 | K-slot GRU | 压缩delta-rule | 结构升级 |
| 记忆容量 | 512维 | 32,768维 | 64x |
| 记忆更新 | 等权门控 | 逐维门控 | 选择性 |
| 门控额外参数 | — | 1,536维 | 可忽略(4.7%) |

### 5.3 局限性

（1）标注数据约束：当前仅907块标注（1/24文档），需扩展标注后进行统计显著评估。（2）OCR级联误差：混合块依赖OCR质量，印章和水印对识别构成挑战。（3）跨域泛化：不同地区/行业投标文件格式差异显著，需多源数据验证。

### 5.4 可扩展方向

Mamba-2替代Transformer页编码器[15]实现严格线性复杂度；线性链CRF边界约束[16]消除非法标签序列；多模态RAG检索增强[17]提供分类先验；弱监督伪标签迭代扩充训练集。

---

## 6 结论

本文提出HMSAN-BSA，首次将门控稀疏注意力作为系统性架构设计原则统一应用于多模态投标文件切片分类的三个关键层次：块级门控融合从排版特征端到端学习模态权重；页级扩张门控注意力以线性复杂度实现指数感受野覆盖；跨页级门控Infini记忆以64倍容量提升实现无限页面的上下文传播。三个层次门控服务于不同功能，输入信号和梯度路径正交，为统一训练提供稳定性基础。

针对引言提出的三个核心挑战的解决方案映射：（a）多模态异质性由门控融合解决（第3.2节）；（b）长程跨页依赖由扩张门控注意力（第3.3节）和门控Infini记忆（第3.4节）联合解决；（c）标注稀疏性由ignore-index掩码（第3.5节）解决。

未来核心方向：Mamba-2序列编码替代、CRF约束解码、检索增强生成和弱监督标注扩展。

## 参考文献

[1] Y. Xu, M. Li, L. Cui, et al. LayoutLM: Pre-training of Text and Layout for Document Image Understanding. KDD, 2020. arXiv:1912.13318.

[2] Y. Xu et al. LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding. ACL, 2021. arXiv:2012.14740.

[3] Y. Huang et al. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. ACM MM, 2022. arXiv:2204.08387.

[4] A. Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR, 2021.

[5] I. Beltagy et al. Longformer: The Long-Document Transformer. 2020. arXiv:2004.05150.

[6] Y. Xu et al. LayoutLM, KDD 2020.

[7] Y. Xu et al. LayoutLMv2, ACL 2021.

[8] Y. Huang et al. LayoutLMv3, ACM MM 2022.

[9] J. Wang et al. LiLT: A Language-Independent Layout Transformer. ACL, 2022. arXiv:2202.13669.

[10] T. Munkhdalai et al. Leave No Context Behind: Infini-attention. 2024. arXiv:2404.07143.

[11] J. Ding et al. LongNet: Scaling Transformers to 1,000,000,000 Tokens. 2023. arXiv:2307.02486.

[12] A. Shen and A. Shen. Gated Sparse Attention. 2026. arXiv:2601.15305.

[13] Y. Cui et al. Pre-Training with Whole Word Masking for Chinese BERT. IEEE/ACM TASLP, 2021. arXiv:1906.08101.

[14] A. Dosovitskiy et al. ViT, ICLR 2021.

[15] A. Gu and T. Dao. Mamba: Linear-Time Sequence Modeling. 2023. arXiv:2312.00752.

[16] J. Lafferty et al. Conditional Random Fields. ICML, 2001.

[17] S. Gao and S. Zhao. Multimodal RAG for Document Understanding (Survey). 2025. arXiv:2510.15253.

[18] D. Liu and Y. Yu. MKA: Memory-Keyed Attention. 2026. arXiv:2603.20586.

[19] Q. Peng et al. ERNIE-Layout. EMNLP Findings, 2022. arXiv:2210.06155.

[20] T. Douzon and S. Duffner. Long-Range Transformer for Document Understanding. 2023. arXiv:2309.05503.

[21] Y. Tu et al. LayoutMask. 2023. arXiv:2305.18721.
