# Gated Sparse Attention for Hierarchical Memory Networks in Tender Document Slice Classification

## Abstract
Tender documents are structurally complex multi-page documents containing diverse section types such as cover pages, tables of contents, financial statements, and qualification certificates. Accurate classification of individual document slices requires modeling three levels of context: local block semantics, within-page spatial layout, and cross-page section continuity. We propose **HMSAN-BSA** (Hierarchical Memory Sparse Attention Network with Boundary-aware Section Aggregation), a multi-modal architecture that processes documents at the Document-Page-Block hierarchy. Our model integrates three recent advances under a unified **Gated Sparse Attention** framework: (1) a **Gated LiLT Block Encoder** that dynamically fuses RoBERTa-wwm-ext text embeddings, ViT-B/16 image features, and layout cues via learnable modality gates predicted from typographic signals; (2) a **Dilated Gated Page Encoder** that combines LongNet-style exponential dilation with per-layer top-k sparse gating, reducing attention complexity from O(N^2) to O(N*w) while maintaining global receptive fields; and (3) a **Gated Infini Compressive Memory** that employs delta-rule memory updates with selective per-dimension write gates and sparse read gates, enabling effective context propagation across 150+ page documents without fixed slot bottlenecks. On a proprietary dataset of 24 tender documents (85,304 blocks, 19 classes, 907 labeled blocks), we design a multi-task training scheme with block classification, section boundary detection, and section-level prediction.


## 1. Introduction

Tender (bid) documents form the backbone of public procurement processes worldwide. A typical tender submission spans 50-200 pages and contains highly structured yet visually diverse content: cover letters, compliance tables, company profiles, bank guarantees, tax certificates, and technical specifications. Automating the classification of these document slices is critical for downstream tasks such as compliance verification, information extraction, and automated bid evaluation.

Document slice classification presents three interconnected challenges. First, **multi-modal blocks**: a single page contains text paragraphs, signature stamps (images), and mixed blocks where text overlaid on seals or logos requires joint understanding of visual and textual modalities. Second, **long-range dependencies**: a financial statement section may span 40 consecutive pages, and a qualification certificate appearing on page 80 must be interpreted in the context of the bidders company profile on page 5. Third, **sparse supervision**: in real-world procurement workflows, only a small fraction of documents receive detailed slice-level annotations, with the majority of blocks remaining unlabeled.

Prior work addresses individual aspects of this problem. LayoutLM [1] and its successors LayoutLMv2 [2], LayoutLMv3 [3] established pre-training paradigms for visually-rich document understanding by jointly modeling text tokens, spatial coordinates, and image patches. LiLT [4] further decoupled layout modeling from language, enabling plug-and-play layout attention bias injection into arbitrary pre-trained text encoders. For long-document processing, Infini-attention [5] introduced compressive memory with delta-rule updates to achieve theoretically infinite context, while LongNet [6] proposed dilated attention windows to scale Transformers to billion-token sequences with sub-quadratic complexity. Most recently, Gated Sparse Attention [7] demonstrated that learned gating mechanisms can dynamically select sparse attention connections, achieving both computational efficiency and training stability.

However, no existing work unifies these three directions -- multi-modal encoding, sparse attention, and compressive memory -- under a coherent document processing pipeline for the tender domain. We propose **HMSAN-BSA** (Hierarchical Memory Sparse Attention Network with Boundary-aware Section Aggregation), a novel architecture that integrates Gated Sparse Attention across all three modeling levels.

The primary contributions of this work are:

1. A novel **unified Gated Sparse Attention framework** that integrates learnable sparsity across modality fusion, page-level encoding, and cross-page memory.
2. **Gated Infini Memory**, an extension of Infini-attention with selective write/read gating mechanisms.
3. **Dilated Gated Page Encoder**, combining LongNet-style exponential dilation with per-layer top-k sparse token selection.
4. A complete **multi-modal, multi-task architecture** for tender document slice classification with boundary-aware section aggregation.


## 2. Related Work

### 2.1 Visually-Rich Document Understanding

The LayoutLM family [1, 2, 3] pioneered the joint pre-training of text, layout, and visual features for document understanding. LayoutLMv1 combined BERT with 2D positional embeddings, while LayoutLMv2 introduced spatial-aware self-attention with relative position biases. LayoutLMv3 unified text and image masking at the patch level using a Vision Transformer backbone. ERNIE-Layout [8] incorporated layout knowledge through reading-order prediction. LiLT [4] proposed a language-independent layout Transformer that injects bbox-derived attention biases into any pre-trained text model. LayoutMask [9] enhanced text-layout interaction through a novel masking strategy.

### 2.2 Sparse and Efficient Attention Mechanisms

Longformer [10] combined local sliding windows with global tokens and dilated windows. BigBird [11] introduced a combination of random, local, and global attention achieving O(n) complexity. LongNet [6] proposed dilated attention with exponentially increasing dilation factors across layers. DiNAT [12] brought dilated neighborhood attention to vision Transformers. Gated Sparse Attention [7] introduced a fundamentally different approach: a learned gate network predicts which token pairs should interact, trained with straight-through estimation.

### 2.3 Memory-Augmented Transformers

Memorizing Transformers [13] augment each attention layer with a kNN-retrieved external memory. Infini-attention [5] introduces compressive memory using delta-rule updates with a learned gate combining memory retrieval and local attention. MKA [14] proposed memory-keyed attention for addressable content retrieval.

### 2.4 Document Boundary Detection

Section boundary detection has been explored through CRF-based decoders [15] and hierarchical BiLSTM-CRF architectures [16]. Our boundary head follows the BIOE tagging scheme and is trained jointly with the classification objective.


## 3. Method

### 3.1 Overall Architecture and Data Flow

HMSAN-BSA processes a document as an ordered sequence of pages, where each page contains a variable number of blocks:

    Document -> [Page1 -> Page2 -> ... -> Page_T]
    Per Page: [b1, b2, ..., b_N]
        |-> Gated LiLT Block Encoder: block_vecs [N, d_model]
        |-> Dilated Gated Page Encoder: page_repr, encoded_blocks
        |-> Gated Infini Memory: fused_page_repr, memory_state
        |-> Heads: ClassificationHead / BoundaryHead / SectionHead

**Table 1: Model Configuration**

| Component | Parameter | Value |
|---|---|---|
| General | Hidden size | 128 |
| | Dropout | 0.1 |
| | Number of classes | 19 |
| | Boundary labels | 4 (B/I/E/O) |
| Block Encoder | Text model | RoBERTa-wwm-ext |
| | Text output dim | 768 |
| | Max text length | 510 tokens |
| | Image model | ViT-B/16 |
| | Image output dim | 768 |
| | Layout output dim | 64 |
| | Modality gates | 3 (text, layout, image) |
| Page Encoder | Layers | 4 |
| | Attention heads | 4 |
| | Base window size | 5 |
| | Dilation sequence | [1, 2, 4, 8] |
| | Global tokens | 4 |
| | Gate top-k | 16 |
| Memory | Slots | 4 |
| | Key dimension d_key | 64 |
| | Value dimension d_value | 128 |


### 3.2 Gated LiLT Block Encoder

The block encoder produces a unified vector representation for blocks of type text, image, or mixed. We propose a **GatedFusionGate** that dynamically weights each modality based on layout-derived cues: alpha = softmax(W2 * GELU(W1 * layout_emb + b1) + b2). The softmax gate is predicted entirely from layout features (bbox, font size, bold/italic flags, font color, block type embedding), based on the intuition that typographic properties are informative about which modality carries the dominant signal: large bold centered text implies high text weight; image-type blocks imply high image weight.

Additionally, a **LiLT-style layout attention bias** is injected as a residual. From each bbox we extract 5-dimensional spatial descriptor [cx, cy, w, h, area] and project it through a two-layer MLP to match the attention head count.

Text blocks are encoded via RoBERTa-wwm-ext [17] with max 510 tokens; blocks exceeding this limit (0.02% of dataset) use head-tail truncation (first 256 + last 254 characters). Image blocks use ViT-B/16 [18] CLS token. Mixed blocks: OCR text through RoBERTa, image through ViT, concatenated and projected.


### 3.3 Dilated Gated Page Encoder

Within-page attention must account for variable spatial and semantic distances between blocks. A fixed local window of +-5 cannot capture long-range dependencies, while full self-attention incurs O(N^2) cost.

**Dilated Candidate Selection (LongNet).** Each layer l receives dilation = 2^l. For a block at position i, the candidate set C(i,l) consists of tokens at positions i +- k*d for k in [1, w]. Effective receptive field at layer l: w * 2^l blocks (40 at l=3). Complexity: O(N * (2w + G + 1)) per layer.

**Gated Sparse Selection.** Within the dilated candidate set, a learned gate selects top-k connections: g_ij = sigma(W_g * [Qi; Kj] + b_g) / tau, mask = top-k(g). During training, straight-through estimation is used: forward uses discrete mask, backward propagates through continuous gate values.

**Global Tokens.** Four learnable global tokens attend to all blocks and are pooled to produce page_repr. The encoder has L=4 layers with GELU FFN (4x expansion) and residual connections.


### 3.4 Gated Infini Compressive Memory

Cross-page context propagation is essential for sections spanning tens of pages. We propose **Gated Infini Memory** extending Infini-attention [5] with per-dimension write and read gates. Memory: M in R^(K x d_key x d_value) with K=4 slots.

**Gated Write:** k = ELU(W_k * p_t) + 1; v = W_v * p_t; gate_w = sigma(W_gw * p_t); r = sigma(k)^T @ M; e = v - r; delta = gate_w (X) (k (X) e); M_new = M + delta. The write gate selectively updates dimensions based on page context.

**Gated Read:** q = ELU(W_q * p_t) + 1; r = sigma(q) @ M / (sigma(q) @ z + epsilon); gate_r = sigma(W_gr * [p_t; r]); memory_out = gate_r (X) r.

**Fusion:** beta = sigma(W_beta * [local_out; memory_out]); fused = beta * local_out + (1 - beta) * memory_out.

Compared to vanilla Infini-attention (no gates), our extension adds 2*d_value parameters per gate with negligible overhead relative to the 32,768-dim memory matrix.


### 3.5 Multi-Task Training

Joint objective: L_total = L_cls + alpha * L_boundary + beta * L_section.

Block classification uses cross-entropy with ignore_index=-100 for unlabeled blocks. Boundary detection uses per-page BIOE cross-entropy (labels derived from block label transitions). Section classification predicts the dominant section label per page.

Six-stage curriculum:
1. Freeze RoBERTa+ViT, train GatedFusionGate+LayoutEncoder
2. Unfreeze top layers, end-to-end block encoder fine-tuning
3. Single-page classification (no memory)
4. Add DilatedGatedPageEncoder
5. Add GatedInfiniMemory with 10-page sliding window
6. Full-document end-to-end joint training


## 4. Experimental Setup

### 4.1 Dataset
| Metric | Value |
|---|---|
| Total PDFs | 24 |
| Total pages | 8,750 |
| Total blocks | 85,304 |
| Text blocks | 68,617 |
| Labeled blocks | 907 |
| Labeled pages | 151 |
| Classes | 19 |
| Median text length | 17 chars |
| 99th percentile length | 204 chars |

Classes: cover page, table of contents, commercial deviation, bid guarantee, relationship statement, company profile, business license, tax certificate, qualification documents, financial statements, financial certificates, performance certificates, scoring support, name change, compliance commitment, ten prohibitions, official seal authorization, others, legal representative authorization.

### 4.2 Implementation
PyTorch + HuggingFace Transformers. AdamW optimizer: 2e-5 for pretrained encoders, 1e-3 for gate/memory/head parameters. Weight decay 1e-4. Batch size 1 (document-level).

### 4.3 Metrics
Block-level: Accuracy, Macro-F1, per-class P/R/F1. Boundary-level: B/I/E/O detection accuracy. Section-level: segment accuracy, section IoU.

### 4.4 Ablation Design
| Variant | Block Encoder | Page Encoder | Memory |
|---|---|---|---|
| Baseline | Concat+Linear | Fixed window w=5 | K-slot GRU |
| +Gated Fusion | GatedFusionGate | Fixed window w=5 | K-slot GRU |
| +Dilated Gate | GatedFusionGate | Dilated+top-k gate | K-slot GRU |
| +Infini | GatedFusionGate | Dilated+top-k gate | Vanilla Infini |
| Full v3 | GatedFusionGate | Dilated+top-k gate | Gated Infini |


## 5. Discussion

### 5.1 Gated Sparse Across All Levels
The gates serve distinct functions: at the modality level (block-type-dependent fusion), at the within-page level (spatial and semantic attention patterns), and at the cross-page level (temporal update policies). These three gate objectives are orthogonal in their input signals and functional roles.

### 5.2 Capacity Analysis
Gated Infini Memory provides 32,768 total parameters for context storage (vs. 512 in baseline). The delta-rule update allows unbounded accumulation without memory overwriting. Per-dimension gates enable learned input-dependent forgetting.

### 5.3 Extensibility
The componentized architecture allows module replacement: additional modalities via expanded softmax gate, Mamba-2 [19] as page encoder alternative, memory-keyed addressing [14] for 1000+ page documents.


## 6. Conclusion

We presented HMSAN-BSA, a hierarchical memory sparse attention network for multi-modal tender document slice classification. Our key contribution is the unified application of Gated Sparse Attention across three modeling levels: modality fusion, dilated page attention, and compressive memory. The model handles sparse annotations via ignore-index masking and employs a six-stage curriculum training strategy. Future work includes retrieval-augmented generation [20], CRF-based decoding [21], and state-space model alternatives [19].

## References

[1] Y. Xu et al., LayoutLM: Pre-training of Text and Layout for Document Image Understanding, KDD 2020. arXiv:1912.13318

[2] Y. Xu et al., LayoutLMv2: Multi-modal Pre-training for Visually-rich Document Understanding, ACL 2021. arXiv:2012.14740

[3] Y. Huang et al., LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking, ACM MM 2022. arXiv:2204.08387

[4] J. Wang et al., LiLT: A Simple yet Effective Language-Independent Layout Transformer, ACL 2022. arXiv:2202.13669

[5] T. Munkhdalai et al., Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention, 2024. arXiv:2404.07143

[6] J. Ding et al., LongNet: Scaling Transformers to 1,000,000,000 Tokens, 2023. arXiv:2307.02486

[7] A. Shen and A. Shen, Gated Sparse Attention: Combining Computational Efficiency with Training Stability, 2026. arXiv:2601.15305

[8] Q. Peng et al., ERNIE-Layout: Layout Knowledge Enhanced Pre-training for Visually-rich Document Understanding, EMNLP 2022. arXiv:2210.06155

[9] Y. Tu et al., LayoutMask: Enhance Text-Layout Interaction in Multi-modal Pre-training, 2023. arXiv:2305.18721

[10] I. Beltagy et al., Longformer: The Long-Document Transformer, 2020. arXiv:2004.05150

[11] M. Zaheer et al., Big Bird: Transformers for Longer Sequences, NeurIPS 2020.

[12] A. Hassani and H. Shi, Dilated Neighborhood Attention Transformer, 2022. arXiv:2209.15001

[13] Y. Wu et al., Memorizing Transformers, ICLR 2022. arXiv:2203.08913

[14] D. Liu and Y. Yu, MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning, 2026. arXiv:2603.20586

[15] J. Lafferty et al., Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data, ICML 2001.

[16] Z. Huang et al., Bidirectional LSTM-CRF Models for Sequence Tagging, 2015. arXiv:1508.01991

[17] Y. Cui et al., Pre-Training with Whole Word Masking for Chinese BERT, IEEE/ACM TASLP 2021. arXiv:1906.08101

[18] A. Dosovitskiy et al., An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, ICLR 2021.

[19] A. Gu and T. Dao, Mamba: Linear-Time Sequence Modeling with Selective State Spaces, 2023. arXiv:2312.00752

[20] S. Gao and S. Zhao, Scaling Beyond Context: A Survey of Multimodal RAG for Document Understanding, 2025. arXiv:2510.15253

[21] T. Douzon and S. Duffner, Long-Range Transformer Architectures for Document Understanding, 2023. arXiv:2309.05503
