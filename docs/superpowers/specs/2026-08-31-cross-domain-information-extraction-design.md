# 跨域信息抽取模型设计

> 状态：设计草案，待用户审阅
> 日期：2026-08-31
> 关联项目：HMSAN-BSA 投标文件切片分类与信息抽取

## 1. 目标

在第一阶段切片分类模型的基础上，新增信息抽取子系统，从投标文件中抽取结构化字段与实体关系。系统采用“规则弱监督 + PLM 跨域迁移 + 小样本人工补标”路线，主要面向约 100 份投标文档的逐步标注场景。

## 2. 范围

### 2.1 本次实现

- 从现有 `Document/Page/Block` 数据构建页级文本序列与字符偏移映射。
- 定义实体、关系、财务属性 schema。
- 生成规则弱标注，支持人工修正与增量合并。
- 实现跨域实体检测、Prompt 类型预测、跨页关系抽取。
- 实现训练、验证、预测、导出 JSON 和论文实验指标。

### 2.2 不实现

- 不改动已训练好的切片分类模型 `outputs/hmsan_bsa_ft3/best_model.pt`。
- 不重构现有切片分类训练代码。
- 不直接使用需要大显存或外部 API 的大语言模型作为主线模型。

## 3. 输出 Schema

### 3.1 实体字段

P0 字段：

- 项目名称
- 招标人/采购人
- 分标编号
- 分标名称
- 包号
- 投标人名称
- 统一社会信用代码
- 注册地址/住所
- 联系人
- 联系电话
- 法定代表人
- 授权代表/被授权人
- 成立日期
- 注册资本/注册资本金

P1 字段：

- 投标报价金额
- 投标保证金金额
- 工期/交货期
- 营业利润
- 营业收入净额

P2 字段：

- 资质等级/证书名称
- 投标日期/授权日期
- 净利润
- 资产总额
- 负债总额

### 3.2 财务指标建模

财务指标统一建模为：

- `type`：指标名称，如“营业利润”。
- `mentions`：数值 Span。
- `attrs.year`：年份，如 `2023`。
- `attrs.value`：数值。
- `attrs.unit`：单位，如“元”。
- `attrs.source`：来源，如“财务状况表”。

### 3.3 关系类型

- `has_section`：项目 → 分标
- `has_package`：分标 → 包
- `bids_for`：投标人 → 项目
- `represented_by`：投标人 → 法定代表人/授权代表
- `has_financial_indicator`：投标人 → 营业利润/注册资本金等

## 4. 数据组织

### 4.1 文档注册表

新增 `data/ie_doc_registry.json`：

```json
{
  "docs": {
    "17441623010036701.pdf": {
      "zip": "training_data_2026-08-04-ocr.zip",
      "split": "train",
      "status": "weak_labeled",
      "annotated": true
    }
  }
}
```

状态：

- `unlabeled`
- `weak_labeled`
- `prelabeled`
- `annotated`
- `test_locked`

### 4.2 标注文件

每个文档一个 JSON 文件，建议路径：

`data/ie_annotations/<pdf_name>.json`

示例：

```json
{
  "pdf_name": "17441623010036701.pdf",
  "pages": [
    {
      "page_num": 1,
      "segments": [
        {
          "segment_id": "b0",
          "block_id": "0",
          "block_type": "text",
          "text": "投标人： 北京晟开源电器设备有限责任公司",
          "ocr_text": ""
        }
      ]
    }
  ],
  "entities": [
    {
      "id": "e1",
      "type": "投标人名称",
      "mentions": [
        {"page_num": 1, "segment_id": "b0", "start": 5, "end": 18}
      ],
      "attrs": {"source": "封面页", "year": null, "value": null, "unit": null}
    },
    {
      "id": "e2",
      "type": "项目名称",
      "mentions": [
        {"page_num": 1, "segment_id": "b0", "start": 0, "end": 4}
      ],
      "attrs": {"source": "封面页", "year": null, "value": null, "unit": null}
    }
  ],
  "relations": [
    {"id": "r1", "head": "e1", "tail": "e2", "type": "bids_for"}
  ]
}
```

约定：

- `entities` 为文档级，允许一个实体有多个 `mentions`。
- `mentions.start/end` 是所在 `segment.text` 内的字符偏移。
- 工具从实体 Span 自动生成 BIO 标签。
- 同一 `pdf_name` 的标注可增量合并，不覆盖历史。

### 4.3 100 份文档划分

- 最终按 `70/15/15` 划分：约 70 份训练、15 份验证、15 份测试。
- 测试集标记为 `test_locked`，不参与训练、调参和人工补标。
- 后续新增文档按目标比例分配进入训练/验证池，已锁定的测试文档不变，使用确定性注册脚本增量加入。
- 单人标注目标为精标约 35–45 份训练/验证文档，其余训练文档由弱标注和模型预标注覆盖。

## 5. 系统架构

### 5.1 数据流

1. 从现有 `Document/Page/Block` 数据构建页级序列。
2. 规则弱标注器生成源域 Span。
3. 标注工具加载规则/模型预标注，人工修正后输出目标域标注。
4. PLM 信息抽取模型完成实体检测、类型预测、关系抽取。
5. 评估/导出器输出字段、关系和论文实验指标。

### 5.2 新增目录

- `src/bid_slicing/ie/schema.py`
- `src/bid_slicing/ie/dataset.py`
- `src/bid_slicing/ie/weak_labels.py`
- `src/bid_slicing/ie/span_detector.py`
- `src/bid_slicing/ie/type_predictor.py`
- `src/bid_slicing/ie/relation_extractor.py`
- `src/bid_slicing/ie/ie_model.py`
- `src/bid_slicing/ie/train.py`
- `src/bid_slicing/ie/evaluate.py`
- `src/bid_slicing/ie/predict.py`
- `configs/ie_labels.yaml`
- `annotation_tool/ie_app.py`

## 6. 模型设计

### 6.1 模块一：布局感知的跨域实体检测

输入：页级字符序列及其来源块；中文文本按 RoBERTa-wwm-ext 字符级 token 处理。

字符表示：

```text
h_i = RoBERTa(char_i) + LayoutProj(layout_i)
```

`layout_i` 包括：

- 块类型
- 归一化 `bbox`
- 字号
- 加粗/斜体
- 表格行/列
- 是否为 OCR 图片块

对 MIXED 图片块，使用轻量 `CrossAttn(text_tokens, ViT_image_context)` 并做门控融合。

Span 头输出 `O / B-ENT / I-ENT`。

领域判别器：

```text
L_domain = BCE(D(GRL(h_i)), domain_id)
λ = 2 / (1 + exp(-γ·p)) - 1
```

其中 `p` 为训练进度，`γ=10`。

`domain_id=0` 表示源域弱标注或公开 NER，`domain_id=1` 表示目标域人工标注。

### 6.2 模块二：Schema 可扩展的 Prompt 类型预测

基础模板：

```text
[CLS] 在招投标文件中，实体“[SPAN]”的类型是 [MASK] 。[SEP]
```

布局增强模板：

```text
[CLS] 位于文档第 X 页{表格第 Y 行}的实体“[SPAN]”的类型是 [MASK] 。[SEP]
```

Verbalizer 聚合：

```text
P(type=t | span) = 1/|V_t| · Σ_{v∈V_t} P_MLM(v | [MASK])
```

目标域小样本使用软提示：在输入前拼接 `m=20` 个可学习向量，冻结 RoBERTa 与 MLM 头。

财务属性由 `AttributeParser` 解析：

- 数值 Span 直接作为 `attrs.value`。
- 年份、单位、来源优先用表格行列规则提取。

### 6.3 模块三：跨页记忆增强的关系抽取

实体表示经页间记忆增强：

```text
e_ctx = SparseAttn(e, M_page)
```

关系 Prompt：

```text
[CLS] [HEAD] 与 [TAIL] 在文档中的关系是 [MASK] 。[SEP]
```

训练时对每个正例采样同页、跨页负例。

### 6.4 总损失

```text
L = L_span + λ1·L_domain + λ2·L_type + λ3·L_rel
```

## 7. 训练顺序

1. 阶段 A：训练实体检测与领域判别器。
2. 阶段 B：冻结 PLM，训练软提示与 Verbalizer。
3. 阶段 C：训练关系抽取。
4. 阶段 D：数据稳定后小学习率联合微调各 Head。

## 8. 评估

指标：

- 实体边界 F1
- 实体类型 F1
- 字段级准确率
- 关系 F1
- 财务属性四元组准确率：`指标、年份、数值、单位`
- 跨域新增类型 F1
- 标注效率：达到同等 F1 所需人工标注时间

评估数据集：

- 验证集：约 15 份文档，用于阶段间选型。
- 测试集：`test_locked` 文档，最终论文报告使用。

## 9. 创新点映射

- 布局感知跨域实体检测：`LayoutProj`、表格行列、OCR 图片上下文。
- 跨页记忆增强关系抽取：`SparseAttn(e, M_page)`。
- 弱监督与主动学习：源域 `domain_id=0`、主动学习选样。
- Schema 可扩展类型预测：Prompt/Verbalizer，新增字段不改架构。
- 财务表格属性解析：`AttributeParser` 与四元组评估。

## 10. 单人实施阶段

1. 实现 schema、文档注册表、标注文件读写。
2. 实现规则弱标注器与标注工具。
3. 实现实体检测训练与验证。
4. 实现类型预测训练与验证。
5. 实现关系抽取训练与验证。
6. 实现端到端预测与 JSON 导出。
7. 生成论文实验表并输出消融实验配置。

## 11. 风险与缓解

- 12GB 显存不足：冻结 RoBERTa/ViT，仅训练轻量 Head、软提示和投影层。
- OCR 文本噪声：MIXED 分支保留图像上下文；人工优先修正低质量 OCR 文档。
- 财务表格结构差异：先用表格行列规则，PLM 只处理规则失败的 Span。
- 后续数据到达导致分布变化：使用文档注册表增量合并，测试集锁定。
