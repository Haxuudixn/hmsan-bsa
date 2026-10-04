# 训练数据说明

训练脚本会扫描 `DataDir` 下所有 `training_data_*.zip` 文件，每个 ZIP 内部需要包含 `annotations.csv`。

## ZIP 格式

`annotations.csv` 建议字段（UTF-8，首列前可带 BOM）:

- pdf_name: 文档名
- page_num: 页码（从 1 开始）
- block_type: text / image / mixed
- bbox: 形如 (x0,y0,x1,y1)
- content: 文本块内容；图片块可填图片相对路径
- font: 字体名
- font_size: 字号
- is_bold / is_italic: 0 或 1
- font_color: 如 #000000
- label: 类别名（见 configs/labels.yaml）
- boundary_label: O / B-SECTION / I-SECTION / E-SECTION
- ocr_text: 图片块 OCR 文本（可选）
- block_id: 全局唯一 id
- block_file: ZIP 内相对路径，例如 blocks/block_0001.txt 或 blocks/block_0001.png

图片块对应的 PNG 文件需放在同一 ZIP 内，并在 `block_file` 中指向它。

## 从标注工具导出生成训练 ZIP

- JSON 导出: `python scripts/synthesize_training_zip.py --json <annotations.json> --pdf-root <PDF目录> --output <输出.zip>`
- CSV 导出: `python scripts/synthesize_training_zip_csv.py --csv <annotations.csv> --pdf-root <PDF目录> --output <输出.zip>`

## 验证数据

- `python scripts/verify_data_loader.py <你的训练.zip>` 可快速检查数据能否被正确加载。
