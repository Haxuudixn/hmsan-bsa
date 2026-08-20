"""投标文件切片标注工具 - PyMuPDF + Streamlit + EasyOCR"""
from __future__ import annotations
import streamlit as st, fitz, pandas as pd, json, os, io, time, numpy as np, easyocr
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import yaml

# ============================================================
# Config
# ============================================================
LABELS_YAML = Path(__file__).parent.parent / "configs" / "labels.yaml"
if LABELS_YAML.exists():
    with open(LABELS_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    LABELS = cfg.get("class_labels", [])
    BOUNDARY_LABELS = cfg.get("boundary_labels", ["O", "B-SECTION", "I-SECTION", "E-SECTION"])
else:
    LABELS = ["封面页","目录","商务偏差表","投标保证金","关系说明","基本情况表","营业执照","税务证明","资格证明文件","财务状况","财务凭证单","资质业绩凭证单","评分支撑材料","名称变更","一致性承诺函","十不准","公章授权书","其他","法定代表人授权委托书"]
    BOUNDARY_LABELS = ["O", "B-SECTION", "I-SECTION", "E-SECTION"]

ANNOTATION_DIR = Path(__file__).parent / "annotations"
ANNOTATION_DIR.mkdir(exist_ok=True)
st.set_page_config(layout="wide", page_title="投标文件标注工具")

# ============================================================
# Session State
# ============================================================
for key, default in [
    ("pdf_name", None), ("pdf_bytes", None), ("current_page", 0),
    ("annotations", {}), ("selected_block", None),
    ("pending_action", None),  # for deferred actions
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# Functions
# ============================================================
def load_progress(pdf_name):
    path = ANNOTATION_DIR / f"{Path(pdf_name).stem}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}

def save_progress(pdf_name, annotations):
    path = ANNOTATION_DIR / f"{Path(pdf_name).stem}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

def parse_page_blocks(doc, page_num):
    page = doc[page_num]
    blocks = page.get_text("dict")["blocks"]
    parsed, block_id = [], 0
    for b in blocks:
        if b["type"] == 0:
            all_spans = []
            for line in b["lines"]:
                all_spans.extend(line["spans"])
            if not all_spans:
                continue
            text = "".join(s["text"] for s in all_spans)
            ms = max(all_spans, key=lambda s: len(s["text"]))
            flags = ms.get("flags", 0)
            parsed.append(dict(
                block_id=block_id, block_type="text", content=text.strip(),
                bbox=[round(v, 1) for v in b["bbox"]],
                font=ms.get("font", ""), font_size=round(ms.get("size", 0), 1),
                is_bold=int(bool(flags & 8)), is_italic=int(bool(flags & 2)),
                font_color="#{:06x}".format(ms.get("color", 0)),
                label="", boundary_label="O", ocr_text=""))
            block_id += 1
        elif b["type"] == 1:
            parsed.append(dict(
                block_id=block_id, block_type="image", content="",
                bbox=[round(v, 1) for v in b["bbox"]],
                font="", font_size=0, is_bold=0, is_italic=0, font_color="",
                label="", boundary_label="O", ocr_text=""))
            block_id += 1
    return parsed

def render_page_with_boxes(doc, page_num, blocks, selected_idx, zoom=2.0):
    page = doc[page_num]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", int(10 * zoom))
    except Exception:
        font = ImageFont.load_default()
    for i, block in enumerate(blocks):
        x0, y0, x1, y1 = [v * zoom for v in block["bbox"]]
        if i == selected_idx:
            color, width = (255, 204, 0, 200), 3
        elif block.get("label"):
            color, width = (0, 255, 100, 140), 2
        elif block["block_type"] == "image":
            color, width = (100, 150, 255, 100), 1
        else:
            color, width = (255, 100, 100, 80), 1
        draw.rectangle([x0, y0, x1, y1], outline=color[:3], width=width)
        if block.get("label"):
            tw = int(zoom * 8 * len(block["label"]))
            draw.rectangle([x0, y0 - int(16 * zoom), x0 + tw, y0],
                          fill=color[:3] + (180,))
            draw.text((x0 + 2, y0 - int(15 * zoom)), block["label"],
                     fill=(0, 0, 0), font=font)
    return img

_ocr_reader = None
def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        with st.spinner("加载OCR..."):
            _ocr_reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
    return _ocr_reader

def ocr_image_block(doc, page_num, bbox, zoom=2.0):
    page = doc[page_num]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=fitz.Rect(bbox))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    if img.width < 10 or img.height < 10:
        return ""
    arr = np.array(img)
    results = get_ocr_reader().readtext(arr)
    return chr(10).join(r[1] for r in results if r[2] > 0.3)

def ocr_all_image_blocks(doc, page_num, blocks):
    count = 0
    for b in blocks:
        if b["block_type"] in ("image", "mixed") and not b.get("ocr_text"):
            t = ocr_image_block(doc, page_num, b["bbox"])
            if t:
                b["ocr_text"] = t
                count += 1
    return count

def export_to_excel():
    rows = []
    for pg, blks in st.session_state.annotations.items():
        for b in blks:
            if not b.get("label"):
                continue
            rows.append(dict(
                pdf_name=st.session_state.pdf_name,
                page_num=int(pg) + 1,
                block_type=b.get("block_type", "text"),
                bbox=f"({b['bbox'][0]},{b['bbox'][1]},{b['bbox'][2]},{b['bbox'][3]})",
                content=b.get("content", ""),
                font=b.get("font", ""),
                font_size=str(b.get("font_size", "")),
                is_bold=b.get("is_bold", 0),
                is_italic=b.get("is_italic", 0),
                font_color=b.get("font_color", ""),
                label=b.get("label", ""),
                boundary_label=b.get("boundary_label", "O"),
                ocr_text=b.get("ocr_text", ""),
                block_id=b.get("block_id", 0),
            ))
    if rows:
        df = pd.DataFrame(rows)
        out = io.BytesIO()
        df.to_excel(out, index=False, engine="openpyxl")
        stem = Path(st.session_state.pdf_name).stem
        st.download_button("下载Excel", data=out.getvalue(),
            file_name=f"annotations_{stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_excel")
        st.success(f"已导出{len(rows)}条")
    else:
        st.info("暂无标注可导出")

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("文件管理")
    uploaded = st.file_uploader("上传PDF", type=["pdf"], key="fu")
    if uploaded is not None:
        pdf_name = uploaded.name
        pdf_bytes = uploaded.read()
        st.session_state.pdf_bytes = pdf_bytes
        st.session_state.pdf_name = pdf_name
        st.session_state.current_page = 0
        st.session_state.selected_block = None
        st.session_state.annotations = load_progress(pdf_name)
        st.success(f"已加载: {pdf_name} ({len(fitz.open(stream=pdf_bytes, filetype='pdf'))} 页)")
    st.divider()
    st.header("设置")
    zoom = st.slider("渲染缩放", 0.5, 4.0, 2.0, 0.5, key="zoom")
    st.divider()
    st.header("统计")
    total_labeled = sum(
        1 for blocks in st.session_state.annotations.values()
        for b in blocks if b.get("label"))
    st.metric("已标注块", total_labeled)

# ============================================================
# Main Area
# ============================================================
if st.session_state.pdf_bytes is None:
    st.info("请从左侧上传PDF文件开始标注")
    st.stop()

# ---- Execute pending actions (deferred from previous click) ----
action = st.session_state.pending_action
if action:
    st.session_state.pending_action = None
    if action["type"] == "nav_page":
        st.session_state.current_page = action["page"]
        st.session_state.selected_block = None
    elif action["type"] == "nav_block":
        st.session_state.selected_block = action["block"]
    elif action["type"] == "apply_page_label":
        cnt = 0
        for b in st.session_state.annotations.get(str(st.session_state.current_page), []):
            if not b.get("label"):
                b["label"] = action["label"]
                cnt += 1
        save_progress(st.session_state.pdf_name, st.session_state.annotations)
    elif action["type"] == "clear_page":
        for b in st.session_state.annotations.get(str(st.session_state.current_page), []):
            b["label"] = ""
        save_progress(st.session_state.pdf_name, st.session_state.annotations)
    elif action["type"] == "quick_label":
        blocks = st.session_state.annotations.get(str(st.session_state.current_page), [])
        si = action["block_idx"]
        if si < len(blocks):
            blocks[si]["label"] = action["label"]
            save_progress(st.session_state.pdf_name, st.session_state.annotations)
            if si < len(blocks) - 1:
                st.session_state.selected_block = si + 1
    elif action["type"] == "ocr_single":
        blocks = st.session_state.annotations.get(str(st.session_state.current_page), [])
        si = st.session_state.selected_block
        if si is not None and si < len(blocks):
            blocks[si]["ocr_text"] = ocr_image_block(doc_hack, st.session_state.current_page, blocks[si]["bbox"])
            save_progress(st.session_state.pdf_name, st.session_state.annotations)
    elif action["type"] == "ocr_batch":
        blocks = st.session_state.annotations.get(str(st.session_state.current_page), [])
        ocr_all_image_blocks(doc_hack, st.session_state.current_page, blocks)
        save_progress(st.session_state.pdf_name, st.session_state.annotations)
    elif action["type"] == "save":
        save_progress(st.session_state.pdf_name, st.session_state.annotations)
        si = st.session_state.selected_block
        blocks = st.session_state.annotations.get(str(st.session_state.current_page), [])
        if si is not None and si < len(blocks) - 1:
            st.session_state.selected_block = si + 1
    elif action["type"] == "skip":
        si = st.session_state.selected_block
        blocks = st.session_state.annotations.get(str(st.session_state.current_page), [])
        if si is not None and si < len(blocks) - 1:
            st.session_state.selected_block = si + 1
    st.rerun()

# ---- Open doc ----
doc = fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf")
doc_hack = doc  # for use in action handlers above
pdf_name = st.session_state.pdf_name
cp = st.session_state.current_page
tp = len(doc)

# ---- Navigation Bar ----
nc = st.columns([1, 1, 2, 1, 1])
with nc[0]:
    if st.button("首页", use_container_width=True, key="nav_first"):
        st.session_state.pending_action = {"type": "nav_page", "page": 0}
        st.rerun()
with nc[1]:
    if st.button("上一页", use_container_width=True, key="nav_prev"):
        st.session_state.pending_action = {"type": "nav_page", "page": max(0, cp - 1)}
        st.rerun()
with nc[2]:
    pi = st.number_input("页码", 1, tp, cp + 1, key="pi", label_visibility="collapsed")
    if pi != cp + 1:
        st.session_state.pending_action = {"type": "nav_page", "page": pi - 1}
        st.rerun()
with nc[3]:
    if st.button("下一页", use_container_width=True, key="nav_next"):
        st.session_state.pending_action = {"type": "nav_page", "page": min(tp - 1, cp + 1)}
        st.rerun()
with nc[4]:
    if st.button("末页", use_container_width=True, key="nav_last"):
        st.session_state.pending_action = {"type": "nav_page", "page": tp - 1}
        st.rerun()

# ---- Parse Blocks ----
pk = str(cp)
if pk not in st.session_state.annotations:
    st.session_state.annotations[pk] = parse_page_blocks(doc, cp)
blocks = st.session_state.annotations[pk]
if not blocks:
    st.session_state.annotations[pk] = parse_page_blocks(doc, cp)
    blocks = st.session_state.annotations[pk]

labeled = sum(1 for b in blocks if b.get("label"))
total = len(blocks)

# ---- Top Bar: Info + Bulk Label + Boundary ----
st.markdown("---")
top_c1, top_c2, top_c3 = st.columns([2, 2, 2])
with top_c1:
    st.markdown(
        f"**第 {cp+1} / {tp} 页** | "
        f"已标注: **{labeled}** / {total} ({labeled*100//max(total,1)}%)"
    )
with top_c2:
    bl_inner = st.columns([2, 1])
    with bl_inner[0]:
        pl = st.selectbox("整页标记", [""] + LABELS, key="pl",
                          label_visibility="collapsed",
                          placeholder="选择标签...")
    with bl_inner[1]:
        if st.button("应用", key="apply_btn", use_container_width=True,
                     disabled=(pl == "")):
            st.session_state.pending_action = {"type": "apply_page_label", "label": pl}
            st.rerun()
with top_c3:
    bdry_inner = st.columns([2, 1])
    with bdry_inner[0]:
        bl_default = st.selectbox("默认边界标签", BOUNDARY_LABELS, key="default_boundary",
                                  label_visibility="collapsed")
    with bdry_inner[1]:
        if st.button("清除", key="clear_btn", use_container_width=True):
            st.session_state.pending_action = {"type": "clear_page"}
            st.rerun()
st.markdown("---")

# ---- Image + Edit Columns ----
ic, ec = st.columns([3, 2])

with ic:
    si = st.session_state.selected_block
    page_img = render_page_with_boxes(doc, cp, blocks, si, zoom)
    st.image(page_img, use_container_width=True)

with ec:
    st.subheader("块列表")
    if not blocks:
        st.info("此页无块")
    else:
        bdisp = []
        for b in blocks:
            mark = f" [{b['label']}]" if b.get("label") else ""
            prev = (b.get("content") or b.get("ocr_text") or f"[{b['block_type']}]")[:40]
            bdisp.append(f"#{b['block_id']} | {b['block_type']} | {prev}{mark}")

        nc1, nc2 = st.columns([3, 1])
        with nc1:
            ni = st.selectbox("选择块", range(len(blocks)),
                index=si if si is not None else 0,
                format_func=lambda i: bdisp[i][:80], key="bs")
            if ni != si:
                st.session_state.selected_block = ni
                st.rerun()
        with nc2:
            if st.button("上", key="block_up", help="上一块") and si is not None and si > 0:
                st.session_state.selected_block = si - 1
                st.rerun()
            if st.button("下", key="block_down", help="下一块") and si is not None and si < len(blocks) - 1:
                st.session_state.selected_block = si + 1
                st.rerun()

        if si is not None and si < len(blocks):
            block = blocks[si]
            st.divider()
            st.subheader(f"编辑块 #{block['block_id']}")

            block["block_type"] = st.selectbox(
                "类型", ["text", "image", "mixed"],
                index=["text", "image", "mixed"].index(block.get("block_type", "text")),
                key="et")

            st.text_area("内容", block.get("content", ""), height=80,
                        key="ec2", disabled=True)
            st.caption(
                f"字体:{block.get('font','')} 字号:{block.get('font_size','')} "
                f"加粗:{block.get('is_bold',False)} 斜体:{block.get('is_italic',False)} "
                f"颜色:{block.get('font_color','')}")

            if block["block_type"] in ("image", "mixed"):
                block["ocr_text"] = st.text_area(
                    "OCR文字", block.get("ocr_text", ""), height=60, key="ocr")
                o1, o2 = st.columns(2)
                with o1:
                    if st.button("OCR当前块", key="os", use_container_width=True):
                        st.session_state.pending_action = {"type": "ocr_single"}
                        st.rerun()
                with o2:
                    if st.button("OCR整页", key="ob", use_container_width=True):
                        st.session_state.pending_action = {"type": "ocr_batch"}
                        st.rerun()

            lo = [""] + LABELS
            cl = block.get("label", "")
            li = lo.index(cl) if cl in lo else 0
            block["label"] = st.selectbox("章节标签", lo, index=li, key="el")

            cb = block.get("boundary_label", "O")
            bi = BOUNDARY_LABELS.index(cb) if cb in BOUNDARY_LABELS else 0
            block["boundary_label"] = st.selectbox("边界标签", BOUNDARY_LABELS,
                                                   index=bi, key="eb")

            st.caption("快速标签:")
            qc = st.columns(5)
            for i, lbl in enumerate(LABELS):
                if qc[i % 5].button(f"{i+1}.{lbl[:4]}", key=f"q{i}",
                                    use_container_width=True):
                    if si < len(blocks):
                        st.session_state.pending_action = {
                            "type": "quick_label", "label": lbl, "block_idx": si}
                    st.rerun()

            sa, sk = st.columns(2)
            with sa:
                if st.button("保存", key="save_btn", use_container_width=True):
                    st.session_state.pending_action = {"type": "save"}
                    st.rerun()
            with sk:
                if st.button("跳过", key="skip_btn", use_container_width=True):
                    st.session_state.pending_action = {"type": "skip"}
                    st.rerun()

# ============================================================
# Export
# ============================================================
if st.session_state.pdf_name:
    with st.sidebar:
        st.divider()
        st.header("导出")
        export_to_excel()
