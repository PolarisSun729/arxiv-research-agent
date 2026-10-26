"""把论文 PDF 解析成带稳定 ID 的全文块，供快速问答整篇送入模型。

解析用 pymupdf4llm 的版面模式（pymupdf-layout 识别正文、标题、表格、图、公式等区域），
不依赖论文来源，任何文本型 PDF 都适用。输出按阅读顺序排列的块：

- 正文、标题、列表、表格、公式、图注等：ID 为 ``p{页码}-{页内序号}``，如 ``p3-2``；
- 图片：ID 为 ``fig-{全文序号}``，文字取同页最近的图注；
- 图片、公式和表格结构识别失败（没有 Markdown）的区域另存为 JPEG，供多模态输入和前端回放。

页眉页脚不进入结果。OCR 关闭：扫描版 PDF 没有文本层，会得到空结果，由调用方报错。

本模块的 parse_pdf_fulltext 在隔离子进程中执行（见 isolated_subprocess），只能依赖可 pickle 的参数和返回值。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# 解析器或清理规则变化会改变块的切分和 ID，必须同步升级，让旧缓存失效。
FULLTEXT_SCHEMA_VERSION = 1

_SKIPPED_BOX_CLASSES = {"page-header", "page-footer"}
_TEXT_KIND_BY_BOX_CLASS = {
    "title": "heading",
    "section-header": "heading",
    "list-item": "list",
    "footnote": "footnote",
    "caption": "caption",
    "table": "table",
    "formula": "formula",
}

# 斜体/粗体标记把数字里的小数点单独包住：``1_._293``、``0**.**5`` → ``1.293``、``0.5``。
_STYLED_NUMBER_SEPARATOR = re.compile(r"(?<=\d)(?:_|\*\*)([.,])(?:_|\*\*)(?=\d)")


def clean_markdown_text(text: str) -> str:
    """清理版面解析的 Markdown 残留；只做不会改变语义的替换。"""
    cleaned = _STYLED_NUMBER_SEPARATOR.sub(r"\1", str(text or ""))
    return cleaned.strip()


def parse_pdf_fulltext(pdf_path: str, image_dir: str, image_dpi: int) -> Dict[str, Any]:
    """解析 PDF 并把图片写入 image_dir，返回可 JSON 序列化的全文结构。"""
    # 导入 pymupdf.layout 才会启用版面模型；放在函数内，父进程不用加载 onnxruntime。
    import pymupdf
    import pymupdf.layout  # noqa: F401
    from pymupdf4llm.helpers import document_layout as layout_helpers
    from pymupdf4llm.ocr import OCRMode

    parsed = layout_helpers.parse_document(
        pdf_path,
        embed_images=True,
        image_dpi=image_dpi,
        image_format="jpg",
        use_ocr=OCRMode.NEVER,
        force_text=False,
    )
    output_dir = Path(image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks: List[Dict[str, Any]] = []
    figure_count = 0
    # 表格结构识别失败时 parse_document 只给空 Markdown，需要自己按区域截图。
    document = pymupdf.open(pdf_path)
    try:
        for page in parsed.pages:
            page_number = int(page.page_number)
            list_levels = layout_helpers.create_list_item_levels(page.boxes)
            captions = [box for box in page.boxes if box.boxclass == "caption"]
            block_index = 0
            for box_index, box in enumerate(page.boxes):
                box_class = str(box.boxclass or "")
                if box_class in _SKIPPED_BOX_CLASSES:
                    continue
                if box_class == "picture":
                    if not isinstance(box.image, bytes):
                        continue
                    figure_count += 1
                    block_id = f"fig-{figure_count}"
                    caption = _nearest_caption_text(box, captions, layout_helpers)
                    blocks.append(_image_block(block_id, page_number, "figure", caption, box.image, output_dir))
                    continue

                block_index += 1
                block_id = f"p{page_number}-{block_index}"
                kind = _TEXT_KIND_BY_BOX_CLASS.get(box_class, "text")
                text = _box_markdown(box, box_class, list_levels.get(box_index, 1), layout_helpers)
                image = box.image if box_class == "formula" else None
                if box_class == "table" and not text:
                    image = _render_region(document[page_number - 1], box, image_dpi)
                if isinstance(image, bytes):
                    blocks.append(_image_block(block_id, page_number, kind, text, image, output_dir))
                elif text:
                    blocks.append({"id": block_id, "page": page_number, "kind": kind, "text": text})
    finally:
        document.close()

    return {
        "schema_version": FULLTEXT_SCHEMA_VERSION,
        "page_count": int(parsed.page_count or 0),
        "blocks": blocks,
        "char_count": sum(len(block["text"]) for block in blocks),
        "image_count": sum(1 for block in blocks if block.get("image")),
    }


def _image_block(block_id: str, page: int, kind: str, text: str, image: bytes, output_dir: Path) -> Dict[str, Any]:
    filename = f"{block_id}.jpg"
    (output_dir / filename).write_bytes(image)
    return {"id": block_id, "page": page, "kind": kind, "text": text, "image": filename}


def _render_region(page: Any, box: Any, image_dpi: int) -> Optional[bytes]:
    import pymupdf

    clip = pymupdf.Rect(box.x0, box.y0, box.x1, box.y1)
    if clip.is_empty:
        return None
    return page.get_pixmap(clip=clip, dpi=image_dpi).tobytes("jpg")


def _box_markdown(box: Any, box_class: str, list_level: int, layout_helpers: Any) -> str:
    """按块类型渲染 Markdown；图片和公式区域没有文字行，返回空串。"""
    if box_class == "table" and isinstance(box.table, dict):
        return clean_markdown_text(box.table.get("markdown") or "")
    textlines = box.textlines or []
    if not textlines:
        return ""
    if box_class == "title":
        rendered = layout_helpers.title_to_md(box.header_level or 1, textlines)
    elif box_class == "section-header":
        rendered = layout_helpers.section_hdr_to_md(box.header_level or 2, textlines)
    elif box_class == "list-item":
        rendered = layout_helpers.list_item_to_md(textlines, list_level)
    elif box_class == "footnote":
        rendered = layout_helpers.footnote_to_md(textlines)
    else:
        rendered = layout_helpers.text_to_md(textlines)
    return clean_markdown_text(rendered)


def _nearest_caption_text(picture: Any, captions: List[Any], layout_helpers: Any) -> str:
    """取同页与图片垂直距离最近、且水平方向有重叠的图注；没有就返回空串。"""
    best: Optional[Any] = None
    best_gap = float("inf")
    for caption in captions:
        if min(picture.x1, caption.x1) <= max(picture.x0, caption.x0):
            continue
        gap = max(caption.y0 - picture.y1, picture.y0 - caption.y1, 0.0)
        if gap < best_gap:
            best, best_gap = caption, gap
    if best is None or not best.textlines:
        return ""
    return clean_markdown_text(layout_helpers.text_to_md(best.textlines))
