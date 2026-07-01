"""
医疗器械体系文件审核 - 文档处理模块
支持 Word (.docx) 和 PDF 文件的文本提取与分块
输出结构化 Markdown，保留标题层级、表格、列表
"""
import os
import re
import json
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Any, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


# ============== Word 标题样式到 Markdown 层级映射 ==============
HEADING_STYLE_MAP = {
    'heading 1': 1, 'heading 2': 2, 'heading 3': 3,
    'heading 4': 4, 'heading 5': 5, 'heading 6': 6,
    '标题 1': 1, '标题 2': 2, '标题 3': 3,
    '标题 4': 4, '标题 5': 5, '标题 6': 6,
    'title': 1,
}


def _detect_heading_level(para) -> int:
    """
    检测段落的大纲层级，按优先级依次尝试三种策略。

    策略1: Word 原生 outline_level（最可靠，是 Word 生成目录的依据）
    策略2: 样式名映射（HEADING_STYLE_MAP，兼容自定义样式）
    策略3: 字体大小启发式（回退方案，大字 = 标题）

    Returns:
        0-6 的标题层级，0 表示非标题正文
    """
    # 策略1: Word 原生大纲层级
    # outline_level 是 Word 内部用于生成目录结构的属性，不依赖样式名语言
    try:
        pf = para.paragraph_format
        if pf is not None:
            ol = pf.outline_level
            if ol is not None and isinstance(ol, int) and 1 <= ol <= 6:
                return ol
    except Exception:
        pass

    # 策略2: 样式名映射
    style_name = (para.style.name or '').lower() if para.style else ''
    mapped = HEADING_STYLE_MAP.get(style_name, 0)
    if mapped > 0:
        return mapped

    # 策略3: 字体大小启发式（样式名不匹配时，用字体大小推断）
    try:
        from docx.shared import Pt
        font = None
        for run in para.runs:
            if run.font.size:
                font = run.font.size
                break
        if font is not None:
            size_pt = font.pt
            if size_pt >= 22:
                return 1
            elif size_pt >= 18:
                return 2
            elif size_pt >= 15:
                return 3
            elif size_pt >= 13:
                return 4
    except Exception:
        pass

    return 0


def extract_text_from_docx(file_path: str) -> str:
    """
    从 Word 文档提取结构化 Markdown 文本

    保留标题层级（#/##/###）、表格、列表，供 LLM 理解文档结构

    Args:
        file_path: .docx 文件路径

    Returns:
        结构化 Markdown 文本
    """
    try:
        from docx import Document
        doc = Document(file_path)
        parts = []

        # 构建 element -> paragraph 的快速查找映射（O(1) 替代原 O(n²) 循环）
        elem_to_para = {}
        for p in doc.paragraphs:
            try:
                elem_to_para[id(p._element)] = p
            except Exception:
                pass

        for element in doc.element.body:
            tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag

            if tag == 'p':
                # 段落处理 - O(1) 查找
                para = elem_to_para.get(id(element))
                if para is None:
                    continue

                text = para.text.strip()
                if not text:
                    parts.append('')
                    continue

                # 检测标题层级（优先级：outline_level > 样式名 > 字体大小）
                heading_level = _detect_heading_level(para)
                style_name = (para.style.name or '').lower() if para.style else ''

                if heading_level > 0:
                    parts.append(f'{"#" * heading_level} {text}')
                elif _is_list_paragraph(para):
                    parts.append(_format_list_item(para, text))
                else:
                    parts.append(text)

            elif tag == 'tbl':
                # 表格处理
                table_md = _extract_table_from_element(element, doc)
                if table_md:
                    parts.append('')
                    parts.append(table_md)
                    parts.append('')

        return '\n'.join(parts)
    except ImportError:
        raise ImportError("请安装 python-docx: pip install python-docx")


def _is_list_paragraph(para) -> bool:
    """判断段落是否为列表项"""
    style_name = (para.style.name or '').lower() if para.style else ''
    list_keywords = ['list', 'listparagraph', '列表']
    return any(kw in style_name for kw in list_keywords)


def _format_list_item(para, text: str) -> str:
    """格式化列表项为 Markdown"""
    style_name = (para.style.name or '').lower() if para.style else ''
    # 判断有序/无序
    if 'number' in style_name or re.match(r'^\d+[.、）)]', text):
        # 有序列表：提取数字前缀或自动编号
        match = re.match(r'^(\d+)[.、）)]\s*', text)
        if match:
            return f"{match.group(1)}. {text[match.end():]}"
        return f"1. {text}"
    else:
        return f"- {text}"


def _extract_table_from_element(table_element, doc) -> str:
    """从 XML 元素提取表格并转为 Markdown 格式"""
    rows = []
    for row_elem in table_element.iterchildren():
        tag = row_elem.tag.split('}')[-1] if '}' in row_elem.tag else row_elem.tag
        if tag != 'tr':
            continue
        cells = []
        for cell_elem in row_elem.iterchildren():
            cell_tag = cell_elem.tag.split('}')[-1] if '}' in cell_elem.tag else cell_elem.tag
            if cell_tag != 'tc':
                continue
            # 提取单元格文本
            cell_text_parts = []
            for p_elem in cell_elem.iterchildren():
                p_tag = p_elem.tag.split('}')[-1] if '}' in p_elem.tag else p_elem.tag
                if p_tag == 'p':
                    texts = []
                    for t_elem in p_elem.iter():
                        t_tag = t_elem.tag.split('}')[-1] if '}' in t_elem.tag else t_elem.tag
                        if t_tag == 't':
                            texts.append(t_elem.text or '')
                    cell_text_parts.append(''.join(texts).strip())
            cells.append(' | '.join(cell_text_parts) if cell_text_parts else ' ')

        if cells:
            rows.append(cells)

    if not rows:
        return ''

    # 统一列数
    max_cols = max(len(r) for r in rows) if rows else 0
    for r in rows:
        while len(r) < max_cols:
            r.append(' ')

    # 构建 Markdown 表格
    md_lines = []
    # 表头
    md_lines.append('| ' + ' | '.join(rows[0]) + ' |')
    # 分隔行
    md_lines.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
    # 数据行
    for row in rows[1:]:
        md_lines.append('| ' + ' | '.join(row) + ' |')

    return '\n'.join(md_lines)


def extract_text_from_pdf(file_path: str) -> str:
    """
    从 PDF 文件提取文本内容，尝试用字体大小识别标题层级

    Args:
        file_path: .pdf 文件路径

    Returns:
        结构化 Markdown 文本
    """
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue

                # 尝试提取字体信息来识别标题
                chars = page.chars
                if chars:
                    text = _pdf_text_with_headings(text, chars)

                text_parts.append(text)
        return '\n'.join(text_parts)
    except ImportError:
        raise ImportError("请安装 pdfplumber: pip install pdfplumber")


def _pdf_text_with_headings(text: str, chars: list) -> str:
    """
    根据 PDF 字符的字体大小，启发式识别标题并转为 Markdown

    逻辑：统计所有字体大小，最大的视为标题层级
    """
    if not chars:
        return text

    # 统计字体大小分布
    size_count = {}
    for c in chars:
        size = round(c.get('size', 0), 1)
        if size > 0:
            size_count[size] = size_count.get(size, 0) + len(c.get('text', ''))

    if not size_count:
        return text

    # 按大小降序排列，前3个尺寸视为标题
    sorted_sizes = sorted(size_count.keys(), reverse=True)
    body_size = sorted_sizes[-1] if sorted_sizes else 10.0  # 最小的通常是正文

    heading_map = {}
    heading_level = 1
    for size in sorted_sizes:
        if size > body_size * 1.15 and heading_level <= 3:  # 比正文大15%以上视为标题
            heading_map[size] = heading_level
            heading_level += 1
        else:
            break

    if not heading_map:
        return text

    # 按行处理，识别标题行
    lines = text.split('\n')
    result = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            result.append('')
            continue

        # 检查该行是否主要是大字体
        line_chars = [c for c in chars if c.get('text', '').strip()
                      and c['text'].strip() in line_stripped]
        if line_chars:
            # 取该行最多的字体大小
            line_sizes = {}
            for c in line_chars:
                s = round(c.get('size', 0), 1)
                line_sizes[s] = line_sizes.get(s, 0) + 1
            dominant_size = max(line_sizes, key=line_sizes.get) if line_sizes else 0

            if dominant_size in heading_map:
                level = heading_map[dominant_size]
                result.append(f'{"#" * level} {line_stripped}')
                continue

        result.append(line_stripped)

    return '\n'.join(result)


def extract_text_from_doc(file_path: str) -> str:
    """
    从旧版 Word .doc 文件提取文本内容
    优先使用 win32com (COM自动化)，回退到 olefile
    """
    # 方法1: win32com (需要 Windows + MS Word)
    try:
        import win32com.client
        import pythoncom
        pythoncom.CoInitialize()
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        try:
            abs_path = os.path.abspath(file_path)
            doc = word.Documents.Open(abs_path, ReadOnly=True)
            text = doc.Content.Text
            doc.Close(False)
            word.Quit()
            pythoncom.CoUninitialize()
            if text and text.strip():
                return text.strip()
        except Exception:
            try:
                word.Quit()
            except Exception:
                pass
            pythoncom.CoUninitialize()
    except Exception:
        pass

    # 方法2: olefile 提取
    try:
        import olefile
        ole = olefile.OleFileIO(file_path)
        for stream_name in ['1Table', '0Table', 'WordDocument']:
            if ole.exists(stream_name):
                data = ole.openstream(stream_name).read()
                try:
                    text = data.decode('utf-16le', errors='ignore')
                    clean = ''.join(c for c in text if c.isprintable() or c in '\n\r\t')
                    if len(clean) > 50:
                        ole.close()
                        return clean
                except Exception:
                    pass
        ole.close()
    except Exception:
        pass

    # 方法3: 二进制扫描
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        text_parts = []
        i = 0
        while i < len(data) - 1:
            char_code = data[i] | (data[i+1] << 8)
            if 0x4e00 <= char_code <= 0x9fff or 0x3000 <= char_code <= 0x303f or char_code in (0x000d, 0x000a):
                text_parts.append(chr(char_code))
                i += 2
            elif 0x0020 <= char_code <= 0x007e:
                text_parts.append(chr(char_code))
                i += 2
            else:
                if text_parts and text_parts[-1] != '\n':
                    text_parts.append('\n')
                i += 1
        result = ''.join(text_parts).strip()
        if len(result) > 50:
            return result
    except Exception:
        pass

    return ""


def extract_text_from_txt(file_path: str) -> str:
    """从文本文件提取内容，自动检测编码"""
    for enc in ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


def extract_text(file_path: str) -> str:
    """
    根据文件扩展名自动识别并提取文本（结构化 Markdown 格式）

    Args:
        file_path: 文件路径

    Returns:
        结构化 Markdown 文本
    """
    ext = Path(file_path).suffix.lower()
    if ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext == '.doc':
        return extract_text_from_doc(file_path)
    elif ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.txt':
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}，仅支持 .docx, .doc, .pdf 和 .txt")


def split_by_markdown_headers(text: str) -> List[Tuple[str, str, int]]:
    """
    按 Markdown 标题层级分割文档

    Args:
        text: Markdown 格式的文档文本

    Returns:
        [(标题, 内容, 层级), ...] 段落列表
        层级: 1=一级标题, 2=二级标题, 3=三级标题...
    """
    sections = []
    current_title = "文档开头"
    current_level = 0
    current_content = []

    for line in text.split('\n'):
        # 检测 Markdown 标题行
        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if match:
            # 保存前一个段落
            if current_content:
                full_content = '\n'.join(current_content).strip()
                if full_content:
                    sections.append((current_title, full_content, current_level))

            current_level = len(match.group(1))
            current_title = match.group(2).strip()
            current_content = []
        else:
            current_content.append(line)

    # 保存最后一个段落
    if current_content:
        full_content = '\n'.join(current_content).strip()
        if full_content:
            sections.append((current_title, full_content, current_level))

    # 如果没有检测到任何标题，把整篇文档作为一个段落
    if not sections and text.strip():
        sections.append(("完整文档", text.strip(), 0))

    return sections


# ============== 编号式标题正则（多策略大纲解析使用）==============
# 顺序: 优先匹配更具体的格式，避免误判
# 层级设计原则：
#   - "第X章" 和 阿拉伯数字一级编号 → L1（最高级）
#   - "X.X" → L2（与 Markdown ## 同级，作为审核单元）
#   - "X.X.X" / 中文 "一、" → L3（合并到父二级小节内）
#   - "X.X.X.X" / "(一)" / "(1)" → L4（更深层，合并）
# 这样在与 Markdown # ## ### 混合出现时，编号标题通常会成为子节而非平级
_NUMBERING_PATTERNS = [
    # "第X章" / "第X篇" / "第X部分" → level 1
    (re.compile(r'^(第[一二三四五六七八九十百千零〇\d]+[章篇部分])[\s::、]*(.*)$'), 1),
    # "X.X.X.X" 四级编号 → level 4
    (re.compile(r'^(\d+\.\d+\.\d+\.\d+)[\s::、)]*(.+)$'), 4),
    # "X.X.X" 三级编号 → level 3
    (re.compile(r'^(\d+\.\d+\.\d+)[\s::、)]*(.+)$'), 3),
    # "X.X" 二级编号 → level 2
    (re.compile(r'^(\d+\.\d+)[\s::、)]*(.+)$'), 2),
    # "X、" 或 "X." 一级数字编号 → level 1（前提：后面有中文/字母标题文字，且长度合理）
    (re.compile(r'^(\d+)[、.)][\s]+([^\d].{1,80})$'), 1),
    # "一、二、三、" 中文数字 → level 3（通常出现在 ## 小节内，作为列举项；不抢占 L2 审核单元位置）
    (re.compile(r'^([一二三四五六七八九十]+)[、.)][\s]*(.{1,80})$'), 3),
    # "(一)(二)" 或 "（一）（二）" 中文带括号 → level 4
    (re.compile(r'^[\(（]([一二三四五六七八九十]+)[\)）][\s]*(.{1,80})$'), 4),
    # "(1)(2)" 或 "（1）（2）" 数字带括号 → level 4
    (re.compile(r'^[\(（](\d+)[\)）][\s]*(.{1,80})$'), 4),
]


def _detect_numbering_heading(line: str) -> Tuple[int, str]:
    """
    检测一行是否为编号式标题。

    Returns:
        (level, title) — level=0 表示非标题；title 为标题文字（含编号前缀）
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        # 标题一般不会太长（>120字符基本是正文）
        return 0, ""
    # 已经是 Markdown 标题的不再二次识别
    if stripped.startswith('#'):
        return 0, ""

    for pattern, level in _NUMBERING_PATTERNS:
        if pattern.match(stripped):
            return level, stripped
    return 0, ""


def parse_document_outline(text: str) -> List[dict]:
    """
    多策略大纲解析：识别文档的章节树状结构。

    策略融合（按优先级）：
    1. Markdown 标题（# / ## / ### ...）— 来自 Word Heading 样式或显式 Markdown
    2. 编号正则（"第X章" / "X.X" / "X.X.X" / "一、" / "(一)" / "(1)" 等）

    Returns:
        树状大纲列表，每个节点：
        {
            "title": str,           # 标题文字
            "level": int,           # 层级 1~6
            "content": str,         # 该标题下的直接正文（不含子节点的正文）
            "children": List[dict]  # 子节点
        }
    """
    # 第一步：扫描每一行，识别"标题行"和"正文行"
    # 标题行: {"is_heading": True, "level": int, "title": str}
    # 正文行: {"is_heading": False, "text": str}
    items = []
    for raw_line in text.split('\n'):
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            items.append({"is_heading": False, "text": ""})
            continue

        # 策略1: Markdown 标题
        md_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if md_match:
            items.append({
                "is_heading": True,
                "level": len(md_match.group(1)),
                "title": md_match.group(2).strip()
            })
            continue

        # 策略2: 编号式标题
        num_level, num_title = _detect_numbering_heading(line)
        if num_level > 0:
            items.append({
                "is_heading": True,
                "level": num_level,
                "title": num_title
            })
            continue

        items.append({"is_heading": False, "text": line})

    # 第二步：把扁平的 items 按层级组装成树
    root_children: List[dict] = []
    # 用栈维护"当前层级路径"，stack[-1] 是最近的祖先节点
    stack: List[dict] = []
    # 一个虚拟根，方便统一处理
    pending_preface: List[str] = []  # 第一个标题之前的正文

    def _new_node(level: int, title: str) -> dict:
        return {"title": title, "level": level, "content": "", "children": []}

    for item in items:
        if item["is_heading"]:
            node = _new_node(item["level"], item["title"])
            # 找到合适的父：栈中第一个 level < 当前 level 的
            while stack and stack[-1]["level"] >= node["level"]:
                stack.pop()
            if stack:
                stack[-1]["children"].append(node)
            else:
                root_children.append(node)
            stack.append(node)
        else:
            # 正文行 → 加到当前栈顶节点的 content；若栈空则放 pending_preface
            text_line = item.get("text", "")
            if stack:
                if stack[-1]["content"]:
                    stack[-1]["content"] += "\n" + text_line
                else:
                    stack[-1]["content"] = text_line
            else:
                pending_preface.append(text_line)

    # 第三步：如果存在 preface（文档开头未归属任何标题的正文），合成一个虚拟节点
    preface_text = "\n".join(pending_preface).strip()
    if preface_text:
        root_children.insert(0, {
            "title": "文档开头",
            "level": 1,
            "content": preface_text,
            "children": []
        })

    # 第四步：清理每个节点 content 的首尾空白
    def _clean(node: dict):
        node["content"] = node["content"].strip()
        for child in node["children"]:
            _clean(child)
    for n in root_children:
        _clean(n)

    return root_children


def _render_subtree_as_content(node: dict, base_level: int = 2) -> str:
    """
    把一个节点及其所有子节点（三级及更深）的内容渲染为合并文本。
    用于"二级小节"作为审核单元时，把它下面的三级/四级内容并入。

    Args:
        node: 起始节点
        base_level: 起始层级（用于决定子节点用几个 # 做副标题）

    Returns:
        合并后的文本（保留子层级的小标题，便于 LLM 理解结构）
    """
    parts: List[str] = []
    if node.get("content"):
        parts.append(node["content"])

    for child in node.get("children", []):
        sub_level = child.get("level", base_level + 1)
        # 渲染为副标题前缀（用 # 标记，便于 LLM 识别）
        hashes = "#" * min(sub_level, 6)
        parts.append("")
        parts.append(f"{hashes} {child['title']}")
        sub_text = _render_subtree_as_content(child, sub_level)
        if sub_text:
            parts.append(sub_text)

    return "\n".join(p for p in parts if p is not None).strip()


def flatten_to_audit_units(outline: List[dict], audit_granularity: int = 3) -> List[Tuple[str, str, int, str]]:
    """
    把树状大纲扁平化为审核单元列表。

    规则（逐小节审核）:
    - audit_granularity: 目标审核层级深度，默认 3 表示以三级小节为最小审核单元
    - 对于每个节点：如其有下一级子节点且下一级层级 ≤ audit_granularity，则深入子节点
    - 如无下一级子节点或下一级层级 > audit_granularity，则节点本身作为审核单元，
      其下所有子孙内容合并到该审核单元中
    - 父节点的直接 content（引言/概述）会作为上下文前缀传递给第一个子节点

    Args:
        outline: parse_document_outline() 的返回值
        audit_granularity: 审核粒度层级，2=逐节(原行为), 3=逐小节(默认), 4=逐小小节

    Returns:
        [(title, content, level, breadcrumb), ...]
        breadcrumb: "第一章 / 1.1 标题 / 1.1.1 小节" 形式的面包屑，便于 LLM 理解上下文
    """
    units: List[Tuple[str, str, int, str]] = []

    def _walk_node(node: dict, breadcrumb_parts: List[str], inherited_preface: str):
        """递归遍历树节点，在目标粒度层级生成审核单元"""
        node_level = node.get("level", 1)
        node_title = node["title"]
        node_content = node.get("content", "").strip()
        children = node.get("children", [])

        # 找到直接下一级的子节点
        next_level = node_level + 1
        direct_children = [c for c in children if c.get("level") == next_level]

        if direct_children and next_level <= audit_granularity:
            # 当前节点在审核粒度之上 → 深入子节点
            current_breadcrumb = breadcrumb_parts + [node_title]

            for i, child in enumerate(direct_children):
                # 将父节点引言仅传递给第一个子节点
                child_preface = ""
                if i == 0:
                    preface_parts = []
                    if inherited_preface:
                        preface_parts.append(inherited_preface)
                    if node_content:
                        preface_parts.append(f"[{node_title}引言]\n{node_content}")
                    child_preface = "\n\n".join(preface_parts).strip()
                _walk_node(child, current_breadcrumb, child_preface)
        else:
            # 当前节点达到或超过审核粒度 → 作为独立审核单元
            content_parts = []
            if inherited_preface:
                content_parts.append(inherited_preface)
            if node_content:
                content_parts.append(node_content)

            # 合并所有子节点的内容
            for child in children:
                hashes = "#" * min(child.get("level", node_level + 1), 6)
                content_parts.append(f"\n{hashes} {child['title']}")
                sub = _render_subtree_as_content(child, child.get("level", node_level + 1))
                if sub:
                    content_parts.append(sub)

            breadcrumb = " / ".join(breadcrumb_parts + [node_title])
            merged = "\n".join(p for p in content_parts if p).strip()
            if merged or children:
                units.append((node_title, merged, node_level, breadcrumb))

    for top_node in outline:
        _walk_node(top_node, [], "")

    return units


# ============== LLM 驱动的文档大纲识别 ==============

_OUTLINE_SYSTEM_PROMPT = """你是医疗器械文档结构分析专家。任务：从给定文档中识别正文的章节(chapter)与小节(subsection)结构。

严格规则：
1. 跳过封面信息表、修订记录表、目录(TOC)、签字/审批栏等非正文内容
   - TOC 行的典型特征：行尾是页码数字（含制表符或空格分隔）
2. chapter 是文档主体的一级章节（如 "1. 产品概述"、"第一章 范围"、"3 功能需求"）
3. subsection 是 chapter 下的二级/三级小节（如 "1.1 产品名称"、"3.2.1 报警机制"）
4. 没有子小节的 chapter，把整章正文作为该章的唯一一个 subsection（subsection 标题同 chapter 标题）
5. 每个节点必须给出 start_line（标题所在行号）与 end_line（该节点正文最后一行行号），均为输入文档中的行号
6. 仅输出 JSON 对象，禁止任何其他文字、Markdown 代码块标记或解释

JSON 输出格式：
{
  "chapters": [
    {
      "title": "1. 产品概述",
      "start_line": 50,
      "end_line": 70,
      "subsections": [
        {"title": "1.1 产品名称", "start_line": 51, "end_line": 53},
        {"title": "1.2 预期用途", "start_line": 54, "end_line": 70}
      ]
    }
  ]
}"""


def _safe_slice_lines(lines: list, start: int, end: int) -> str:
    """安全切片：clamp 行号边界，避免 LLM 给出越界 line_no 时崩溃"""
    n = len(lines)
    s = max(0, min(start, n))
    e = max(s, min(end, n - 1))
    return "\n".join(lines[s:e + 1])


def _strip_code_fence(text: str) -> str:
    """去除 LLM 返回中可能包裹的 ```json ... ``` 围栏"""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _repair_malformed_json(text: str) -> str:
    """
    修复 LLM 返回的、字符串内含未转义字符的 JSON。

    LLM (尤其是中文场景下的 deepseek-v4-pro) 经常在 JSON 字符串值里直接写入
    ASCII 双引号 (例如 "title": "1.1 "产品" 概述")，导致标准 json.loads 失败。
    本函数以状态机方式扫描:
      - 字符串外的字符原样保留
      - 字符串内的 ASCII "  若是合法结束 (后跟 , : } ] 或 EOF) → 保留
      - 字符串内的 ASCII "  否则视为嵌入引号 → 替换为全角 "" (左/右交替)
      - 字符串内的原始 \\n / \\r / \\t → 转义为 \\\\n \\\\r \\\\t

    对合法 JSON 是恒等变换 (不会改动正常输出)。
    """
    out: list = []
    in_string = False
    escape = False
    n = len(text)
    # 交替使用左右全角引号 ("" / "")，让结果更可读
    left_quote, right_quote = "“", "”"
    pending_quote = left_quote
    i = 0
    while i < n:
        ch = text[i]
        if escape:
            out.append(ch)
            escape = False
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            escape = True
            i += 1
            continue
        if ch == '"':
            if not in_string:
                in_string = True
                out.append(ch)
                i += 1
                continue
            # 当前已在字符串内：判断是结束还是嵌入
            j = i + 1
            while j < n and text[j] in " \t":
                j += 1
            nxt = text[j] if j < n else ""
            if nxt in (",", ":", "}", "]", ""):
                # 合法结束引号
                in_string = False
                out.append(ch)
            else:
                # 嵌入引号 → 替换为全角引号
                out.append(pending_quote)
                pending_quote = right_quote if pending_quote == left_quote else left_quote
            i += 1
            continue
        if in_string and ch == "\n":
            out.append("\\n")
            i += 1
            continue
        if in_string and ch == "\r":
            out.append("\\r")
            i += 1
            continue
        if in_string and ch == "\t":
            out.append("\\t")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _log_outline_failure(raw: str, cleaned: str, error: Exception, repaired: str = "") -> None:
    """
    将 LLM 大纲解析失败的原始输出落盘到 backend/logs/llm_outline_failures.log，
    便于事后分析模型行为 (哪些字符易触发未转义引号)。
    """
    try:
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "llm_outline_failures.log"
        # 截断过长输出避免日志爆炸
        head = raw[:1500]
        tail = raw[-500:] if len(raw) > 2000 else ""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"\n{'=' * 80}\n"
                f"[{ts}] {type(error).__name__}: {error}\n"
                f"--- raw (head 1500) ---\n{head}\n"
            )
            if tail:
                f.write(f"--- raw (tail 500) ---\n{tail}\n")
            f.write(
                f"--- cleaned (head 800) ---\n{cleaned[:800]}\n"
            )
            if repaired:
                f.write(
                    f"--- repaired (head 800) ---\n{repaired[:800]}\n"
                )
    except Exception as log_err:
        # 日志失败不应影响主流程
        logger.warning(f"_log_outline_failure 写日志失败: {log_err}")


# 候选章节行模式：与 parse_document_outline 标题识别保持一致
_OUTLINE_HEADING_PATTERNS = [
    re.compile(r"^#{1,6}\s+\S"),                                  # Markdown 标题
    re.compile(r"^第[一二三四五六七八九十百千零〇\d]+[章节部分篇]"),  # 第X章 / 第1章
    re.compile(r"^[一二三四五六七八九十]+[、.\s]\s*\S"),            # 一、 / 一.
    re.compile(r"^\d+(\.\d+){0,4}\s*\S"),                          # 1, 1.1, 1.1.1
    re.compile(r"^\(\s*[一二三四五六七八九十\d]+\s*\)\s*\S"),       # (一) / (1)
]

# 抽取 prompt 时使用的最大行数（防止 LLM 输入过长导致超时）
_OUTLINE_MAX_LINES = 1500
# 抽取 prompt 时使用的最大字节数（约 1 万个中文字符，避免触发 LLM 长文本慢响应）
_OUTLINE_MAX_BYTES = 200_000

# 单段调用 LLM 时的阈值（候选行较少时无需切分）
_OUTLINE_SEGMENT_MAX_LINES = 400
_OUTLINE_SEGMENT_MAX_BYTES = 80_000
# 相邻段之间保留的重叠候选行数（避免章节被切碎）
_OUTLINE_SEGMENT_OVERLAP_LINES = 20
# 单段调用最大重试次数（仅对超时异常重试）
_OUTLINE_SEGMENT_MAX_ATTEMPTS = 3


def _extract_outline_candidates(document_text: str) -> tuple:
    """
    从完整文档中抽取"看起来像章节标题"的行（含原行号），返回带 [NNNN] 前缀的文本。

    目的：把传给 LLM 的 prompt 体积从整篇文档（动辄 5-15 万字符）压缩到几百行，
    显著降低 LLM 响应时间与超时概率。LLM 只需要识别"哪些行是章节起始"即可，
    正文内容由后续 _safe_slice_lines 按行号切片得到，不依赖 LLM 看正文。

    策略：
    1. 始终保留前 5 行（封面/标题）与最后 5 行（签字栏）
    2. 匹配章节标题模式的行全部保留
    3. 候选行太少（<30）→ 文档结构不典型，回退为截取前 _OUTLINE_MAX_LINES 行
    4. 候选行数 > _OUTLINE_MAX_LINES 或总字节 > _OUTLINE_MAX_BYTES → 截断
    """
    lines = document_text.split("\n")
    n = len(lines)

    keep_indices: set = set()
    keep_indices.update(range(min(5, n)))
    if n > 10:
        keep_indices.update(range(max(0, n - 5), n))

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if any(p.match(stripped) for p in _OUTLINE_HEADING_PATTERNS):
            keep_indices.add(i)

    # 候选过少 → 文档无明显章节结构，保留全部但限制行数
    if len(keep_indices) < 30:
        selected = list(range(min(n, _OUTLINE_MAX_LINES)))
    else:
        selected = sorted(keep_indices)
        if len(selected) > _OUTLINE_MAX_LINES:
            selected = selected[:_OUTLINE_MAX_LINES]

    numbered = "\n".join(f"[{i:04d}] {lines[i]}" for i in selected)
    if len(numbered.encode("utf-8")) > _OUTLINE_MAX_BYTES:
        # 按字节硬截断（按行号顺序丢弃尾部）
        truncated_bytes = b""
        kept = []
        for i in selected:
            seg = f"[{i:04d}] {lines[i]}\n".encode("utf-8")
            if len(truncated_bytes) + len(seg) > _OUTLINE_MAX_BYTES:
                break
            truncated_bytes += seg
            kept.append(i)
        numbered = truncated_bytes.decode("utf-8", errors="ignore")
        selected = kept

    return numbered, selected, n


def _recompute_chapter_end_lines(chapters_raw: list, total_lines: int) -> list:
    """
    按 "next chapter start - 1" 的规则重算每个 chapter 与 subsection 的 end_line，
    让章节范围在代码层可控，不依赖 LLM 给出的 end_line 是否准确。

    规则：
    - end_line = min(原end_line, 下一节点.start - 1, total_lines - 1)
    - 最后一个节点的 end_line = total_lines - 1
    """
    if not chapters_raw:
        return chapters_raw

    # 收集所有节点的 (start, ref) 一次性排序
    flat = []
    for ci, chap in enumerate(chapters_raw):
        c_start = int(chap.get("start_line", 0))
        flat.append((c_start, ("chap", ci, None)))
        for si, sub in enumerate(chap.get("subsections") or []):
            s_start = int(sub.get("start_line", 0))
            flat.append((s_start, ("sub", ci, si)))
    flat.sort(key=lambda x: (x[0], 0 if x[1][0] == "chap" else 1))

    next_starts = {}
    for idx, (s, ref) in enumerate(flat):
        if idx + 1 < len(flat):
            next_starts[ref] = flat[idx + 1][0] - 1
        else:
            next_starts[ref] = total_lines - 1

    # 写回：end_line = min(next_start, total_lines-1, max(原end_line, start_line))
    for ci, chap in enumerate(chapters_raw):
        c_start = int(chap.get("start_line", 0))
        c_orig_end = int(chap.get("end_line", c_start))
        chap_ref = ("chap", ci, None)
        new_end = next_starts[chap_ref]
        # 至少不能比 start_line 小，且不超过 next_start 与 total_lines-1
        chap["end_line"] = max(c_start, min(new_end, total_lines - 1, c_orig_end if c_orig_end > c_start else new_end))

        for si, sub in enumerate(chap.get("subsections") or []):
            s_start = int(sub.get("start_line", 0))
            s_orig_end = int(sub.get("end_line", s_start))
            sub_ref = ("sub", ci, si)
            sub_new_end = next_starts[sub_ref]
            sub["end_line"] = max(s_start, min(sub_new_end, total_lines - 1, s_orig_end if s_orig_end > s_start else sub_new_end))

    return chapters_raw


def _split_numbered_into_segments(numbered: str) -> List[str]:
    """将 `_extract_outline_candidates` 产生的候选章节行文本切分为多段。

    切分原则:
    - 单段不超过 `_OUTLINE_SEGMENT_MAX_LINES` 行 / `_OUTLINE_SEGMENT_MAX_BYTES` 字节
    - 相邻段之间保留 `_OUTLINE_SEGMENT_OVERLAP_LINES` 行重叠（避免跨段章节切碎）
    - 候选行较少时直接返回单段（保持与未切分时行为一致）
    """
    if not numbered:
        return [numbered]
    lines = numbered.split("\n")
    n = len(lines)
    if n <= _OUTLINE_SEGMENT_MAX_LINES and len(numbered.encode("utf-8")) <= _OUTLINE_SEGMENT_MAX_BYTES:
        return [numbered]

    segments: List[str] = []
    start = 0
    while start < n:
        end = min(start + _OUTLINE_SEGMENT_MAX_LINES, n)
        seg_lines = lines[start:end]
        # 字节阈值兜底：如果按行数切完仍超过字节预算，回退缩短直到满足
        while (
            len(("\n".join(seg_lines)).encode("utf-8")) > _OUTLINE_SEGMENT_MAX_BYTES
            and len(seg_lines) > 1
        ):
            seg_lines.pop()
        segments.append("\n".join(seg_lines))
        actual_end = start + len(seg_lines)
        if actual_end >= n:
            break
        next_start = actual_end - _OUTLINE_SEGMENT_OVERLAP_LINES
        # 防止 overlap 过大导致死循环
        start = next_start if next_start > start else actual_end
    return segments


def _invoke_outline_llm_with_retry(
    llm: Any,
    system_prompt: str,
    user_prompt: str,
    timeout_exc_types: tuple,
    label: str = "",
    max_attempts: int = 3,
) -> Any:
    """带超时重试的 LLM 调用 (仅对超时异常重试，其他异常直接抛出)。"""
    from langchain_core.messages import SystemMessage, HumanMessage

    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            return llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
        except Exception as e:
            is_timeout = bool(timeout_exc_types) and isinstance(e, timeout_exc_types)
            if not is_timeout and type(e).__name__ in (
                "APITimeoutError", "ReadTimeout", "TimeoutException"
            ):
                is_timeout = True
            if is_timeout:
                last_err = e
                tail = "重试..." if attempt + 1 < max_attempts else "放弃"
                prefix = f" {label}" if label else ""
                logger.warning(
                    f"llm_parse_outline{prefix}: 第 {attempt+1}/{max_attempts} 次调用超时 "
                    f"({type(e).__name__}: {e})，{tail}"
                )
                continue
            raise
    assert last_err is not None
    raise last_err


def _parse_outline_response_json(response: Any) -> dict:
    """解析 LLM 响应文本为 JSON dict（含 code-fence 剥离 + 多级修复回退）。"""
    raw = response.content if hasattr(response, "content") else str(response)
    if not raw or not raw.strip():
        raise ValueError("LLM 返回空响应，请检查 API 连通性或尝试缩短文档")
    cleaned = _strip_code_fence(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        _log_outline_failure(raw, cleaned, e)
        # 1) 先尝试从文本中抓出第一个 {...} 块（处理 LLM 夹杂说明文字的情况）
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if not m:
            raise ValueError(f"LLM 大纲输出非 JSON: {raw[:200]}") from e
        block = m.group(0)
        try:
            return json.loads(block)
        except json.JSONDecodeError as e2:
            # 2) 修复字符串内的未转义引号/换行
            repaired = _repair_malformed_json(block)
            _log_outline_failure(raw, cleaned, e2, repaired=repaired)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as e3:
                raise ValueError(
                    f"LLM 大纲 JSON 修复后仍失败 ({e3}); 原始前 200 字符: {raw[:200]}"
                ) from e3


def _dedupe_chapters(chapters: list) -> list:
    """合并多段返回的 chapters：按 (规范化标题, start_line) 去重；
    冲突时合并 subsections（按 sub.start_line 去重）。

    返回值按 chapter.start_line 升序排序，subsections 同样升序。
    """
    seen: Dict[tuple, dict] = {}
    for ch in chapters:
        title = (ch.get("title") or "").strip()
        if not title:
            continue
        key = (re.sub(r"\s+", "", title.lower()), int(ch.get("start_line", 0)))
        if key in seen:
            existing = seen[key]
            existing_subs = existing.setdefault("subsections", [])
            existing_starts = {int(s.get("start_line", 0)) for s in existing_subs}
            for s in (ch.get("subsections") or []):
                s_start = int(s.get("start_line", 0))
                if s_start not in existing_starts:
                    existing_subs.append(s)
                    existing_starts.add(s_start)
        else:
            seen[key] = dict(ch)

    out = list(seen.values())
    out.sort(key=lambda c: int(c.get("start_line", 0)))
    for ch in out:
        if ch.get("subsections"):
            ch["subsections"] = sorted(
                ch["subsections"], key=lambda s: int(s.get("start_line", 0))
            )
    return out


def llm_parse_outline(document_text: str, llm: Any) -> tuple:
    """
    用 LLM 一次性识别文档的章节与小节结构。

    Args:
        document_text: 完整文档文本（Markdown 形式）
        llm: ChatOpenAI 实例（或任何有 invoke 方法的 LLM 对象）

    Returns:
        (outline, sections)
        - outline: 树状大纲 [{title, level=1, content, children:[{title, level=2, content, children:[]}]}, ...]
        - sections: 扁平审核单元 [{title, content, level, breadcrumb, content_length, content_preview}, ...]

    Raises:
        ValueError: LLM 返回不符合预期（无 chapters / JSON 解析失败 / 全部章节为空）
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    # 收集可能出现的超时异常类（不同 SDK 不一样，name-based 兜底）
    try:
        from openai import APITimeoutError as _OpenAITimeout
    except Exception:
        _OpenAITimeout = None  # type: ignore
    try:
        from httpx import TimeoutException as _HttpxTimeout
    except Exception:
        _HttpxTimeout = None  # type: ignore
    import asyncio as _asyncio  # noqa: F401

    lines = document_text.split("\n")
    total_lines = len(lines)

    # 抽取候选章节行（含原行号），避免把整篇文档塞给 LLM 触发超时
    numbered, _selected, _ = _extract_outline_candidates(document_text)

    timeout_exc_types: tuple = tuple(
        t for t in (_OpenAITimeout, TimeoutError, _asyncio.TimeoutError, _HttpxTimeout) if t is not None
    )

    # 根据候选行规模切分为多段（小文档则单段）
    segments = _split_numbered_into_segments(numbered)
    seg_count = len(segments)
    candidate_line_count = numbered.count("\n") + 1 if numbered else 0
    logger.info(
        f"llm_parse_outline: 候选行 {candidate_line_count} 条, 切分为 {seg_count} 段调用 LLM"
    )

    all_chapters: list = []
    for idx, seg_text in enumerate(segments):
        if seg_count == 1:
            user_prompt = (
                "下面是待分析文档的**候选章节行**（已过滤非标题内容），每行前缀为 [行号]（4 位补零）。"
                "请仅根据这些行的行号与标题文字识别正文章节与小节结构，"
                "正文范围由行号决定，不需要阅读内容细节。"
                "按系统指令的 JSON 格式输出。\n\n"
                f"{seg_text}"
            )
            seg_label = ""
        else:
            user_prompt = (
                f"下面是待分析文档的**候选章节行第 {idx+1}/{seg_count} 段**（已过滤非标题内容，"
                "相邻段之间存在少量行重叠以避免跨段切碎）。"
                "每行前缀为 [行号]（4 位补零）。请仅根据本段行号与标题文字识别该段范围内的章节与小节，"
                "按系统指令的 JSON 格式输出（chapters 仅含本段内的章节即可，重复部分由后续合并去重处理）。\n\n"
                f"{seg_text}"
            )
            seg_label = f"[seg {idx+1}/{seg_count}]"

        try:
            response = _invoke_outline_llm_with_retry(
                llm,
                _OUTLINE_SYSTEM_PROMPT,
                user_prompt,
                timeout_exc_types,
                label=seg_label,
                max_attempts=_OUTLINE_SEGMENT_MAX_ATTEMPTS,
            )
        except Exception as e:
            if seg_count > 1:
                logger.warning(
                    f"llm_parse_outline {seg_label}: 段调用最终失败 "
                    f"({type(e).__name__}: {e})，跳过该段继续合并其他段"
                )
                continue
            raise

        try:
            data = _parse_outline_response_json(response)
        except ValueError as e:
            if seg_count > 1:
                logger.warning(
                    f"llm_parse_outline {seg_label}: JSON 解析失败 ({e})，跳过该段"
                )
                continue
            raise

        seg_chapters = data.get("chapters", []) or []
        all_chapters.extend(seg_chapters)
        if seg_count > 1:
            logger.info(
                f"llm_parse_outline {seg_label}: 本段识别 {len(seg_chapters)} 个章节"
            )

    if not all_chapters:
        raise ValueError("LLM 未识别到任何章节")

    # 多段返回需要合并去重（单段时此调用是无害的恒等操作）
    chapters_raw = _dedupe_chapters(all_chapters)
    if seg_count > 1:
        logger.info(
            f"llm_parse_outline: 多段合并后 {len(chapters_raw)} 个章节（去重前 {len(all_chapters)}）"
        )

    # 重要：在代码层重算 end_line（按"下一节点 start - 1"），不依赖 LLM 给的值。
    # 原因：(1) 候选行不含正文，LLM 无法准确判断章节结尾；(2) 即便发全文，LLM 也常给错。
    chapters_raw = _recompute_chapter_end_lines(chapters_raw, total_lines)

    outline: list = []
    sections: list = []

    for chap in chapters_raw:
        chap_title = (chap.get("title") or "").strip()
        if not chap_title:
            continue
        chap_start = int(chap.get("start_line", 0))
        chap_end = int(chap.get("end_line", chap_start))

        chap_node = {
            "title": chap_title,
            "level": 1,
            "content": "",
            "children": [],
        }

        subs = chap.get("subsections") or []
        if not subs:
            # 无子小节：整章作为唯一审核单元（包含标题行，避免单行小节内容为空）
            content = _safe_slice_lines(lines, chap_start, chap_end)
            chap_node["content"] = content
            content_preview = content[:300] + "..." if len(content) > 300 else content
            sections.append({
                "title": chap_title,
                "content": content,
                "level": 1,
                "breadcrumb": chap_title,
                "content_length": len(content),
                "content_preview": content_preview,
            })
        else:
            for sub in subs:
                sub_title = (sub.get("title") or "").strip()
                if not sub_title:
                    continue
                sub_start = int(sub.get("start_line", 0))
                sub_end = int(sub.get("end_line", sub_start))
                # 包含标题行：对"标题+内容同一行"的情况至关重要（不会产生空 content）
                sub_content = _safe_slice_lines(lines, sub_start, sub_end)
                content_preview = sub_content[:300] + "..." if len(sub_content) > 300 else sub_content
                chap_node["children"].append({
                    "title": sub_title,
                    "level": 2,
                    "content": sub_content,
                    "children": [],
                })
                sections.append({
                    "title": sub_title,
                    "content": sub_content,
                    "level": 2,
                    "breadcrumb": f"{chap_title} / {sub_title}",
                    "content_length": len(sub_content),
                    "content_preview": content_preview,
                })

        outline.append(chap_node)

    if not sections:
        raise ValueError("LLM 识别出的章节均为空")

    return outline, sections


def parse_document_structure(text: str, llm: Any = None) -> Tuple[List[dict], List[dict]]:
    """
    解析文档结构：优先使用 LLM，失败时回退到正则解析。

    Args:
        text: 文档全文文本（Markdown 形式）
        llm: 可选的 LLM 实例（有 invoke 方法）。为 None 时直接使用正则解析。

    Returns:
        (outline, sections)
        - outline: 树状大纲 [{title, level, content, children}, ...]
        - sections: 扁平审核单元 [{title, content, level, breadcrumb, content_length, content_preview}, ...]
    """
    if llm is not None:
        try:
            outline, sections = llm_parse_outline(text, llm)
            logger.info(f"parse_document_structure: LLM 模式 — {len(outline)} 章, {len(sections)} 节")
            return outline, sections
        except Exception as e:
            logger.warning(
                f"parse_document_structure: LLM 失败 ({type(e).__name__}: {e})，回退正则解析"
            )

    # 正则回退
    outline = parse_document_outline(text)
    sections_raw = flatten_to_audit_units(outline)
    sections = []
    for sec in sections_raw:
        title, content, level, breadcrumb = sec
        sections.append({
            "title": title,
            "content": content,
            "level": level,
            "breadcrumb": breadcrumb,
            "content_length": len(content),
            "content_preview": content[:300] + "..." if len(content) > 300 else content,
        })
    logger.info(f"parse_document_structure: 正则模式 — {len(outline)} 章, {len(sections)} 节")
    return outline, sections


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    将长文本分块，便于向量检索

    Args:
        text: 待分块的文本
        chunk_size: 每块字符数
        overlap: 相邻块重叠字符数

    Returns:
        分块后的文本列表
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:])
            break

        # 在句号、换行或逗号处截断，保证语义完整
        chunk = text[start:end]
        for sep in ['\n\n', '\n', '。', '；', '，', '. ', '; ', ', ']:
            last_sep = chunk.rfind(sep)
            if last_sep > chunk_size * 0.5:
                end = start + last_sep + len(sep)
                chunk = chunk[:last_sep + len(sep)]
                break

        chunks.append(chunk)
        start = end - overlap

    return chunks


def get_file_metadata(file_path: str) -> dict:
    """
    获取文件元数据

    Args:
        file_path: 文件路径

    Returns:
        包含文件信息的字典
    """
    path = Path(file_path)
    return {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
    }
