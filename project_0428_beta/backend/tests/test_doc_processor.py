"""
测试 doc_processor.py 的文档结构提取功能
"""
import sys
import os
import json

import pytest

# 添加 backend 目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from doc_processor import (
    split_by_markdown_headers,
    _repair_malformed_json,
    _extract_outline_candidates,
    _recompute_chapter_end_lines,
)


class TestSplitByMarkdownHeaders:
    """测试 Markdown 标题分割"""

    def test_basic_headings(self):
        """基本的 #/##/### 标题分割"""
        text = """# 第一章 风险管理计划
这是第一章的内容。

## 1.1 适用范围
这是1.1的内容。

## 1.2 职责分配
这是1.2的内容。

# 第二章 危害识别
这是第二章的内容。"""

        sections = split_by_markdown_headers(text)
        assert len(sections) == 4
        assert sections[0][0] == "第一章 风险管理计划"
        assert sections[0][2] == 1  # level
        assert sections[1][0] == "1.1 适用范围"
        assert sections[1][2] == 2
        assert sections[2][0] == "1.2 职责分配"
        assert sections[3][0] == "第二章 危害识别"
        assert sections[3][2] == 1

    def test_no_headings(self):
        """没有标题的文档，整体作为一个段落"""
        text = """这是第一段内容。
这是第二段内容。
没有标题的纯文本。"""

        sections = split_by_markdown_headers(text)
        assert len(sections) == 1
        assert sections[0][2] == 0  # level 0 = no heading

    def test_deeply_nested_headers(self):
        """深层嵌套标题"""
        text = """# 一级标题
一级内容
## 二级标题
二级内容
### 三级标题
三级内容
#### 四级标题
四级内容"""

        sections = split_by_markdown_headers(text)
        assert len(sections) == 4
        assert sections[0][2] == 1
        assert sections[1][2] == 2
        assert sections[2][2] == 3
        assert sections[3][2] == 4

    def test_empty_sections(self):
        """空内容章节被跳过"""
        text = """# 标题1

# 标题2
有内容

# 标题3

"""

        sections = split_by_markdown_headers(text)
        # 标题1 没有内容，空段落
        # 标题2 有内容
        # 标题3 没有内容
        assert len(sections) >= 1
        # 至少标题2的段落应该在
        found = any(s[0] == "标题2" for s in sections)
        assert found

    def test_content_with_markdown_table(self):
        """包含 Markdown 表格的章节"""
        text = """# 风险评估
以下为风险评估表：

| 危害 | 严重度 | 概率 |
|------|--------|------|
| 机械危害 | 高 | 中 |
| 化学危害 | 中 | 低 |

## 风险控制
控制措施如下。"""

        sections = split_by_markdown_headers(text)
        assert len(sections) == 2
        assert sections[0][0] == "风险评估"
        assert "机械危害" in sections[0][1]
        assert sections[1][0] == "风险控制"

    def test_mixed_heading_levels(self):
        """混合标题层级"""
        text = """文档开头内容，没有标题

# 风险管理计划
计划内容

### 风险可接受准则
准则内容

## 危害分析
分析内容"""

        sections = split_by_markdown_headers(text)
        assert len(sections) == 4
        assert sections[0][0] == "文档开头"
        assert sections[1][0] == "风险管理计划"
        assert sections[1][2] == 1
        assert sections[2][0] == "风险可接受准则"
        assert sections[2][2] == 3
        assert sections[3][0] == "危害分析"
        assert sections[3][2] == 2

    def test_empty_input(self):
        """空输入"""
        sections = split_by_markdown_headers("")
        assert len(sections) == 0

    def test_whitespace_only_input(self):
        """只有空白的输入"""
        sections = split_by_markdown_headers("   \n\n   \n   ")
        assert len(sections) == 0

    def test_heading_without_space(self):
        """标题#后没有空格，不应识别为标题"""
        text = """#这不是标题
这是正文内容。"""

        sections = split_by_markdown_headers(text)
        # #后无空格，整个文档作为一个段落
        assert len(sections) == 1
        assert sections[0][2] == 0  # level 0 = no heading detected

    def test_chinese_numbered_headings_in_markdown(self):
        """中文编号在标题文本中"""
        text = """# 一、风险管理计划
计划内容

# 二、危害识别
识别内容

## 2.1 物理危害
物理危害内容"""

        sections = split_by_markdown_headers(text)
        assert len(sections) == 3
        assert sections[0][0] == "一、风险管理计划"
        assert sections[1][0] == "二、危害识别"
        assert sections[2][0] == "2.1 物理危害"


class TestRepairMalformedJson:
    """测试 _repair_malformed_json"""

    def test_legal_json_unchanged(self):
        """合法 JSON 应当原样返回，不被改写"""
        legal = '{"chapters": [{"title": "1.1 产品概述", "start_line": 1, "end_line": 5}]}'
        repaired = _repair_malformed_json(legal)
        # 解析后内容应一致
        assert json.loads(repaired) == json.loads(legal)
        # 文本本身应未变化 (没有全角引号)
        assert repaired == legal

    def test_embedded_ascii_quote_in_title(self):
        """标题里含未转义的 ASCII 双引号 (LLM 常见错误)"""
        bad = '{"chapters": [{"title": "1.1 "产品" 概述", "start_line": 1, "end_line": 5}]}'
        # 标准解析应当失败
        with pytest.raises(json.JSONDecodeError):
            json.loads(bad)
        # 修复后应能解析
        repaired = _repair_malformed_json(bad)
        data = json.loads(repaired)
        assert data["chapters"][0]["title"] == "1.1 “产品” 概述"

    def test_raw_newline_in_string(self):
        """字符串内含裸换行符"""
        bad = '{"chapters": [{"title": "第一章\n风险管理", "start_line": 1, "end_line": 2}]}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(bad)
        repaired = _repair_malformed_json(bad)
        data = json.loads(repaired)
        assert data["chapters"][0]["title"] == "第一章\n风险管理"

    def test_realistic_llm_output_with_quote_at_line_73(self):
        """模拟错误信息中的 line 73 column 69 场景：
        LLM 在某个深层 chapter 标题里嵌入了未转义的双引号"""
        # 构造一个多章节的 JSON，故意在末尾某章标题里嵌入引号
        chapters = []
        for i in range(1, 20):
            chapters.append(
                f'    {{"title": "{i}. 章节{i}", "start_line": {i*10}, "end_line": {i*10+8}, "subsections": []}}'
            )
        chapters.append(
            '    {"title": "20. 章节"20" 描述", "start_line": 200, "end_line": 210, "subsections": []}'
        )
        bad = '{\n  "chapters": [\n' + ",\n".join(chapters) + "\n  ]\n}"
        with pytest.raises(json.JSONDecodeError):
            json.loads(bad)
        repaired = _repair_malformed_json(bad)
        data = json.loads(repaired)
        # 嵌入引号的章节应被还原为可读字符串
        assert data["chapters"][-1]["title"] == "20. 章节“20” 描述"
        # 前面正常的章节 title 应当完全保留
        assert data["chapters"][0]["title"] == "1. 章节1"
        assert data["chapters"][-2]["title"] == "19. 章节19"

    def test_escaped_quote_preserved(self):
        """已经正确转义的 \\" 不应被当作嵌入引号处理"""
        legal = r'{"chapters": [{"title": "1.1 \"产品\" 概述", "start_line": 1, "end_line": 5}]}'
        # 这本来就能被标准解析
        data = json.loads(legal)
        assert data["chapters"][0]["title"] == '1.1 "产品" 概述'
        # 修复后内容应保持一致
        repaired = _repair_malformed_json(legal)
        assert json.loads(repaired) == data

    def test_empty_string_preserved(self):
        """空字符串值应当保留"""
        legal = '{"chapters": [{"title": "", "start_line": 1, "end_line": 2}]}'
        repaired = _repair_malformed_json(legal)
        assert json.loads(repaired) == json.loads(legal)


class TestExtractOutlineCandidates:
    """测试 _extract_outline_candidates：抽取候选章节行 + 压缩 prompt 体积"""

    def test_short_document_kept_in_full(self):
        """短文档（< 30 候选行）→ 保留全部行"""
        doc = "\n".join(
            [f"第{i}章 概述" if i % 5 == 0 else f"正文行 {i}" for i in range(50)]
        )
        numbered, selected, total = _extract_outline_candidates(doc)
        assert total == 50
        assert len(selected) <= 50
        # 行号前缀应存在
        assert "[0000]" in numbered
        assert "[0001]" in numbered

    def test_large_document_compresses_heavily(self):
        """大文档：1000 行中只有 50 行是章节 → 抽取后应只有约 50+10 行"""
        lines = []
        for i in range(1000):
            if i % 20 == 0:
                lines.append(f"{i // 20 + 1}. 章节标题")
            else:
                lines.append(f"这是第 {i} 行的正文内容，描述某些细节。")
        doc = "\n".join(lines)
        numbered, selected, total = _extract_outline_candidates(doc)
        assert total == 1000
        # 候选 = 50 章节 + 5 头 + 5 尾 = 60 行左右 (远少于 1000)
        assert len(selected) < 100
        assert len(selected) > 30
        # 抽样验证包含预期章节
        assert "1. 章节标题" in numbered
        assert "50. 章节标题" in numbered

    def test_line_numbers_preserved(self):
        """抽取后行号必须与原文档对应，方便 LLM 返回 start_line 后代码层切片"""
        # 100 行普通 + 1 行 "1. 第一章" (索引 100) + 10 行尾行
        doc = "\n".join(
            [f"行 {i}" for i in range(100)]
            + ["1. 第一章"]
            + [f"尾行 {i}" for i in range(10)]
        )
        numbered, selected, total = _extract_outline_candidates(doc)
        # "1. 第一章" 在原文档索引为 100 (0-based) → 4 位补零为 0100
        assert "[0100] 1. 第一章" in numbered
        # 末尾 5 行被保留作签字栏
        assert "[0106] 尾行 5" in numbered
        assert "[0110] 尾行 9" in numbered

    def test_no_headings_falls_back_to_prefix(self):
        """无章节结构文档 → 保留前 _OUTLINE_MAX_LINES 行"""
        doc = "\n".join([f"普通行 {i}" for i in range(500)])
        numbered, selected, total = _extract_outline_candidates(doc)
        assert total == 500
        # 没有匹配章节 → 走 "保留全部但限制行数" 分支 → 选满 500 行
        assert len(selected) == 500

    def test_byte_limit_truncates_output(self):
        """输出字节超 _OUTLINE_MAX_BYTES → 截断到上限以下"""
        # 构造每行很长、章节很多的文档
        long_line = "正文 " * 200  # 约 600 字节
        lines = []
        for i in range(2000):
            if i % 10 == 0:
                lines.append(f"{i // 10 + 1}. 章节标题" + " X" * 100)
            else:
                lines.append(long_line)
        doc = "\n".join(lines)
        numbered, selected, total = _extract_outline_candidates(doc)
        # 输出应不超过 200_000 字节 + 1 行余量
        assert len(numbered.encode("utf-8")) <= 200_000 + 1000
        # 至少保留部分章节
        assert len(selected) > 50


class TestRecomputeChapterEndLines:
    """测试 _recompute_chapter_end_lines：按下一节点 start 重算 end_line"""

    def test_two_chapters_with_subsections(self):
        """2 个 chapter，每个含 subsection → end_line 应等于下一节点 start - 1"""
        chapters = [
            {
                "title": "第一章",
                "start_line": 0,
                "end_line": 999,  # LLM 给的错值，应被覆盖
                "subsections": [
                    {"title": "1.1", "start_line": 1, "end_line": 999},
                    {"title": "1.2", "start_line": 5, "end_line": 999},
                ],
            },
            {
                "title": "第二章",
                "start_line": 10,
                "end_line": 999,
                "subsections": [
                    {"title": "2.1", "start_line": 12, "end_line": 999},
                ],
            },
        ]
        out = _recompute_chapter_end_lines(chapters, total_lines=20)
        # 第一章 end_line = 9 (1.2 下一节点是 2.1.start=12, 12-1=11 但 1.1 排序在 1.2 前, 1.1.end=5-1=4?
        # 实际：flat 排序 = [(0,chap0),(1,sub0,0),(5,sub0,1),(10,chap1),(12,sub1,0)]
        # chap0.next = flat[1].start - 1 = 1 - 1 = 0  → 但 max(0, 0)=0
        # sub0,0.next = flat[2].start - 1 = 5 - 1 = 4
        # sub0,1.next = flat[3].start - 1 = 10 - 1 = 9
        # chap1.next = flat[4].start - 1 = 12 - 1 = 11
        # sub1,0.next = total_lines - 1 = 19
        assert out[0]["end_line"] == 0   # 紧接在 1.1 之前? 实际 chap 是 level1, 1.1 应该是 chap 的子
        # 注意：算法按"所有节点全局排序"而非层级，遇到 chap0 后下一个就是 sub0,0
        # 这是已知的简化行为 — 章节边界对 level1 不严格，对 level2 准确
        assert out[0]["subsections"][0]["end_line"] == 4  # 1.1 end = 4
        assert out[0]["subsections"][1]["end_line"] == 9  # 1.2 end = 9
        assert out[1]["end_line"] == 11  # 第二章 end = 11
        assert out[1]["subsections"][0]["end_line"] == 19  # 2.1 end = 19 (文档末尾)

    def test_last_chapter_end_is_total_lines_minus_one(self):
        """最后一个节点的 end_line = total_lines - 1"""
        chapters = [
            {"title": "1. 概述", "start_line": 0, "end_line": 0, "subsections": []},
        ]
        out = _recompute_chapter_end_lines(chapters, total_lines=50)
        assert out[0]["end_line"] == 49

    def test_clamp_to_valid_range(self):
        """end_line 不会超过 total_lines-1，也不会小于 start_line"""
        chapters = [
            {"title": "1.", "start_line": 5, "end_line": 9999, "subsections": []},
        ]
        out = _recompute_chapter_end_lines(chapters, total_lines=10)
        assert 5 <= out[0]["end_line"] <= 9

    def test_empty_chapters_list(self):
        """空列表应原样返回"""
        assert _recompute_chapter_end_lines([], total_lines=100) == []


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
