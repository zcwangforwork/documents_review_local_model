"""多 Agent 子 Agent 工厂

实现说明:
- LangChain `langchain.agents.create_agent` API 在不同版本上的可用性不一，
  本模块采用更可靠的实现方式：直接基于 `langchain_core` + `langgraph` 构建
  子 Agent 行为（System Prompt + LLM + 必要时的工具循环）。
- 每个子 Agent 函数接受一个上下文 dict，返回结构化结果，便于 Supervisor
  Graph 通过 Send API 调度。
- 复用 `agent_graph._build_audit_system_prompt` 与 `agent_tools.create_audit_tools`，
  避免重复造轮子。

子 Agent 列表:
  1. analyze_structure_agent  : 结构分析 Agent（一次性调用 LLM/正则）
  2. audit_chapter_agent      : 章节审核 Agent（内部循环每个小节）
  3. synthesize_report_agent  : 综合报告 Agent（汇总章节结果）
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)


# ============================================================
# Structure Analyzer Agent
# ============================================================


def analyze_structure_agent(
    document_text: str,
    outline: List[dict],
    sections: List[dict],
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """文档结构分析 Agent。

    职责: 将 outline + sections 整理成"章节级"结构 chapter_structure，
    每个 chapter 包含其下属 subsection 在原始 sections 列表中的索引。

    实现策略:
      - 优先复用 `agent_graph.analyze_outline_node` 的逻辑，避免重复造轮子
      - 输出与 supervisor_graph 期望的 chapter_structure 字段对齐

    Args:
        document_text: 原始文档文本
        outline: 文档大纲（来自 doc_processor.parse_document_structure）
        sections: 平铺的小节列表
        llm: 可选 LLM 实例（仅 sections 为空时兜底使用）

    Returns:
        dict {
            "chapter_structure": [{chapter_idx, chapter_title, subsection_indices, section_count}, ...],
            "outline_summary": {chapter_count, section_count, tree_text, chapters, source},
            "total_chapters": int,
            "total_sections": int,
        }
    """
    # 复用现有的 analyze_outline_node 逻辑（保证与单 Agent 模式产出一致）
    from agent_graph import analyze_outline_node

    pseudo_state = {
        "document_text": document_text,
        "outline": outline,
        "sections": sections,
    }
    delta = analyze_outline_node(pseudo_state, llm=llm)
    outline_summary = delta.get("outline_summary", {}) or {}

    chapters_info = outline_summary.get("chapters", []) or []
    chapter_structure: List[dict] = []
    for c_idx, chap in enumerate(chapters_info):
        sub_indices = chap.get("subsection_indices", []) or []
        chapter_structure.append({
            "chapter_idx": c_idx,
            "chapter_title": chap.get("title", f"章节{c_idx + 1}"),
            "level": chap.get("level", 1),
            "subsection_indices": sub_indices,
            "section_count": len(sub_indices),
        })

    total_chapters = len(chapter_structure)
    total_sections = outline_summary.get("section_count", len(sections))

    logger.info(
        f"[StructureAnalyzer] 识别 {total_chapters} 个章节, {total_sections} 个小节 "
        f"(source={outline_summary.get('source', 'regex')})"
    )

    return {
        "chapter_structure": chapter_structure,
        "outline_summary": outline_summary,
        "total_chapters": total_chapters,
        "total_sections": total_sections,
    }


# ============================================================
# Chapter Auditor Agent
# ============================================================


def _audit_one_subsection(
    section: dict,
    doc_type: str,
    knowledge_context: str,
    section_knowledge: str,
    llm: Any,
    chapter_context: str = "",
) -> dict:
    """对单个小节执行一次 LLM 审核，返回结构化结果。

    复用 `agent_graph._build_audit_system_prompt` 与 `_extract_json`，
    保证审核 prompt 与单 Agent 模式完全一致。
    """
    from agent_graph import _build_audit_system_prompt, _extract_json

    system_prompt = _build_audit_system_prompt(
        section=section,
        depth="standard",
        focus_standards=[],
        doc_type=doc_type,
        knowledge_context=knowledge_context,
        section_knowledge=section_knowledge,
        conversation_context=chapter_context,
    )

    title = section.get("title", "Untitled")
    breadcrumb = section.get("breadcrumb", "") or title
    content_text = section.get("content", "") or ""

    user_prompt = (
        f"## Current Section\n"
        f"**Title**: {title}\n"
        f"**Breadcrumb**: {breadcrumb}\n"
        f"**Content**:\n{content_text[:8000]}"
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        result_content = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"[ChapterAuditor] LLM 调用失败 (section='{title}'): {e}")
        result_content = f"[LLM Error: {e}]"

    metadata = _extract_json(result_content, {"confidence": 3, "key_findings": []})

    return {
        "title": title,
        "breadcrumb": breadcrumb,
        "depth": "standard",
        "findings": metadata.get("key_findings", []),
        "confidence": int(metadata.get("confidence", 3)),
        "round_count": 1,
        "result_content": result_content,
    }


async def _audit_one_subsection_async(
    section: dict,
    doc_type: str,
    knowledge_context: str,
    section_knowledge: str,
    llm: Any,
    chapter_context: str = "",
) -> dict:
    """异步版本的小节审核（Chapter Auditor 内部并发调用使用）。"""
    from agent_graph import _build_audit_system_prompt, _extract_json

    system_prompt = _build_audit_system_prompt(
        section=section,
        depth="standard",
        focus_standards=[],
        doc_type=doc_type,
        knowledge_context=knowledge_context,
        section_knowledge=section_knowledge,
        conversation_context=chapter_context,
    )

    title = section.get("title", "Untitled")
    breadcrumb = section.get("breadcrumb", "") or title
    content_text = section.get("content", "") or ""

    user_prompt = (
        f"## Current Section\n"
        f"**Title**: {title}\n"
        f"**Breadcrumb**: {breadcrumb}\n"
        f"**Content**:\n{content_text[:8000]}"
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        result_content = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"[ChapterAuditor:async] LLM 调用失败 (section='{title}'): {e}")
        result_content = f"[LLM Error: {e}]"

    metadata = _extract_json(result_content, {"confidence": 3, "key_findings": []})

    return {
        "title": title,
        "breadcrumb": breadcrumb,
        "depth": "standard",
        "findings": metadata.get("key_findings", []),
        "confidence": int(metadata.get("confidence", 3)),
        "round_count": 1,
        "result_content": result_content,
    }


async def audit_chapter_agent(
    chapter_idx: int,
    chapter_title: str,
    subsections: List[dict],
    subsection_indices: List[int],
    doc_type: str,
    retriever: Any,
    llm: Any,
    knowledge_context: str = "",
    inner_concurrency: int = 3,
) -> dict:
    """章节审核 Agent — 负责审核一个章节的所有小节。

    流程:
      1. 对该章节下每个小节，从知识库检索专用法规条款
      2. 使用 deepseek-v4-pro 对照法规审核每个小节
      3. 章节内的小节使用 asyncio.Semaphore 限流并发
      4. 输出 chapter_summary（章节级总结）+ subsection_results

    Args:
        chapter_idx: 章节索引（0-based）
        chapter_title: 章节标题
        subsections: 该章节下属小节列表
        subsection_indices: 这些小节在原始 sections 中的索引
        doc_type: 文档类型（risk_management/design_dev/...）
        retriever: RAGRetriever 实例
        llm: ChatOpenAI 实例
        knowledge_context: 顶层预检索的知识库摘要（可选）
        inner_concurrency: 章节内小节级并发上限

    Returns:
        dict {
            "chapter_idx", "chapter_title",
            "subsection_results": [...],  # 每个小节的详细结果
            "subsection_indices": [...],
            "chapter_summary": str,       # 章节级综合摘要
            "agent_steps": int,
        }
    """
    import asyncio
    from agent_graph import _retrieve_section_knowledge

    if not subsections:
        return {
            "chapter_idx": chapter_idx,
            "chapter_title": chapter_title,
            "subsection_results": [],
            "subsection_indices": subsection_indices,
            "chapter_summary": f"章节 {chapter_title} 无可审核小节。",
            "agent_steps": 0,
        }

    logger.info(
        f"[ChapterAuditor#{chapter_idx}] 开始审核章节 '{chapter_title}', "
        f"小节数={len(subsections)}, 内部并发={inner_concurrency}"
    )

    semaphore = asyncio.Semaphore(max(1, inner_concurrency))

    async def _bounded_audit(idx: int, section: dict) -> dict:
        async with semaphore:
            section_knowledge = ""
            try:
                section_knowledge = _retrieve_section_knowledge(section, doc_type, retriever)
            except Exception as e:
                logger.warning(
                    f"[ChapterAuditor#{chapter_idx}] 小节知识检索失败 idx={idx}: {e}"
                )
            chapter_context = (
                f"This subsection belongs to chapter '{chapter_title}'. "
                f"Audit each subsection in coherent reference to its sibling subsections."
            )
            res = await _audit_one_subsection_async(
                section=section,
                doc_type=doc_type,
                knowledge_context=knowledge_context,
                section_knowledge=section_knowledge,
                llm=llm,
                chapter_context=chapter_context,
            )
            res["section_idx"] = subsection_indices[idx] if idx < len(subsection_indices) else idx
            res["chapter_idx"] = chapter_idx
            return res

    tasks = [_bounded_audit(i, sec) for i, sec in enumerate(subsections)]
    try:
        results = await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"[ChapterAuditor#{chapter_idx}] 并发审核异常: {e}")
        # 退化为顺序执行，并捕获每个失败
        results = []
        for i, sec in enumerate(subsections):
            try:
                results.append(await _bounded_audit(i, sec))
            except Exception as ee:
                logger.error(
                    f"[ChapterAuditor#{chapter_idx}] 小节 {i} 审核失败: {ee}"
                )
                results.append({
                    "section_idx": subsection_indices[i] if i < len(subsection_indices) else i,
                    "chapter_idx": chapter_idx,
                    "title": sec.get("title", ""),
                    "breadcrumb": sec.get("breadcrumb", ""),
                    "depth": "standard",
                    "findings": [],
                    "confidence": 1,
                    "round_count": 1,
                    "result_content": f"[Error: {ee}]",
                })

    # 排序保持稳定输出
    results.sort(key=lambda r: int(r.get("section_idx", 0)))

    # 章节级摘要：汇总所有小节关键发现
    chapter_summary = _build_chapter_summary(chapter_title, results)

    logger.info(
        f"[ChapterAuditor#{chapter_idx}] 章节审核完成 '{chapter_title}', "
        f"输出小节数={len(results)}"
    )

    return {
        "chapter_idx": chapter_idx,
        "chapter_title": chapter_title,
        "subsection_results": results,
        "subsection_indices": subsection_indices,
        "chapter_summary": chapter_summary,
        "agent_steps": len(results),
    }


def _build_chapter_summary(chapter_title: str, subsection_results: List[dict]) -> str:
    """根据小节结果生成章节级摘要文本。"""
    if not subsection_results:
        return f"### {chapter_title}\n\n_本章节无可审核小节。_"

    lines = [f"### {chapter_title}", ""]
    severity_counts = {"critical": 0, "major": 0, "minor": 0, "suggestion": 0}
    confidences: List[int] = []
    for r in subsection_results:
        confidences.append(int(r.get("confidence", 3)))
        for finding in r.get("findings", []) or []:
            f_lower = str(finding).lower()
            if "critical" in f_lower or "严重" in f_lower or "🔴" in f_lower:
                severity_counts["critical"] += 1
            elif "major" in f_lower or "需修改" in f_lower or "🟡" in f_lower:
                severity_counts["major"] += 1
            elif "minor" in f_lower:
                severity_counts["minor"] += 1
            else:
                severity_counts["suggestion"] += 1

    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    lines.append(
        f"- 小节数: {len(subsection_results)} | 平均可信度: {avg_conf:.1f}/5"
    )
    lines.append(
        "- 发现项分布: "
        f"🔴严重 {severity_counts['critical']} / 🟡需修改 {severity_counts['major']} / "
        f"🟢轻微 {severity_counts['minor']} / 💡建议 {severity_counts['suggestion']}"
    )

    # 附带前几条关键发现摘要
    flat_findings: List[str] = []
    for r in subsection_results:
        for f in (r.get("findings", []) or [])[:2]:
            flat_findings.append(f"  - [{r.get('breadcrumb', r.get('title', ''))}] {f}")
    if flat_findings:
        lines.append("- 关键发现:")
        lines.extend(flat_findings[:8])

    return "\n".join(lines)


# ============================================================
# Report Synthesizer Agent
# ============================================================


def synthesize_report_agent(
    chapter_results: List[dict],
    outline_summary: dict,
    contradictions: List[dict],
    document_type: str,
    llm: Any,
) -> dict:
    """综合报告 Agent — 汇总所有章节结果生成最终报告。

    复用 `agent_graph.generate_report_node` 的报告渲染逻辑，
    通过把 chapter_results 展平为 audit_results 的形式喂入。

    Returns:
        dict {
            "final_report": str,
            "subsection_results": [...],   # 展平后供前端展示
        }
    """
    # 展平为单 Agent 模式兼容的 audit_results 形式
    audit_results: List[dict] = []
    for ch in sorted(chapter_results, key=lambda c: int(c.get("chapter_idx", 0))):
        for sub in ch.get("subsection_results", []) or []:
            sub_copy = dict(sub)
            sub_copy.setdefault("chapter_idx", ch.get("chapter_idx"))
            sub_copy.setdefault("chapter_title", ch.get("chapter_title"))
            audit_results.append(sub_copy)

    # 复用 generate_report_node
    from agent_graph import generate_report_node

    pseudo_state = {
        "audit_results": audit_results,
        "contradictions": contradictions or [],
        "document_type": document_type,
        "outline_summary": outline_summary,
    }
    delta = generate_report_node(pseudo_state, llm=llm)

    final_report = delta.get("final_report", "")

    # 在报告末尾追加章节级摘要（多 Agent 模式独有）
    if chapter_results:
        chapter_block_lines = ["", "## 6. 章节级审核摘要（多 Agent 协作）", ""]
        for ch in sorted(chapter_results, key=lambda c: int(c.get("chapter_idx", 0))):
            chapter_block_lines.append(ch.get("chapter_summary", ""))
            chapter_block_lines.append("")
        final_report = final_report + "\n" + "\n".join(chapter_block_lines)

    return {
        "final_report": final_report,
        "subsection_results": audit_results,
    }


def cross_validate_chapters(
    chapter_results: List[dict],
    llm: Any,
) -> List[dict]:
    """章节间交叉验证 — 复用 `agent_graph.cross_validate_node`。"""
    from agent_graph import cross_validate_node

    audit_results: List[dict] = []
    for ch in chapter_results:
        for sub in ch.get("subsection_results", []) or []:
            audit_results.append(sub)

    pseudo_state = {"audit_results": audit_results}
    delta = cross_validate_node(pseudo_state, llm=llm)
    return delta.get("contradictions", []) or []
