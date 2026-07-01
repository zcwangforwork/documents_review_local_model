"""多 Agent Supervisor StateGraph

拓扑（方案A · LangGraph Subagents + Send API 模式）:

  START
    │
    ▼
  analyze_structure        (Structure Analyzer Agent — 一次)
    │
    │  Send API fan-out（每个章节一个 Send packet）
    ▼
  audit_chapter ×N         (Chapter Auditor Agent — 章节级并行)
    │
    │  reducer 自动汇集 chapter_results
    ▼
  synthesize_report        (Report Synthesizer Agent — 一次)
    │
    ▼
   END

关键设计:
- `chapter_results` 字段使用 `Annotated[List[dict], operator.add]` reducer，
  Send API 并行触发的多个 audit_chapter 节点各自返回 `{"chapter_results": [single_dict]}`，
  LangGraph 自动合并成完整列表。
- Chapter Auditor 内部用 asyncio.Semaphore 做小节级并发；Supervisor Send API 做章节级并发。
- 顶层 LLM/RAG 实例通过 `compile_supervisor_graph(llm, retriever, ...)` 闭包注入，
  避免污染 MultiAgentState。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, List

from langgraph.graph import StateGraph, END
from langgraph.types import Send

from multi_agent.state import MultiAgentState
from multi_agent.agents import (
    analyze_structure_agent,
    audit_chapter_agent,
    synthesize_report_agent,
    cross_validate_chapters,
)

logger = logging.getLogger(__name__)


# 章节内并发（小节级）
INNER_CONCURRENCY = int(os.getenv("MULTI_AGENT_INNER_CONCURRENCY", "3"))
# 是否启用章节间交叉验证
ENABLE_CROSS_VALIDATION = os.getenv("MULTI_AGENT_CROSS_VALIDATION", "1") == "1"


# ============================================================
# 节点函数
# ============================================================


def _make_analyze_structure_node(llm: Any):
    """构建结构分析节点（闭包绑定 LLM 用作兜底）。"""

    def _node(state: MultiAgentState) -> dict:
        document_text = state.get("document_text", "") or ""
        outline = state.get("outline", []) or []
        sections = state.get("sections", []) or []

        result = analyze_structure_agent(
            document_text=document_text,
            outline=outline,
            sections=sections,
            llm=llm,
        )

        logger.info(
            f"[Supervisor] Structure 分析完成: "
            f"chapters={result.get('total_chapters', 0)}, "
            f"sections={result.get('total_sections', 0)}"
        )

        return {
            "chapter_structure": result.get("chapter_structure", []),
            "outline_summary": result.get("outline_summary", {}),
            "total_chapters": result.get("total_chapters", 0),
            "total_sections": result.get("total_sections", 0),
            "current_stage": "analyze_structure_done",
        }

    return _node


def _dispatch_chapters(state: MultiAgentState):
    """Send API 条件边 — 为每个章节 fan-out 一个 Send 调用。

    每个 Send 携带的 packet 是 `audit_chapter` 节点的输入参数 dict，
    LangGraph 会以并行方式调度这些 audit_chapter 节点。
    """
    chapter_structure = state.get("chapter_structure", []) or []
    sections = state.get("sections", []) or []
    document_type = state.get("document_type", "") or ""

    if not chapter_structure:
        logger.warning("[Supervisor:dispatch] 无章节结构，跳过 audit_chapter")
        return [Send("synthesize_report", {"_skip": True})]

    sends: List[Send] = []
    for chap in chapter_structure:
        chapter_idx = int(chap.get("chapter_idx", 0))
        chapter_title = chap.get("chapter_title", f"章节{chapter_idx + 1}")
        sub_indices = chap.get("subsection_indices", []) or []

        # 截取该章节下属的 subsection 内容
        subsections: List[dict] = []
        for s_idx in sub_indices:
            if 0 <= s_idx < len(sections):
                subsections.append(sections[s_idx])

        if not subsections:
            logger.info(
                f"[Supervisor:dispatch] 章节 #{chapter_idx} '{chapter_title}' "
                f"无可审小节，跳过"
            )
            continue

        sends.append(
            Send(
                "audit_chapter",
                {
                    "chapter_idx": chapter_idx,
                    "chapter_title": chapter_title,
                    "subsections": subsections,
                    "subsection_indices": sub_indices,
                    "doc_type": document_type,
                },
            )
        )

    if not sends:
        logger.warning("[Supervisor:dispatch] 所有章节均无可审小节")
        return [Send("synthesize_report", {"_skip": True})]

    logger.info(f"[Supervisor:dispatch] 派发 {len(sends)} 个章节级 Send 调用")
    return sends


def _make_audit_chapter_node(llm: Any, retriever: Any):
    """构建章节审核节点（闭包绑定 LLM + RAG retriever）。

    注意:
    - 此节点接收 Send packet 作为完整的 input state（不是 MultiAgentState 全量）
    - 返回值 {"chapter_results": [single_result]} 会被 reducer (operator.add)
      自动合并到全局 chapter_results 列表
    """

    async def _node(state: dict) -> dict:
        # _skip 标志由 dispatch 在无可审章节时设置
        if state.get("_skip"):
            return {"chapter_results": []}

        chapter_idx = int(state.get("chapter_idx", 0))
        chapter_title = state.get("chapter_title", "")
        subsections = state.get("subsections", []) or []
        subsection_indices = state.get("subsection_indices", []) or []
        doc_type = state.get("doc_type", "") or ""

        try:
            result = await audit_chapter_agent(
                chapter_idx=chapter_idx,
                chapter_title=chapter_title,
                subsections=subsections,
                subsection_indices=subsection_indices,
                doc_type=doc_type,
                retriever=retriever,
                llm=llm,
                knowledge_context="",
                inner_concurrency=INNER_CONCURRENCY,
            )
        except Exception as e:
            logger.error(
                f"[Supervisor:audit_chapter#{chapter_idx}] 章节审核失败: {e}"
            )
            result = {
                "chapter_idx": chapter_idx,
                "chapter_title": chapter_title,
                "subsection_results": [],
                "subsection_indices": subsection_indices,
                "chapter_summary": f"### {chapter_title}\n\n_章节审核失败: {e}_",
                "agent_steps": 0,
            }

        # reducer 自动合并：返回单元素列表
        return {"chapter_results": [result]}

    return _node


def _make_synthesize_report_node(llm: Any):
    """构建综合报告节点。"""

    def _node(state: MultiAgentState) -> dict:
        chapter_results = state.get("chapter_results", []) or []
        outline_summary = state.get("outline_summary", {}) or {}
        document_type = state.get("document_type", "") or ""

        # 章节间交叉验证（可选）
        contradictions: List[dict] = []
        if ENABLE_CROSS_VALIDATION and chapter_results:
            try:
                contradictions = cross_validate_chapters(
                    chapter_results=chapter_results,
                    llm=llm,
                )
                logger.info(
                    f"[Supervisor:synthesize] 交叉验证发现 {len(contradictions)} 处矛盾"
                )
            except Exception as e:
                logger.warning(f"[Supervisor:synthesize] 交叉验证失败: {e}")
                contradictions = []

        # 综合报告
        try:
            result = synthesize_report_agent(
                chapter_results=chapter_results,
                outline_summary=outline_summary,
                contradictions=contradictions,
                document_type=document_type,
                llm=llm,
            )
        except Exception as e:
            logger.error(f"[Supervisor:synthesize] 综合报告生成失败: {e}")
            result = {
                "final_report": f"# 审核报告生成失败\n\n错误: {e}",
                "subsection_results": [],
            }

        # 统计 agent steps（每个章节的小节数总和 + 结构分析1次 + 综合报告1次）
        total_steps = 1 + sum(int(c.get("agent_steps", 0)) for c in chapter_results) + 1

        logger.info(
            f"[Supervisor:synthesize] 报告生成完成, total_agent_steps={total_steps}"
        )

        return {
            "subsection_results": result.get("subsection_results", []),
            "contradictions": contradictions,
            "final_report": result.get("final_report", ""),
            "finished": True,
            "total_agent_steps": total_steps,
            "current_stage": "synthesize_done",
        }

    return _node


# ============================================================
# Graph 构建 / 编译
# ============================================================


def build_supervisor_graph(llm: Any, retriever: Any) -> StateGraph:
    """构建 Supervisor StateGraph (未编译)。

    Args:
        llm: ChatOpenAI 实例（顶层 LLM）
        retriever: RAGRetriever 实例

    Returns:
        StateGraph: 待 compile() 的图对象
    """
    graph = StateGraph(MultiAgentState)

    # ===== 节点 =====
    graph.add_node("analyze_structure", _make_analyze_structure_node(llm))
    graph.add_node("audit_chapter", _make_audit_chapter_node(llm, retriever))
    graph.add_node("synthesize_report", _make_synthesize_report_node(llm))

    # ===== 边 =====
    graph.set_entry_point("analyze_structure")

    # Send API 条件边：analyze_structure → audit_chapter ×N
    graph.add_conditional_edges(
        "analyze_structure",
        _dispatch_chapters,
        ["audit_chapter", "synthesize_report"],
    )

    # 所有 audit_chapter 完成后 → synthesize_report
    graph.add_edge("audit_chapter", "synthesize_report")

    # synthesize_report → END
    graph.add_edge("synthesize_report", END)

    return graph


def compile_supervisor_graph(
    llm: Any,
    retriever: Any,
    checkpointer: Optional[Any] = None,
):
    """构建并编译 Supervisor Graph。

    Args:
        llm: ChatOpenAI 实例
        retriever: RAGRetriever 实例
        checkpointer: 可选的 LangGraph checkpointer（如 AsyncSqliteSaver）

    Returns:
        Compiled CompiledGraph 对象（可直接 .ainvoke / .astream）
    """
    graph = build_supervisor_graph(llm=llm, retriever=retriever)

    if checkpointer is not None:
        compiled = graph.compile(checkpointer=checkpointer)
        logger.info("[Supervisor] Graph 编译完成（带 checkpointer）")
    else:
        compiled = graph.compile()
        logger.info("[Supervisor] Graph 编译完成（无 checkpointer）")

    return compiled
