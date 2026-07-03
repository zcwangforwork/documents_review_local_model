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
import time as _time
from typing import Any, Optional, List

from langgraph.graph import StateGraph, END
from langgraph.types import Send, interrupt
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.messages import HumanMessage, AIMessage

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
# 重审循环上限
MAX_RE_AUDIT_CYCLES = int(os.getenv("MAX_RE_AUDIT_CYCLES", "3"))
# 长对话摘要触发阈值
_SUMMARIZE_TRIGGER_LEN = 16
_SUMMARIZE_KEEP_RECENT = 6
_END_KEYWORDS = ("结束", "退出", "完成", "exit", "quit", "done", "bye", "再见")


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
# 对话循环扩展节点（checkpoint / re-audit / chat / summarize）
# ============================================================


def _make_checkpoint_node(checkpoint_name: str, stage_name: str):
    """多 Agent 版 checkpoint 节点工厂 — 使用 interrupt() 暂停等待用户输入。

    与单 agent 版类似，但读取 chapter_results / subsection_results 而非 audit_results。
    """

    async def checkpoint_node(state: MultiAgentState, *, config) -> dict:
        chapter_results = state.get("chapter_results", []) or []
        subsection_results = state.get("subsection_results", []) or []
        sections = state.get("sections", []) or []
        chapter_structure = state.get("chapter_structure", []) or []

        summary_lines: List[str] = []
        structured: dict = {}

        if checkpoint_name == "structure_review":
            summary_lines.append(f"文档解析完成，supervisor 分章 {len(chapter_structure)} 章")
            summary_lines.append(f"共 {len(sections)} 个小节待审核")
            structured["strategy"] = {
                "chapter_count": len(chapter_structure),
                "section_count": len(sections),
                "chapters": [
                    {
                        "chapter_idx": c.get("chapter_idx"),
                        "chapter_title": c.get("chapter_title", ""),
                        "section_count": len(c.get("subsection_indices", [])),
                    }
                    for c in chapter_structure[:30]
                ],
            }
        elif checkpoint_name == "post_audit":
            total_findings = sum(
                len(sub.get("findings", [])) for sub in subsection_results
            )
            critical = sum(
                1 for sub in subsection_results
                for f in sub.get("findings", [])
                if "critical" in str(f).lower() or "严重" in str(f)
            )
            summary_lines.append(f"已审核 {len(chapter_results)} 章 / {len(subsection_results)} 小节")
            summary_lines.append(f"发现 {total_findings} 个问题（含 {critical} 个严重问题）")
            if state.get("contradictions"):
                summary_lines.append(f"交叉验证发现 {len(state['contradictions'])} 处矛盾")
            structured["overview"] = {
                "chapter_count": len(chapter_results),
                "subsection_count": len(subsection_results),
                "total_findings": total_findings,
                "critical": critical,
                "contradictions_count": len(state.get("contradictions") or []),
            }
        elif checkpoint_name == "post_completion":
            summary_lines.append("审核报告已生成。")
            summary_lines.append("您可以继续提问或要求修改报告内容；回复 \"结束\" / \"exit\" 即可关闭会话。")
            structured["completion"] = {
                "chapter_count": len(chapter_results),
                "subsection_count": len(subsection_results),
                "has_report": bool(state.get("final_report")),
            }

        summary = "\n".join(summary_lines) if summary_lines else f"Stage: {stage_name}"

        _token = var_child_runnable_config.set(config)
        try:
            feedback = interrupt({
                "type": "checkpoint",
                "checkpoint": checkpoint_name,
                "stage": stage_name,
                "summary": summary,
                "chapter_count": len(chapter_results),
                "subsection_count": len(subsection_results),
                "section_count": len(sections),
                "structured": structured,
                "timestamp": _time.time(),
            })
        finally:
            var_child_runnable_config.reset(_token)

        return {
            "user_feedback": str(feedback) if feedback else "",
            "pending_checkpoint": checkpoint_name,
            "current_stage": stage_name,
        }

    return checkpoint_node


def _dispatch_re_audit_chapters(state: MultiAgentState):
    """Send API 条件边 — 仅对 re_audit_chapters 中的章节重新 fan-out audit_chapter。"""
    re_indices = set(state.get("re_audit_chapters", []) or [])
    chapter_structure = state.get("chapter_structure", []) or []
    sections = state.get("sections", []) or []
    document_type = state.get("document_type", "") or ""

    if not re_indices:
        logger.info("[Supervisor:re-audit] 无重审章节，直接合成报告")
        return [Send("synthesize_report", {"_skip": True})]

    sends: List[Send] = []
    for chap in chapter_structure:
        chapter_idx = int(chap.get("chapter_idx", 0))
        if chapter_idx not in re_indices:
            continue
        chapter_title = chap.get("chapter_title", f"章节{chapter_idx + 1}")
        sub_indices = chap.get("subsection_indices", []) or []
        subsections: List[dict] = []
        for s_idx in sub_indices:
            if 0 <= s_idx < len(sections):
                subsections.append(sections[s_idx])
        if not subsections:
            continue
        sends.append(Send(
            "audit_chapter",
            {
                "chapter_idx": chapter_idx,
                "chapter_title": chapter_title,
                "subsections": subsections,
                "subsection_indices": sub_indices,
                "doc_type": document_type,
            },
        ))

    if not sends:
        return [Send("synthesize_report", {"_skip": True})]

    logger.info(f"[Supervisor:re-audit] 重新派发 {len(sends)} 个章节")
    return sends


async def _re_audit_clear_node(state: MultiAgentState) -> dict:
    """re_audit 进入前的清理节点：递增计数器，清空 re_audit_chapters。

    实际重审由 _dispatch_re_audit_chapters Send 触发 audit_chapter 节点完成。
    """
    return {
        "re_audit_chapters": [],
        "re_audit_cycle_count": int(state.get("re_audit_cycle_count", 0)) + 1,
        "current_stage": "re_audit_dispatched",
    }


async def _dedupe_chapters_node(state: MultiAgentState) -> dict:
    """去重节点 — 按 chapter_idx 保留最新的 chapter_results 条目。

    Send API re-audit 会追加新的 chapter_results（operator.add），
    此节点确保同一 chapter_idx 只保留最后一次审核结果。
    """
    chapters = state.get("chapter_results", []) or []
    if not chapters:
        return {}
    # 按 chapter_idx 倒序去重（后写入的覆盖先写入的）
    seen: dict = {}
    for ch in chapters:
        idx = ch.get("chapter_idx")
        if idx is not None:
            seen[idx] = ch
        else:
            # 无 idx 的保留
            seen[f"_noid_{id(ch)}"] = ch
    deduped = sorted(seen.values(), key=lambda c: int(c.get("chapter_idx", 0)))
    return {"chapter_results": deduped}


def _route_after_post_audit(state: MultiAgentState) -> str:
    """post_audit checkpoint 后的路由：重审 or 合成报告。"""
    feedback = (state.get("user_feedback", "") or "").strip().lower()
    re_audit_chapters = state.get("re_audit_chapters", []) or []
    re_audit_cycle_count = int(state.get("re_audit_cycle_count", 0))

    re_audit_keywords = ["重审", "重新审核", "re-audit", "reaudit", "再审", "重新审"]

    if re_audit_chapters and re_audit_cycle_count < MAX_RE_AUDIT_CYCLES:
        return "re_audit"
    if any(kw in feedback for kw in re_audit_keywords) and re_audit_cycle_count < MAX_RE_AUDIT_CYCLES:
        return "re_audit"
    return "synthesize"


def _route_after_completion(state: MultiAgentState) -> str:
    """post_completion checkpoint 后的路由：退出 or 继续聊天。"""
    feedback = (state.get("user_feedback", "") or "").strip().lower()
    if not feedback:
        return "end"
    if any(kw in feedback for kw in _END_KEYWORDS):
        return "end"
    return "chat"


def _route_before_chat(state: MultiAgentState) -> str:
    """根据 messages 长度判断是否需要先摘要再回答。"""
    history = state.get("messages", []) or []
    if len(history) >= _SUMMARIZE_TRIGGER_LEN:
        return "summarize"
    return "chat"


def _route_post_completion(state: MultiAgentState) -> str:
    decision = _route_after_completion(state)
    if decision == "end":
        return "end"
    return _route_before_chat(state)


def _make_chat_response_node(llm: Any):
    """报告后自由对话节点 — 基于 final_report 回答用户问题。"""

    async def _node(state: MultiAgentState) -> dict:
        question = (state.get("user_feedback", "") or "").strip()
        final_report = state.get("final_report", "") or ""
        history = state.get("messages", []) or []
        summary = (state.get("summary", "") or "").strip()

        if not question:
            return {
                "current_stage": "post_completion",
                "messages": [AIMessage(content="（未收到您的提问，会话已就绪。）")],
            }

        system_parts = [
            "你是医疗器械文档审核专家。审核已完成，以下是审核报告摘要，"
            "请基于报告内容简明地回答用户问题（200 字以内，必要时引用小节编号）。",
            f"## 审核报告摘要\n{final_report[:2000]}",
        ]
        if summary:
            system_parts.append(f"## 此前对话摘要\n{summary[:1500]}")
        system_prompt = "\n\n".join(system_parts)

        try:
            response = await llm.ainvoke([
                HumanMessage(content=system_prompt),
                *history[-6:],
                HumanMessage(content=question),
            ])
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"[MultiAgent:chat_response] 失败: {e}")
            answer = f"抱歉，处理您的问题时出错：{e}"

        return {
            "current_stage": "post_completion",
            "messages": [HumanMessage(content=question), AIMessage(content=answer)],
        }

    return _node


def _make_summarize_conversation_node(llm: Any):
    """长对话摘要节点 — 压缩早期 messages 到 summary 字段。"""

    async def _node(state: MultiAgentState) -> dict:
        try:
            from langchain_core.messages import RemoveMessage
        except ImportError:
            RemoveMessage = None  # type: ignore

        history = state.get("messages", []) or []
        if len(history) < _SUMMARIZE_TRIGGER_LEN:
            return {}

        prev_summary = (state.get("summary", "") or "").strip()
        to_summarize = history[:-_SUMMARIZE_KEEP_RECENT]

        lines: List[str] = []
        for m in to_summarize:
            role = "用户" if isinstance(m, HumanMessage) else (
                "助手" if isinstance(m, AIMessage) else "系统"
            )
            content = getattr(m, "content", "") or ""
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            if content:
                lines.append(f"{role}: {content[:400]}")
        transcript = "\n".join(lines)

        system_prompt = (
            "你是对话摘要助手。请把以下医疗器械审核会话的早期对话浓缩为不超过 400 字的中文摘要，"
            "重点保留：用户提出的问题/补充信息、关键审核结论、用户对章节的反馈。"
            "如果存在此前的摘要，请把它纳入并更新。"
        )
        user_prompt_parts = []
        if prev_summary:
            user_prompt_parts.append(f"## 此前摘要\n{prev_summary}")
        user_prompt_parts.append(f"## 待压缩对话\n{transcript}")
        user_prompt = "\n\n".join(user_prompt_parts)

        try:
            response = await llm.ainvoke([
                HumanMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            new_summary = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning(f"[MultiAgent:summarize] 失败，跳过: {e}")
            return {}

        update: dict = {"summary": new_summary.strip()}
        if RemoveMessage is not None:
            remove_ops = []
            for m in to_summarize:
                mid = getattr(m, "id", None)
                if mid:
                    remove_ops.append(RemoveMessage(id=mid))
            if remove_ops:
                update["messages"] = remove_ops
        return update

    return _node


# ============================================================
# Graph 构建 / 编译
# ============================================================


def build_supervisor_graph(llm: Any, retriever: Any, conversation_mode: bool = False) -> StateGraph:
    """构建 Supervisor StateGraph (未编译)。

    Args:
        llm: ChatOpenAI 实例（顶层 LLM）
        retriever: RAGRetriever 实例
        conversation_mode: 是否启用对话循环拓扑（interrupt 检查点 + re-audit + chat 环）

    Returns:
        StateGraph: 待 compile() 的图对象
    """
    graph = StateGraph(MultiAgentState)

    # ===== 节点 =====
    graph.add_node("analyze_structure", _make_analyze_structure_node(llm))
    graph.add_node("audit_chapter", _make_audit_chapter_node(llm, retriever))
    graph.add_node("synthesize_report", _make_synthesize_report_node(llm))

    if not conversation_mode:
        # ===== 一次性审核拓扑（原有）=====
        graph.set_entry_point("analyze_structure")
        graph.add_conditional_edges(
            "analyze_structure",
            _dispatch_chapters,
            ["audit_chapter", "synthesize_report"],
        )
        graph.add_edge("audit_chapter", "synthesize_report")
        graph.add_edge("synthesize_report", END)
        return graph

    # ===== 对话循环拓扑 =====
    # checkpoint 节点
    graph.add_node("checkpoint_structure", _make_checkpoint_node("structure_review", "分章确认"))
    graph.add_node("checkpoint_post_audit", _make_checkpoint_node("post_audit", "审核结果确认"))
    graph.add_node("checkpoint_post_completion", _make_checkpoint_node("post_completion", "报告完成"))

    # re-audit / dedupe / chat / summarize 节点
    graph.add_node("re_audit_clear", _re_audit_clear_node)
    graph.add_node("dedupe_chapters", _dedupe_chapters_node)
    graph.add_node("chat_response", _make_chat_response_node(llm))
    graph.add_node("summarize_conversation", _make_summarize_conversation_node(llm))

    # ===== 边 =====
    graph.set_entry_point("analyze_structure")

    # analyze_structure → checkpoint_structure (interrupt)
    graph.add_edge("analyze_structure", "checkpoint_structure")

    # checkpoint_structure → audit_chapter ×N (Send fan-out)
    graph.add_conditional_edges(
        "checkpoint_structure",
        _dispatch_chapters,
        ["audit_chapter", "synthesize_report"],
    )

    # audit_chapter 完成 → dedupe_chapters → checkpoint_post_audit (interrupt)
    graph.add_edge("audit_chapter", "dedupe_chapters")
    graph.add_edge("dedupe_chapters", "checkpoint_post_audit")

    # checkpoint_post_audit → [re_audit_clear → re-dispatch | synthesize_report]
    graph.add_conditional_edges(
        "checkpoint_post_audit",
        _route_after_post_audit,
        {"re_audit": "re_audit_clear", "synthesize": "synthesize_report"},
    )

    # re_audit_clear → audit_chapter ×M (Send re-dispatch) → dedupe_chapters → checkpoint_post_audit (循环)
    graph.add_conditional_edges(
        "re_audit_clear",
        _dispatch_re_audit_chapters,
        ["audit_chapter", "synthesize_report"],
    )

    # synthesize_report → checkpoint_post_completion (interrupt)
    graph.add_edge("synthesize_report", "checkpoint_post_completion")

    # checkpoint_post_completion → [chat_response | summarize→chat | END]
    graph.add_conditional_edges(
        "checkpoint_post_completion",
        _route_post_completion,
        {"summarize": "summarize_conversation", "chat": "chat_response", "end": END},
    )
    graph.add_edge("summarize_conversation", "chat_response")
    graph.add_edge("chat_response", "checkpoint_post_completion")

    logger.info("[Supervisor] 对话循环拓扑构建完成 "
                f"(max_re_audit={MAX_RE_AUDIT_CYCLES})")
    return graph


def compile_supervisor_graph(
    llm: Any,
    retriever: Any,
    checkpointer: Optional[Any] = None,
    conversation_mode: bool = False,
):
    """构建并编译 Supervisor Graph。

    Args:
        llm: ChatOpenAI 实例
        retriever: RAGRetriever 实例
        checkpointer: 可选的 LangGraph checkpointer（如 AsyncSqliteSaver）
        conversation_mode: 是否启用对话循环拓扑

    Returns:
        Compiled CompiledGraph 对象（可直接 .ainvoke / .astream）
    """
    graph = build_supervisor_graph(
        llm=llm, retriever=retriever, conversation_mode=conversation_mode,
    )

    if checkpointer is not None:
        compiled = graph.compile(checkpointer=checkpointer)
        logger.info(f"[Supervisor] Graph 编译完成（带 checkpointer, conversation={conversation_mode}）")
    else:
        compiled = graph.compile()
        logger.info("[Supervisor] Graph 编译完成（无 checkpointer）")

    return compiled
