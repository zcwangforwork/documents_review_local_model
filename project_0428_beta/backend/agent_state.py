"""
Agent State — LangGraph 状态图的核心数据类型
"""
import operator
from typing import TypedDict, List, Annotated, Optional, Any
from langgraph.graph.message import add_messages


class AuditState(TypedDict):
    """文档审核 Agent 的全局状态"""

    # ====== 文档信息 (初始化时填充) ======
    document_text: str
    document_type: str
    outline: List[dict]
    sections: List[dict]
    # sections 元素: {"title": str, "content": str,
    #                 "level": int, "breadcrumb": str}

    # ====== 文档结构概览 (analyze_outline_node 节点产出) ======
    outline_summary: dict
    # {
    #   "chapter_count": int,        # 一级章节数
    #   "section_count": int,        # 待审核小节数（即 len(sections)）
    #   "tree_text": str,            # 章节树文本（用于报告展示）
    #   "chapters": [                # 各章节的小节归属
    #     {"title": str, "level": int, "subsection_indices": [int, ...]}
    #   ]
    # }

    # ====== 审核策略 (plan_strategy 节点产出) ======
    audit_plan: dict
    # {section_idx: {"depth": "quick"|"standard"|"deep",
    #                 "focus_standards": [...]}}

    # ====== 预检索知识库上下文 (plan_strategy 节点产出) ======
    knowledge_context: str
    # 从向量库预检索到的法规条款原文，供 audit_section 直接使用

    # ====== 审核进度 ======
    current_section_idx: int
    round_count: int
    total_agent_steps: int

    # ====== 当前小节审核中间结果 (audit_section 产出, evaluate_result 消费) ======
    confidence: int
    key_findings: List[str]
    current_section_result_content: str
    # 顺序模式下，audit_section_node 把当前小节的完整 LLM 审核文本暂存于此，
    # 由 evaluate_result_node 持久化到 audit_results 对应条目的 result_content 字段

    # ====== 跨章节上下文 (operator.add 实现追加而非覆盖) ======
    global_context: Annotated[List[str], operator.add]

    # ====== 审核结果 ======
    audit_results: List[dict]
    # [{"section_idx": 0, "title": "...", "depth": "standard",
    #   "findings": [...], "score": 7.5, "confidence": 4}]

    # ====== 消息历史 (LangGraph add_messages reducer) ======
    messages: Annotated[list, add_messages]

    # ====== 对话摘要（Phase 2 长程记忆）======
    # 当 messages 长度超过阈值时，由 _summarize_conversation_node 把
    # 较早的消息压缩为一段中文摘要，避免 token 无限膨胀。
    # _chat_response_node 在生成回复时优先消费 summary + 最近若干条消息。
    summary: str

    # ====== 最终输出 ======
    contradictions: Optional[List[dict]]
    final_report: Optional[str]
    finished: bool

    # ====== 对话交互层（Conversation Mode） ======
    current_stage: str                    # 当前所处阶段: intake/strategy/audit/cross_validate/report/revision
    user_feedback: str                    # 用户最新反馈文本
    pending_checkpoint: str               # 当前等待用户的检查点名称
    stage_history: List[dict]             # 各阶段执行记录 [{stage, timestamp, summary}]
    revision_requests: List[dict]         # 用户修订请求队列 [{section_idx, request, standard}]
    re_audit_sections: List[int]          # 需要重新审核的章节索引
    conversation_context: str             # 对话上下文（用户补充信息，注入重审时合并到 System Prompt）
    skip_sections: List[int]              # 跳过的章节索引
    re_audit_cycle_count: int             # 重审循环计数（上限3次）
    conversation_mode: bool               # 是否为对话模式


def make_initial_state(
    document_text: str,
    document_type: str,
    outline: Optional[List[dict]] = None,
    sections: Optional[List[dict]] = None,
    conversation_mode: bool = False,
) -> dict:
    """创建 Agent 的初始状态"""
    state = {
        "document_text": document_text,
        "document_type": document_type,
        "outline": outline or [],
        "sections": sections or [],
        "outline_summary": {},
        "audit_plan": {},
        "knowledge_context": "",
        "global_context": [],
        "current_section_idx": 0,
        "round_count": 0,
        "total_agent_steps": 0,
        "confidence": 3,
        "key_findings": [],
        "current_section_result_content": "",
        "audit_results": [],
        "messages": [],
        "summary": "",
        "contradictions": None,
        "final_report": None,
        "finished": False,
        # Conversation mode fields
        "current_stage": "intake" if conversation_mode else "",
        "user_feedback": "",
        "pending_checkpoint": "",
        "stage_history": [],
        "revision_requests": [],
        "re_audit_sections": [],
        "conversation_context": "",
        "skip_sections": [],
        "re_audit_cycle_count": 0,
        "conversation_mode": conversation_mode,
    }
    return state
