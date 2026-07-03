"""多 Agent 分步协作状态定义

设计参考:
- `agent_state.py` 的 AuditState（向后兼容字段保持名称一致）
- LangChain Subagents 模式：每个子 Agent 在其上下文中独立运行
- LangGraph Send API：使用 `Annotated[list, operator.add]` reducer
  自动汇集多个并行章节的审核结果
"""
import operator
from typing import TypedDict, List, Annotated, Optional, Any

from langgraph.graph.message import add_messages


class MultiAgentState(TypedDict):
    """多 Agent 协作的全局状态。

    关键字段说明:

    - chapter_structure : Structure Analyzer Agent 产出的章节归属
      [{chapter_idx, chapter_title, subsection_indices, section_count}]
    - chapter_results   : Chapter Auditor Agent 并行写入（reducer 自动汇集）
      [{chapter_idx, chapter_title, subsection_results, chapter_summary, ...}]
    - subsection_results: 展平后的逐小节结果（与 AuditState.audit_results 兼容）
    """

    # ====== 输入：文档信息 ======
    document_text: str
    document_type: str
    outline: List[dict]
    sections: List[dict]

    # ====== Structure Analyzer Agent 产出 ======
    outline_summary: dict
    chapter_structure: List[dict]

    # ====== Chapter Auditor Agent 并行写入（Send API + reducer 汇集）======
    chapter_results: Annotated[List[dict], operator.add]

    # ====== Report Synthesizer 中间产出 ======
    subsection_results: List[dict]
    contradictions: Optional[List[dict]]
    final_report: Optional[str]
    finished: bool

    # ====== 进度统计 ======
    total_chapters: int
    total_sections: int
    total_agent_steps: int

    # ====== 消息历史（用于 streaming/对话）======
    messages: Annotated[list, add_messages]

    # ====== 对话模式扩展（可选）======
    conversation_mode: bool
    current_stage: str
    user_feedback: str
    pending_checkpoint: str
    conversation_context: str

    # ====== 完整对话循环扩展（re-audit / skip / chat memory）======
    re_audit_chapters: List[int]       # 需重审的章节索引（chapter_idx）
    revision_requests: List[dict]      # 用户修订请求（含 standard_override）
    skip_chapters: List[int]           # 需跳过的章节索引
    re_audit_cycle_count: int          # 重审循环计数（上限 MAX_RE_AUDIT_CYCLES）
    summary: str                       # 长对话摘要（chat 环用）
    awaiting_input: bool               # 是否等待用户输入（free chat 用）


def make_multi_agent_initial_state(
    document_text: str,
    document_type: str,
    outline: Optional[List[dict]] = None,
    sections: Optional[List[dict]] = None,
    conversation_mode: bool = False,
) -> dict:
    """创建多 Agent 协作的初始状态。"""
    return {
        # 输入
        "document_text": document_text,
        "document_type": document_type,
        "outline": outline or [],
        "sections": sections or [],
        # 结构分析
        "outline_summary": {},
        "chapter_structure": [],
        # 章节审核结果
        "chapter_results": [],
        # 综合
        "subsection_results": [],
        "contradictions": None,
        "final_report": None,
        "finished": False,
        # 统计
        "total_chapters": 0,
        "total_sections": 0,
        "total_agent_steps": 0,
        # 消息
        "messages": [],
        # 对话模式
        "conversation_mode": conversation_mode,
        "current_stage": "intake" if conversation_mode else "",
        "user_feedback": "",
        "pending_checkpoint": "",
        "conversation_context": "",
        # 完整对话循环
        "re_audit_chapters": [],
        "revision_requests": [],
        "skip_chapters": [],
        "re_audit_cycle_count": 0,
        "summary": "",
        "awaiting_input": False,
    }


class ChapterAuditPacket(TypedDict, total=False):
    """单个 Send 调用所携带的章节审核包。"""
    chapter_idx: int
    chapter_title: str
    chapter_breadcrumb: str
    subsections: List[dict]            # 该章节下属小节（可直接审核的最小单元）
    subsection_indices: List[int]      # 在原始 sections 列表中的索引
    doc_type: str
    knowledge_context: str             # 该章节相关的预检索知识库内容
