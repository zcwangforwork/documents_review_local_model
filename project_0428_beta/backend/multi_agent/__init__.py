"""多 Agent 分步协作架构（LangGraph Subagents + Send API 模式）

将单 Agent StateGraph 改造为多 Agent 协作系统：
  Structure Analyzer Agent  → 分析章节结构
  Chapter Auditor Agent ×N → 章节级并行（Send API fan-out）
  Report Synthesizer        → 汇总章节结果生成最终报告

公共导出:
- MultiAgentState : 多 Agent 状态类型
- make_multi_agent_initial_state : 初始状态工厂
- build_supervisor_graph : Supervisor StateGraph 构建函数
- compile_supervisor_graph : 编译图（可选 checkpointer）
"""

from multi_agent.state import (
    MultiAgentState,
    make_multi_agent_initial_state,
)
from multi_agent.supervisor_graph import (
    build_supervisor_graph,
    compile_supervisor_graph,
)

__all__ = [
    "MultiAgentState",
    "make_multi_agent_initial_state",
    "build_supervisor_graph",
    "compile_supervisor_graph",
]
