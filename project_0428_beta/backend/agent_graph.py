"""
StateGraph 构建与编译 — LangGraph 文档审核 Agent 的核心
包含图拓扑、全部节点函数、路由函数、checkpoint 持久化

支持两种拓扑模式（由 PARALLEL_AUDIT_ENABLED 环境变量控制）:

  并行模式（默认）:
    plan_strategy → batch_audit_all_sections (内部 asyncio.gather + Semaphore 并发)
                  → cross_validate → generate_report → END

  顺序模式（向后兼容）:
    plan_strategy → audit_section (seq) → evaluate_result (纯解析)
                  → update_context → next/done → cross_validate → report
"""
import os
import re
import json
import asyncio
import logging
from typing import Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)


# ============================================================
# 领域专项审核 Prompt 注入（方案一：复用经典模式检查清单）
# ------------------------------------------------------------
# 从 rag_retriever.RAGRetriever.SECTION_PROMPTS 取对应领域的检查清单
# prompt，剔除尾部的「严重度评级」「章节量化评分」段（与方案一致：
# 报告不再输出评分/评级），再注入 Agent 模式所需的 JSON metadata 块。
# 失败时回退到原通用英文 prompt，保证向后兼容。
# ============================================================

# 匹配领域 prompt 末尾的「### 严重度评级」和「### 章节量化评分」整段
# （含后续的评分表格）。非贪婪到 prompt 末尾。
_DOMAIN_SCORE_TAIL_RE = re.compile(
    r"###\s*严重度评级.*?\Z",
    re.DOTALL,
)


def _get_domain_prompt(doc_type: str) -> Optional[str]:
    """按 doc_type 取经典模式领域专项 prompt（已剔除评分/评级段）。

    Returns:
        净化后的领域 prompt（含检查清单 + 输出格式，无评分评级），
        或 None 表示无对应领域 / 导入失败 → 调用方回退通用 prompt。
    """
    try:
        from rag_retriever import RAGRetriever
    except Exception as e:
        logger.warning(f"导入 RAGRetriever 失败，回退通用 prompt: {e}")
        return None

    prompts = getattr(RAGRetriever, "SECTION_PROMPTS", None)
    if not prompts:
        return None

    raw = prompts.get(doc_type) or prompts.get("general")
    if not raw:
        return None

    # 剔除尾部的「严重度评级」+「章节量化评分」整段
    cleaned = _DOMAIN_SCORE_TAIL_RE.sub("", raw).rstrip() + "\n"

    # 追加 Agent 模式必需的 JSON metadata 块要求
    cleaned += (
        "\n## 审核元数据（必须追加在响应末尾）\n"
        "在响应最末尾追加一个 JSON 元数据块，用 ```json 包裹：\n"
        "```json\n"
        "{\"confidence\": <1-5>, \"key_findings\": [\"<发现项摘要>\", ...]}\n"
        "```"
    )
    return cleaned
from langchain_openai import ChatOpenAI
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.runnables import RunnableConfig

from agent_state import AuditState

logger = logging.getLogger(__name__)

# ============== 并行审核配置 ==============

PARALLEL_AUDIT_ENABLED = os.getenv("PARALLEL_AUDIT_ENABLED", "true").lower() == "true"
AUDIT_CONCURRENCY = int(os.getenv("AUDIT_CONCURRENCY", "5"))

# Phase 3.1: 是否启用 tool-calling agent 变体审核单章节。
# 启用后，单章节审核会通过 create_agent + agent_tools.py 暴露的工具集
# (rag_search / check_regulation / assess_completeness 等) 让 LLM 自主决定调用顺序。
# 默认关闭以保持向后兼容，可通过 USE_TOOL_CALLING_AUDIT=1 打开。
USE_TOOL_CALLING_AUDIT = os.getenv("USE_TOOL_CALLING_AUDIT", "0") == "1"

# Phase 3.3: 是否启用 LangGraph Send API 进行章节并行审核。
# 启用时将通过子图 fan-out 到多个 audit_one 节点并行执行；默认关闭，
# 仍走 asyncio.gather 路径以保持原有性能与稳定性。
USE_SEND_API_AUDIT = os.getenv("USE_SEND_API_AUDIT", "0") == "1"

# ============== LLM 工厂 ==============

_DEFAULT_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "ollama"
)


def create_llm(temperature: float = 0.3, max_tokens: int = 4096, **kwargs) -> ChatOpenAI:
    """创建 LLM 实例 — 指向本地 Ollama qwen3.5:122b（OpenAI 兼容接口）

    streaming=True 让模型走流式接口，配合 LangGraph 的 stream_mode="messages"
    可在前端按 token 实时渲染审核内容（避免长节点 UI 假死）。

    Extra kwargs (e.g. request_timeout) are forwarded to ChatOpenAI.
    """
    params = dict(
        model=os.getenv("OPENAI_MODEL", "qwen3.5:122b"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11435/v1"),
        api_key=_DEFAULT_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        streaming=True,
        request_timeout=600,
    )
    params.update(kwargs)
    return ChatOpenAI(**params)


# ============== JSON 提取工具 ==============

def _extract_json(response_text: str, default: dict) -> dict:
    """从 LLM 响应中提取 JSON，带 fallback"""
    match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return default


# ============== 节点函数 ==============

from doc_processor import llm_parse_outline  # LLM 文档大纲解析（已统一到 doc_processor）


def analyze_outline_node(state: AuditState, llm: Optional[ChatOpenAI] = None) -> dict:
    """
    文档结构分析节点 — 在所有审核动作之前，识别并统计文档章节/小节。

    工作模式：
    - **主路径**：直接使用 main.py 已通过 `parse_document_structure` (LLM优先/正则回退)
      提供的 outline 与 sections，不在此节点重复调用 LLM。
    - **兜底**：当 sections 为空且 document_text 非空时，尝试 LLM 应急解析。
    """
    document_text = state.get("document_text", "") or ""
    outline = state.get("outline", []) or []
    sections = state.get("sections", []) or []

    # ----- 仅当 sections 为空时兜底尝试 LLM（极少数情况）-----
    llm_used = False
    if not sections and llm is not None and document_text.strip():
        try:
            new_outline, new_sections = llm_parse_outline(document_text, llm)
            outline = new_outline
            sections = new_sections
            llm_used = True
            logger.info(
                f"analyze_outline: LLM 兜底 — 识别 {len(outline)} 章节, {len(sections)} 小节"
            )
        except Exception as e:
            logger.warning(
                f"analyze_outline: LLM 兜底失败 ({type(e).__name__}: {e})"
            )

    # ----- 一级章节列表（level==1 的顶层节点）-----
    top_chapters = [n for n in outline if n.get("level", 1) == 1]
    if not top_chapters:
        # 文档没有一级标题（罕见，多数为偏离规范的扫描件），把所有顶层节点都视为一级
        top_chapters = list(outline)

    chapter_count = len(top_chapters)

    # ----- 计算每个章节包含的小节索引（按 breadcrumb 前缀匹配）-----
    chapters_info: list = []
    for chap in top_chapters:
        chap_title = chap.get("title", "")
        sub_indices = []
        for idx, sec in enumerate(sections):
            breadcrumb = sec.get("breadcrumb", "") or ""
            # breadcrumb 形如 "第一章 / 1.1 标题 / 1.1.1 小节"，首段即一级章节标题
            head = breadcrumb.split(" / ")[0] if breadcrumb else ""
            if head == chap_title:
                sub_indices.append(idx)
        chapters_info.append({
            "title": chap_title,
            "level": chap.get("level", 1),
            "subsection_indices": sub_indices,
        })

    # 兜底：若没匹配上任何小节（breadcrumb 风格不一致），把所有 sections 平铺
    matched_total = sum(len(c["subsection_indices"]) for c in chapters_info)
    if matched_total == 0 and sections:
        chapters_info = [{
            "title": "全文",
            "level": 1,
            "subsection_indices": list(range(len(sections))),
        }]
        chapter_count = 1

    # ----- 渲染章节树文本（用于报告直接显示）-----
    tree_lines = []
    tree_lines.append(f"- **总计**：识别 {chapter_count} 个章节，共 {len(sections)} 个待审核小节")
    tree_lines.append("")
    tree_lines.append("**章节结构**：")
    for c_idx, chap in enumerate(chapters_info, 1):
        sub_count = len(chap["subsection_indices"])
        tree_lines.append(f"{c_idx}. {chap['title']}（包含 {sub_count} 个小节）")
        for s_idx in chap["subsection_indices"]:
            sec = sections[s_idx]
            title = sec.get("title", "")
            breadcrumb = sec.get("breadcrumb", "") or title
            content_len = sec.get("content_length", len(sec.get("content", "")))
            tree_lines.append(f"   - [{s_idx}] {breadcrumb}（{content_len} 字符）")

    tree_text = "\n".join(tree_lines)

    summary = {
        "chapter_count": chapter_count,
        "section_count": len(sections),
        "tree_text": tree_text,
        "chapters": chapters_info,
        "source": "llm" if llm_used else "regex",
    }

    logger.info(
        f"analyze_outline: 识别 {chapter_count} 个章节, {len(sections)} 个小节 (source={'llm' if llm_used else 'regex'})"
    )

    result = {
        "outline_summary": summary,
        "messages": [AIMessage(
            content=(f"[文档结构分析完成 / source={'LLM' if llm_used else '正则'}] "
                     f"共 {chapter_count} 个章节，{len(sections)} 个待审核小节。")
        )],
    }
    # LLM 模式下，把新的 outline/sections 写回 state 供下游节点使用
    if llm_used:
        result["outline"] = outline
        result["sections"] = sections
    return result


def plan_strategy_node(state: AuditState, llm: ChatOpenAI, retriever) -> dict:
    """
    策略规划节点 — 解析大纲 + 预检索法规上下文

    1. LLM 分析大纲，为每个 section 制定审核深度和焦点标准
    2. 程序化预检索所有相关法规条款（非工具调用），存入 knowledge_context
    3. 后续 audit_section_node 直接使用预检索上下文，无需工具循环
    """
    sections = state.get("sections", [])
    doc_type = state.get("document_type", "")

    if not sections:
        return {"audit_plan": {}, "knowledge_context": ""}

    # ----- 步骤1: LLM 制定审核策略 -----
    sections_summary = []
    for i, sec in enumerate(sections):
        content_preview = sec.get("content_preview", sec.get("content", "")[:300])
        sections_summary.append(
            f"### Section {i}: {sec.get('title', 'Untitled')} "
            f"(面包屑: {sec.get('breadcrumb', '')})\n"
            f"Content preview: {content_preview}\n"
            f"Content length: {sec.get('content_length', len(sec.get('content', '')))} chars"
        )

    sections_text = "\n\n".join(sections_summary)

    prompt = f"""You are an audit strategist. Analyze the document outline below and create a differentiated audit plan.

Document type: {doc_type}
Total sections: {len(sections)}

## Document Outline
{sections_text}

## Your Task
For each section, assess:
1. **Completeness score (1-5)**: How complete does this section appear based on title and content length?
   - 4-5: Well-structured, substantial content → quick audit
   - 2-3: Moderate, some gaps suspected → standard audit
   - 1: Sparse, clearly incomplete → deep audit

2. **Focus standards**: Which standards are most relevant? (ISO 14971:2019, ISO 13485:2016, IEC 62304, MDR 2017/745, NMPA GMP, GB/T 42062-2022)

## Output Format (JSON only)
```json
{{
  "plan": {{
    "0": {{"depth": "standard", "completeness": 3, "focus_standards": ["ISO 14971:2019"]}},
    "1": {{"depth": "deep", "completeness": 1, "focus_standards": ["ISO 14971:2019", "IEC 62304"]}}
  }}
}}
```"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        audit_plan_raw = _extract_json(content, {}).get("plan", {})
        # Normalize keys from string (JSON) to int
        audit_plan = {}
        for k, v in audit_plan_raw.items():
            try:
                audit_plan[int(k)] = v
            except (ValueError, TypeError):
                audit_plan[k] = v
    except Exception as e:
        logger.error(f"plan_strategy 失败: {e}")
        audit_plan = {}

    if not audit_plan:
        audit_plan = {
            i: {"depth": "standard", "completeness": 3, "focus_standards": []}
            for i in range(len(sections))
        }

    # ----- 步骤2: 预检索法规上下文 -----
    all_standards = set()
    for plan_entry in audit_plan.values():
        for std in plan_entry.get("focus_standards", []):
            all_standards.add(std)

    if not all_standards:
        all_standards = {"ISO 14971:2019", "ISO 13485:2016", "IEC 62304"}

    knowledge_parts = []
    search_queries = {
        "ISO 14971:2019": [
            "ISO 14971 风险管理 危害识别 风险分析 风险评价 风险控制",
            "ISO 14971 剩余风险 受益风险分析 风险管理报告",
            "ISO 14971:2019 风险控制措施 验证 有效性",
        ],
        "ISO 13485:2016": [
            "ISO 13485 设计开发 设计控制 设计评审 设计验证",
            "ISO 13485:2016 文件控制 记录控制 质量管理体系",
            "ISO 13485 采购 生产 过程控制 可追溯性",
        ],
        "IEC 62304": [
            "IEC 62304 软件安全分级 软件生命周期 软件需求",
            "IEC 62304 软件架构 软件详细设计 软件单元测试",
            "IEC 62304 软件配置管理 软件问题解决 软件维护",
        ],
        "MDR 2017/745": [
            "MDR 2017/745 医疗器械法规 技术文档 符合性评估",
            "MDR 医疗器械分类 临床评价 上市后监督",
        ],
        "NMPA GMP": [
            "NMPA 医疗器械生产质量管理规范 GMP 现场检查",
            "NMPA GMP 生产管理 质量控制 设备管理 供应商管理",
        ],
        "GB/T 42062-2022": [
            "GB/T 42062 医疗器械风险管理 风险分析 风险评价",
            "GB/T 42062-2022 风险管理过程 风险控制 综合剩余风险",
        ],
    }

    retrieved_count = 0
    for standard in all_standards:
        queries = search_queries.get(standard, [f"{standard} 医疗器械 要求 条款"])
        for query in queries[:2]:
            try:
                docs = retriever.search_sync(query, top_k=3)
                if docs:
                    for doc in docs:
                        source = doc.get("source", "unknown")
                        text = doc.get("text", "")[:1000]
                        knowledge_parts.append(f"### [{standard}] {source}\n{text}")
                        retrieved_count += 1
            except Exception as e:
                logger.warning(f"预检索失败 [{standard}] {query}: {e}")

    seen = set()
    unique_parts = []
    for part in knowledge_parts:
        key = part[:100]
        if key not in seen:
            seen.add(key)
            unique_parts.append(part)

    knowledge_context = "\n\n---\n\n".join(unique_parts)
    logger.info(f"plan_strategy: 预检索完成, {retrieved_count} 条, 去重后 {len(unique_parts)} 条")

    return {
        "audit_plan": audit_plan,
        "knowledge_context": knowledge_context,
    }


def _retrieve_section_knowledge(section: dict, doc_type: str, retriever, top_k: int = 5) -> str:
    """为单个小节检索相关知识库内容，构建专用审核参考"""
    title = section.get("title", "")
    breadcrumb = section.get("breadcrumb", "")
    content = section.get("content", "")

    query_parts = []
    if breadcrumb:
        query_parts.append(breadcrumb)
    if title and title not in (breadcrumb or ""):
        query_parts.append(title)
    content_snippet = content[:500].strip()
    if content_snippet:
        query_parts.append(content_snippet)

    query = " ".join(query_parts)[:1000]

    try:
        # 优先使用带 LLM listwise 重排的检索（方案A），回退到普通检索
        if hasattr(retriever, "search_sync_with_rerank"):
            docs = retriever.search_sync_with_rerank(query, top_k=top_k)
        else:
            docs = retriever.search_sync(query, top_k=top_k)
        if not docs:
            return ""

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.get("source", "unknown")
            text = doc.get("text", "")[:800]
            parts.append(f"**[{source}]**\n{text}")

        return "\n\n---\n\n".join(parts)
    except Exception as e:
        logger.warning(f"逐小节检索失败 '{title}': {e}")
        return ""


def _build_audit_system_prompt(
    section: dict, depth: str, focus_standards: list,
    doc_type: str, knowledge_context: str,
    global_context: Optional[list] = None,
    round_count: Optional[int] = None,
    conversation_context: str = "",
    section_knowledge: str = "",
) -> str:
    """构建审核 System Prompt — 统一版本，支持顺序/并行/对话三种模式"""
    context_block = ""
    if global_context:
        context_block = "## Global Context (findings from previous sections)\n"
        for ctx in global_context[-5:]:
            context_block += f"- {ctx}\n"

    extra_context_block = ""
    if conversation_context:
        extra_context_block = (
            "## Additional Context from User\n"
            f"{conversation_context[:2000]}\n"
        )

    depth_guidance = {
        "quick": "Quick audit: basic compliance check against key requirements.",
        "standard": "Standard audit: thorough review against relevant standards.",
        "deep": "Deep audit: exhaustive analysis with specific clause references required.",
    }

    knowledge_block = ""
    if knowledge_context:
        max_knowledge = 4000
        knowledge_block = knowledge_context[:max_knowledge]
        if len(knowledge_context) > max_knowledge:
            knowledge_block += "\n\n[知识库上下文已截断]"

    section_knowledge_block = ""
    if section_knowledge:
        max_section = 4000
        section_knowledge_block = (
            "## Section-Specific Regulatory References\n"
            "The following clauses were retrieved specifically for this section. "
            "Use them as the PRIMARY audit reference:\n\n"
            + section_knowledge[:max_section]
        )
        if len(section_knowledge) > max_section:
            section_knowledge_block += "\n\n[小节专用知识库上下文已截断]"

    round_info = ""
    if round_count is not None:
        round_info = f"- Re-audit round: {round_count} (round {round_count + 1} of max 2)\n"

    # 方案一：优先复用经典模式领域专项 prompt（含逐条检查清单），
    # 已剔除尾部的评分/评级段并追加 JSON metadata 要求。
    # 失败时回退原通用英文 prompt，保证向后兼容。
    domain_prompt = _get_domain_prompt(doc_type)
    if domain_prompt:
        return f"""## 审核上下文
- 文档类型: {doc_type}
- 当前小节: {section.get('title', 'Untitled')}（{section.get('breadcrumb', '')}）
- 审核深度: {depth} — {depth_guidance.get(depth, depth_guidance['standard'])}
- 关注标准: {', '.join(focus_standards) if focus_standards else '所有相关标准'}
{round_info}
{context_block}
{extra_context_block}
## 预检索法规知识（通用）
{knowledge_block if knowledge_block else "[无预检索知识 — 基于通用原则审核]"}

{section_knowledge_block}

---

{domain_prompt}"""

    # 回退：原通用英文 prompt
    return f"""You are a medical device documentation audit expert specializing in insulin pump systems.

## Audit Context
- Document type: {doc_type}
- Current section: {section.get('title', 'Untitled')} ({section.get('breadcrumb', '')})
- Audit depth: {depth} — {depth_guidance.get(depth, depth_guidance['standard'])}
- Focus standards: {', '.join(focus_standards) if focus_standards else 'All relevant standards'}
{round_info}
{context_block}
{extra_context_block}
## Pre-Retrieved Regulatory Knowledge (General)
{knowledge_block if knowledge_block else "[No pre-retrieved knowledge available — audit based on general principles]"}

{section_knowledge_block}

## Audit Principles
1. Compare section content against the pre-retrieved regulatory requirements above
2. Identify gaps, inconsistencies, and missing elements
3. Rate severity of each finding: critical / major / minor / suggestion
4. Cite specific standard clauses in findings (from the pre-retrieved knowledge)
5. Provide actionable corrective recommendations

## Output Format
Output the audit result with these sections:
1. **原文摘要**: Brief summary of section content
2. **标准要求**: Key regulatory requirements from the pre-retrieved knowledge
3. **发现项**: Numbered findings with severity (critical/major/minor/suggestion)
4. **差距分析**: Detailed gap analysis with standard clause references
5. **修改建议**: Specific, actionable recommendations

At the very end of your response, you MUST append a JSON metadata block wrapped in ```json:
```json
{{"confidence": <1-5>, "key_findings": ["<finding summary>", ...]}}
```"""


async def batch_audit_all_sections_node(state: AuditState, llm: ChatOpenAI, retriever) -> dict:
    """
    批量并发审核所有章节 — 替代原有的 audit_section 顺序循环

    使用 asyncio.gather + Semaphore 实现并发控制，
    单个节点内完成所有章节的审核。每个章节独立审核（不依赖 global_context）。

    预期加速: 3-5x（41章节 / 5并发 ≈ 9轮，每轮约等于最慢章节的耗时）
    """
    sections = state.get("sections", [])
    audit_plan = state.get("audit_plan", {})
    knowledge_context = state.get("knowledge_context", "")
    doc_type = state.get("document_type", "")
    total_steps = state.get("total_agent_steps", 0)

    if not sections:
        return {
            "audit_results": [],
            "global_context": [],
            "messages": [AIMessage(content="[No sections to audit]")],
        }

    semaphore = asyncio.Semaphore(AUDIT_CONCURRENCY)
    logger.info(f"batch_audit: 开始并行审核 {len(sections)} 章节, 并发数={AUDIT_CONCURRENCY}")

    async def audit_one_section(idx: int) -> dict:
        async with semaphore:
            section = sections[idx]
            depth_config = audit_plan.get(idx, {})
            depth = depth_config.get("depth", "standard")
            focus_standards = depth_config.get("focus_standards", [])

            # 逐小节检索：为当前小节从知识库检索专用法规条款
            section_knowledge = _retrieve_section_knowledge(section, doc_type, retriever)

            section_llm = create_llm()

            system_prompt = _build_audit_system_prompt(
                section=section,
                depth=depth,
                focus_standards=focus_standards,
                doc_type=doc_type,
                knowledge_context=knowledge_context,
                section_knowledge=section_knowledge,
            )

            content_text = section.get('content', '')
            user_prompt = (
                f"## Current Section\n"
                f"**Title**: {section.get('title', 'Untitled')}\n"
                f"**Breadcrumb**: {section.get('breadcrumb', '')}\n"
                f"**Content**:\n{content_text[:8000]}"
            )

            try:
                response = await section_llm.ainvoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])
                result_content = response.content if hasattr(response, "content") else str(response)
            except Exception as e:
                logger.error(f"batch_audit LLM 调用失败 (section {idx}): {e}")
                result_content = f"[LLM Error: {e}]"

            metadata = _extract_json(result_content, {"confidence": 3, "key_findings": []})

            return {
                "section_idx": idx,
                "title": section.get("title", ""),
                "breadcrumb": section.get("breadcrumb", ""),
                "depth": depth,
                "findings": metadata.get("key_findings", []),
                "confidence": int(metadata.get("confidence", 3)),
                "round_count": 1,
                "result_content": result_content,
            }

    # 创建所有任务并并发执行
    all_indices = list(range(len(sections)))
    try:
        all_results = await asyncio.gather(*[audit_one_section(i) for i in all_indices])
    except Exception as e:
        logger.error(f"batch_audit 并发执行异常: {e}")
        # Fallback: 逐个收集成功的结果
        all_results = []
        for i in all_indices:
            try:
                result = await audit_one_section(i)
                all_results.append(result)
            except Exception as e2:
                logger.error(f"batch_audit section {i} fallback 也失败: {e2}")
                all_results.append({
                    "section_idx": i,
                    "title": sections[i].get("title", ""),
                    "breadcrumb": sections[i].get("breadcrumb", ""),
                    "depth": "standard",
                    "findings": [],
                    "confidence": 1,
                    "round_count": 1,
                    "result_content": f"[Error: {e2}]",
                })

    # 按 section_idx 排序
    all_results.sort(key=lambda r: int(r.get("section_idx", 0)))

    # 提取 global context (用于后续报告生成)
    global_context = []
    for r in all_results:
        title = r.get("title", "")
        for finding in r.get("findings", [])[:2]:
            global_context.append(f"[{r['section_idx']}: {title}] {finding}")

    logger.info(f"batch_audit: 完成 {len(all_results)} 章节审核, global_context 条目: {len(global_context)}")

    return {
        "audit_results": all_results,
        "global_context": global_context,
        "total_agent_steps": total_steps + len(sections),
        "messages": [AIMessage(content=f"[All {len(sections)} sections audited in parallel]")],
    }


def audit_section_node(state: AuditState, llm: ChatOpenAI, retriever) -> dict:
    """
    审核执行节点 — 直接LLM调用（不绑定工具, 无工具循环）

    使用 plan_strategy 预检索的 knowledge_context 直接审核当前小节。
    输出结构化 JSON 元数据 (confidence + key_findings), 供 evaluate_result 直接解析。
    每节仅消耗 1 个 super-step。
    """
    idx = state.get("current_section_idx", 0)
    sections = state.get("sections", [])
    if idx >= len(sections):
        return {"messages": [AIMessage(content="[All sections audited]")], "confidence": 3, "key_findings": []}

    section = sections[idx]
    depth_config = state.get("audit_plan", {}).get(idx, {})
    depth = depth_config.get("depth", "standard")
    focus_standards = depth_config.get("focus_standards", [])
    round_count = state.get("round_count", 0)
    knowledge_context = state.get("knowledge_context", "")
    total_steps = state.get("total_agent_steps", 0)

    # 逐小节检索
    section_knowledge = _retrieve_section_knowledge(section, state.get("document_type", ""), retriever)

    system_prompt = _build_audit_system_prompt(
        section=section,
        depth=depth,
        focus_standards=focus_standards,
        doc_type=state.get("document_type", ""),
        knowledge_context=knowledge_context,
        global_context=state.get("global_context", []),
        round_count=round_count,
        section_knowledge=section_knowledge,
    )

    user_prompt = f"""## Current Section to Audit

**Title**: {section.get('title', 'Untitled')}
**Breadcrumb**: {section.get('breadcrumb', '')}
**Content**:
```
{section.get('content', '')[:8000]}
```

Please audit this section using the pre-retrieved regulatory knowledge provided above."""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        result_content = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"audit_section LLM 调用失败 (section {idx}): {e}")
        result_content = f"[LLM 调用失败: {str(e)}]"

    # 解析 JSON 元数据
    metadata = _extract_json(result_content, {"confidence": 3, "key_findings": []})
    confidence = metadata.get("confidence", 3)
    key_findings = metadata.get("key_findings", [])

    return {
        "messages": [AIMessage(content=result_content)],
        "confidence": int(confidence),
        "key_findings": key_findings,
        "total_agent_steps": total_steps + 1,
        "current_section_result_content": result_content,
    }


def evaluate_result_node(state: AuditState) -> dict:
    """
    评估节点 — 纯解析函数（无 LLM 调用）

    从 state 中读取 audit_section 产出的 confidence 和 key_findings，
    直接构建 audit_results 条目。LLM 调用次数: 0。
    同时持久化 audit_section_node 暂存的 result_content（完整 LLM 审核文本），
    使顺序模式与并行模式在最终 audit_results 上的结构一致。
    """
    idx = state.get("current_section_idx", 0)
    round_count = state.get("round_count", 0)
    confidence = state.get("confidence", 3)
    key_findings = state.get("key_findings", [])
    result_content = state.get("current_section_result_content", "")

    sections = state.get("sections", [])
    section_title = sections[idx].get("title", "") if idx < len(sections) else ""
    section_breadcrumb = sections[idx].get("breadcrumb", "") if idx < len(sections) else ""
    depth_config = state.get("audit_plan", {}).get(idx, {})
    depth = depth_config.get("depth", "standard")

    new_result = {
        "section_idx": idx,
        "title": section_title,
        "breadcrumb": section_breadcrumb,
        "depth": depth,
        "findings": key_findings,
        "confidence": confidence,
        "round_count": round_count + 1,
        "result_content": result_content,
    }
    existing_results = list(state.get("audit_results", []))

    updated = False
    for i, r in enumerate(existing_results):
        if r.get("section_idx") == idx:
            existing_results[i] = new_result
            updated = True
            break
    if not updated:
        existing_results.append(new_result)

    return {
        "round_count": round_count + 1,
        "audit_results": existing_results,
    }


def update_context_node(state: AuditState) -> dict:
    """更新跨章节全局上下文 — 纯函数节点"""
    idx = state.get("current_section_idx", 0)
    sections = state.get("sections", [])
    if idx >= len(sections):
        return {"current_section_idx": idx + 1, "round_count": 0}

    section = sections[idx]
    section_title = section.get("title", "")

    key_findings = state.get("key_findings", [])
    context_entries = []
    for finding in key_findings[:3]:
        context_entries.append(f"[Section {idx}: {section_title}] {finding}")

    if not context_entries:
        messages = state.get("messages", [])
        for m in reversed(messages):
            content = m.content if hasattr(m, "content") else ""
            has_tool_calls = hasattr(m, "tool_calls") and m.tool_calls
            if content and not has_tool_calls and len(content) > 50:
                context_entries.append(f"[Section {idx}: {section_title}] {content[:300]}")
                break

    return {
        "global_context": context_entries if context_entries else [],
        "current_section_idx": idx + 1,
        "round_count": 0,
    }


def cross_validate_node(state: AuditState, llm: ChatOpenAI) -> dict:
    """交叉验证节点 — 检查不同小节的审核结果之间是否存在矛盾"""
    audit_results = state.get("audit_results", [])
    if len(audit_results) < 2:
        return {"contradictions": []}

    results_summary = []
    for r in audit_results:
        findings = r.get("findings", [])
        results_summary.append(
            f"### Section {r.get('section_idx')}: {r.get('title')}\n"
            f"Findings: {'; '.join(findings) if findings else 'No findings'}\n"
            f"Confidence: {r.get('confidence', '?')}"
        )

    results_text = "\n\n".join(results_summary)

    prompt = f"""You are a medical device documentation audit quality reviewer.
Check the following section audit results for contradictions or inconsistencies.

## Section Audit Results
{results_text}

## Task
1. Check if any two sections have contradictory findings
2. Check if findings are consistent across sections
3. Identify any cross-cutting issues that span multiple sections

## Output Format (JSON only)
```json
{{
  "contradictions": [
    {{
      "sections": [0, 3],
      "description": "Section 0 says X but Section 3 says Y about the same topic",
      "severity": "critical/major/minor",
      "recommendation": "..."
    }}
  ],
  "cross_cutting_issues": ["issue 1", "issue 2"],
  "overall_consistency_score": 8
}}
```

If no contradictions found, return empty arrays."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        result = _extract_json(content, {"contradictions": [], "cross_cutting_issues": [], "overall_consistency_score": 5})
    except Exception as e:
        logger.error(f"cross_validate 失败: {e}")
        result = {"contradictions": [], "cross_cutting_issues": [], "overall_consistency_score": 5}

    return {"contradictions": result.get("contradictions", [])}


def generate_report_node(state: AuditState, llm: ChatOpenAI) -> dict:
    """
    综合报告生成节点

    报告结构（与单纯 LLM 综述不同，本节点直接拼接逐小节明细，避免信息丢失）：

      # 审核报告
      ## 1. 文档结构分析       ← 来自 outline_summary.tree_text（纯文本拼接）
      ## 2. 总体概述           ← LLM 基于 audit_results 生成（仅高层综述）
      ## 3. 逐小节审核明细      ← 直接拼接每个 audit_results[i].result_content
      ## 4. 交叉验证发现        ← 来自 contradictions（纯文本拼接）
      ## 5. 审核结论与建议      ← LLM 基于全局发现生成（仅高层结论）

    LLM 调用次数: 1（一次性生成开头综述+结尾结论的合并 prompt）
    """
    audit_results = state.get("audit_results", [])
    contradictions = state.get("contradictions", [])
    document_type = state.get("document_type", "")
    outline_summary = state.get("outline_summary", {}) or {}

    if not audit_results:
        return {
            "final_report": "# 审核报告\n\n未能生成审核结果，请检查文档是否包含有效内容。",
            "finished": True,
        }

    # ----- 段落 1: 文档结构分析 -----
    structure_block = "## 1. 文档结构分析\n\n"
    if outline_summary.get("tree_text"):
        structure_block += outline_summary["tree_text"] + "\n"
    else:
        structure_block += (
            f"- **总计**：识别 {len(audit_results)} 个待审核小节\n"
        )

    # ----- 段落 3: 逐小节审核明细（直接拼接 result_content）-----
    sections_detail_lines = ["## 3. 逐小节审核明细", ""]
    for r in sorted(audit_results, key=lambda x: int(x.get("section_idx", 0))):
        idx = r.get("section_idx", 0)
        title = r.get("title", "Untitled")
        breadcrumb = r.get("breadcrumb", "") or title
        depth = r.get("depth", "standard")
        round_count = r.get("round_count", 1)
        content = r.get("result_content", "")

        sections_detail_lines.append(f"### 3.{idx + 1} {breadcrumb}")
        sections_detail_lines.append("")
        sections_detail_lines.append(
            f"> **审核深度**：{depth} ｜ **审核轮次**：{round_count}"
        )
        sections_detail_lines.append("")

        if content and content.strip():
            sections_detail_lines.append(content.strip())
        else:
            findings = r.get("findings", [])
            if findings:
                sections_detail_lines.append("**关键发现**：")
                for f in findings:
                    sections_detail_lines.append(f"- {f}")
            else:
                sections_detail_lines.append("_（本小节未产生审核详情）_")

        sections_detail_lines.append("")
        sections_detail_lines.append("---")
        sections_detail_lines.append("")

    sections_detail_block = "\n".join(sections_detail_lines)

    # ----- 段落 4: 交叉验证 -----
    contradictions_block = "## 4. 交叉验证发现\n\n"
    if contradictions:
        for i, c in enumerate(contradictions, 1):
            sec_ref = c.get("sections", [])
            contradictions_block += (
                f"{i}. **涉及小节 {sec_ref}**\n"
                f"   - 描述：{c.get('description', '')}\n"
                f"   - 建议：{c.get('recommendation', '')}\n\n"
            )
    else:
        contradictions_block += "_未检测到明显矛盾。_\n"

    # ----- 调用 LLM 生成"开头综述"和"结尾结论" -----
    findings_summary_lines = []
    for r in audit_results:
        title = r.get("title", "")
        findings = r.get("findings", [])
        if findings:
            findings_summary_lines.append(
                f"- [{r.get('section_idx')}: {title}] " + "; ".join(findings[:3])
            )

    findings_brief = "\n".join(findings_summary_lines[:30])

    overview_prompt = f"""你是一名资深医疗器械文档审核专家。请基于下面的逐小节关键发现摘要，生成两段独立内容：

## 文档信息
- 审核类型: {document_type}
- 已审核小节数: {len(audit_results)}
- 章节总数: {outline_summary.get('chapter_count', '?')}

## 各小节关键发现（摘要）
{findings_brief if findings_brief else '(无显式发现项)'}

## 交叉验证概况
{f"检测到 {len(contradictions)} 处跨小节矛盾" if contradictions else "未检测到跨小节矛盾"}

## 你的任务
请用中文输出严格遵循以下 Markdown 格式的结果（不要添加任何额外内容、不要包裹代码块）：

<<OVERVIEW>>
[此处写 3-5 句的整体概述：覆盖范围、整体合规情况、最值得关注的几个跨章节问题。不要列出每小节细节，不要给出评分或合规等级，那些在报告下半段已展示。]
<<END_OVERVIEW>>

<<CONCLUSION>>
[此处写审核结论与改进建议：
1. 优先级 Top-3 整改事项（每条一行，标注涉及小节编号）；
2. 后续建议的步骤（如复审、补充材料、外部专家介入等）。]
<<END_CONCLUSION>>"""

    try:
        response = llm.invoke([HumanMessage(content=overview_prompt)])
        llm_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"generate_report LLM 综述失败: {e}")
        llm_text = ""

    def _extract_block(text: str, start_tag: str, end_tag: str, fallback: str) -> str:
        if not text:
            return fallback
        m = re.search(
            re.escape(start_tag) + r"(.*?)" + re.escape(end_tag),
            text,
            re.DOTALL,
        )
        if m:
            return m.group(1).strip()
        return fallback

    overview_text = _extract_block(
        llm_text, "<<OVERVIEW>>", "<<END_OVERVIEW>>",
        fallback="（综述生成失败，请直接查阅下方逐小节审核明细。）",
    )
    conclusion_text = _extract_block(
        llm_text, "<<CONCLUSION>>", "<<END_CONCLUSION>>",
        fallback="（结论生成失败，请基于逐小节明细自行判断。）",
    )

    # ----- 拼装最终报告 -----
    final_report = "\n".join([
        f"# 审核报告（{document_type or '通用'}）",
        "",
        structure_block,
        "",
        "## 2. 总体概述",
        "",
        overview_text,
        "",
        sections_detail_block,
        contradictions_block,
        "## 5. 审核结论与建议",
        "",
        conclusion_text,
        "",
    ])

    return {
        "final_report": final_report,
        "finished": True,
    }


# ============== 路由函数 ==============

def route_after_eval(state: AuditState) -> str:
    """evaluate_result 之后的路由: 复审还是通过"""
    confidence = state.get("confidence", 3)
    round_count = state.get("round_count", 0)
    max_rounds = 2

    if confidence < 4 and round_count < max_rounds:
        return "reaudit"
    return "pass"


def route_next_section(state: AuditState) -> str:
    """update_context 之后的路由: 下一节还是汇总"""
    idx = state.get("current_section_idx", 0)
    total = len(state.get("sections", []))
    if idx < total:
        return "next"
    return "done"


# ============== 图构建 ==============

def build_audit_graph(llm: ChatOpenAI, retriever) -> StateGraph:
    """
    构建文档审核 StateGraph

    根据 PARALLEL_AUDIT_ENABLED 环境变量使用不同的图拓扑：

    并行模式 (PARALLEL_AUDIT_ENABLED=true，默认):
      plan_strategy → batch_audit_all_sections (内部 asyncio.gather 并发)
                   → cross_validate → generate_report → END

    顺序模式 (PARALLEL_AUDIT_ENABLED=false):
      plan_strategy → audit_section → evaluate_result(纯解析)
                    → (reaudit loop: max 2) → update_context
                    → next/done → cross_validate → generate_report → END
    """
    from functools import partial

    workflow = StateGraph(AuditState)

    plan_node = partial(plan_strategy_node, llm=llm, retriever=retriever)
    cross_val_node = partial(cross_validate_node, llm=llm)
    report_node = partial(generate_report_node, llm=llm)
    outline_node = partial(analyze_outline_node, llm=llm)

    workflow.add_node("analyze_outline", outline_node)
    workflow.add_node("plan_strategy", plan_node)
    workflow.add_node("cross_validate", cross_val_node)
    workflow.add_node("generate_report", report_node)

    workflow.set_entry_point("analyze_outline")
    workflow.add_edge("analyze_outline", "plan_strategy")

    if PARALLEL_AUDIT_ENABLED:
        # ===== 并行模式: batch_audit 节点内部并发审核所有章节 =====
        batch_node = partial(batch_audit_all_sections_node, llm=llm, retriever=retriever)
        workflow.add_node("batch_audit", batch_node)
        workflow.add_edge("plan_strategy", "batch_audit")
        workflow.add_edge("batch_audit", "cross_validate")
        logger.info(f"Agent Graph: 并行模式 (asyncio.gather, 并发={AUDIT_CONCURRENCY})")
    else:
        # ===== 顺序模式: 保留原有 audit_section → evaluate_result → update_context 循环 =====
        audit_node = partial(audit_section_node, llm=llm, retriever=retriever)
        workflow.add_node("audit_section", audit_node)
        workflow.add_node("evaluate_result", evaluate_result_node)
        workflow.add_node("update_context", update_context_node)

        workflow.add_edge("plan_strategy", "audit_section")
        workflow.add_edge("audit_section", "evaluate_result")

        workflow.add_conditional_edges(
            "evaluate_result",
            route_after_eval,
            {"reaudit": "audit_section", "pass": "update_context"},
        )

        workflow.add_conditional_edges(
            "update_context",
            route_next_section,
            {"next": "audit_section", "done": "cross_validate"},
        )
        logger.info("Agent Graph: 顺序模式（向后兼容）")

    workflow.add_edge("cross_validate", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow


def compile_graph(llm: Optional[ChatOpenAI] = None, retriever=None,
                  checkpointer: Optional[object] = None):
    """
    编译并返回可执行的 graph

    Args:
        llm: ChatOpenAI 实例，不传则自动创建
        retriever: RAGRetriever 实例（用于 plan_strategy 预检索）
        checkpointer: LangGraph checkpointer，不传则使用 MemorySaver

    Returns:
        CompiledGraph
    """
    if llm is None:
        llm = create_llm()
    if checkpointer is None:
        checkpointer = MemorySaver()

    workflow = build_audit_graph(llm, retriever)
    return workflow.compile(checkpointer=checkpointer)


def get_sqlite_checkpointer(db_path: str = "checkpoints/audit.db") -> object:
    """获取 SQLite checkpoint 实例 — 用于生产环境持久化（async 版本）"""
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

    try:
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)
    except ImportError:
        logger.warning("langgraph-checkpoint-sqlite 未安装，使用 MemorySaver")
        return MemorySaver()


def compile_graph_with_persistence(llm: Optional[ChatOpenAI] = None,
                                   retriever=None,
                                   db_path: str = "checkpoints/audit.db"):
    """编译带 SQLite 持久化的 graph"""
    checkpointer = get_sqlite_checkpointer(db_path)
    return compile_graph(llm=llm, retriever=retriever, checkpointer=checkpointer)


# ============== 对话模式 (Conversation Mode) ==============

AUDIT_BATCH_SIZE = int(os.getenv("AUDIT_BATCH_SIZE", "5"))
MAX_RE_AUDIT_CYCLES = int(os.getenv("MAX_RE_AUDIT_CYCLES", "3"))


def _classify_user_intent(feedback: str, llm: ChatOpenAI) -> dict:
    """LLM-based intent classification for user feedback at checkpoints"""
    if not feedback or not feedback.strip():
        return {"intent": "approve", "re_audit_sections": [], "skip_sections": [],
                "supplement_text": "", "standard_override": "", "confidence": 1.0}

    prompt = f"""Classify the user's feedback into one of these intents. The user is responding to an audit agent at a checkpoint.

## Intent Types
- **approve**: User wants to continue/proceed (e.g. "继续", "好的", "没问题", "approve", "ok", "yes")
- **adjust**: User wants to re-audit specific sections, possibly with a different standard (e.g. "用IEC 62304重审第3节", "re-audit section 3 and 5", "第2节重新审核")
- **supplement**: User provides additional context/info (e.g. "第5节的风险控制已在CR-2024-003中更新", "this section was already reviewed internally")
- **skip**: User wants to skip sections (e.g. "跳过第7节", "skip section 4", "ignore section 2")
- **question**: User asks a question (e.g. "为什么第2节评分这么低?", "what does this finding mean?")
- **regenerate**: User wants to regenerate/modify the report format (e.g. "报告太啰嗦", "make it shorter", "generate executive summary")

## User Feedback
{feedback[:500]}

## Output (JSON only)
```json
{{
  "intent": "approve|adjust|supplement|skip|question|regenerate",
  "re_audit_sections": [2, 5],
  "skip_sections": [7],
  "supplement_text": "additional context to inject",
  "standard_override": "IEC 62304",
  "question_text": "the user's question",
  "regenerate_instruction": "how to regenerate"
}}
```

For approve intent: re_audit_sections=[], skip_sections=[]
For question intent: extract the question into question_text
For regenerate: extract formatting instructions into regenerate_instruction"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        result = _extract_json(content, {"intent": "approve"})
        if not result.get("intent") or result["intent"] not in (
            "approve", "adjust", "supplement", "skip", "question", "regenerate"
        ):
            result["intent"] = "approve"
        return {
            "intent": result.get("intent", "approve"),
            "re_audit_sections": result.get("re_audit_sections", []),
            "skip_sections": result.get("skip_sections", []),
            "supplement_text": result.get("supplement_text", ""),
            "standard_override": result.get("standard_override", ""),
            "question_text": result.get("question_text", ""),
            "regenerate_instruction": result.get("regenerate_instruction", ""),
            "confidence": float(result.get("confidence", 0.8)),
        }
    except Exception as e:
        logger.warning(f"Intent classification failed, defaulting to approve: {e}")
        return {
            "intent": "approve", "re_audit_sections": [], "skip_sections": [],
            "supplement_text": "", "standard_override": "", "confidence": 0.5,
        }


def _create_checkpoint_node(checkpoint_name: str, stage_name: str):
    """
    工厂函数：创建 checkpoint 节点

    Checkpoint 节点使用 LangGraph interrupt() 暂停图执行，
    等待用户反馈后通过 Command(resume=...) 恢复。

    Args:
        checkpoint_name: 检查点标识 (e.g. "strategy_review", "post_audit")
        stage_name: 用户可读的阶段名称 (e.g. "审核策略确认", "审核结果确认")
    """

    async def checkpoint_node(state: AuditState, *, config: RunnableConfig) -> dict:
        sections = state.get("sections", [])
        audit_results = state.get("audit_results", [])
        audit_plan = state.get("audit_plan", {})

        summary_lines = []
        # Phase 2.4: 结构化数据，供前端渲染卡片（structured payload）
        structured: dict = {}
        if checkpoint_name == "strategy_review":
            summary_lines.append(f"文档解析完成，共 {len(sections)} 个章节")
            deep_count = sum(1 for v in audit_plan.values() if v.get("depth") == "deep")
            standard_count = sum(1 for v in audit_plan.values() if v.get("depth") == "standard")
            quick_count = sum(1 for v in audit_plan.values() if v.get("depth") == "quick")
            summary_lines.append(
                f"审核策略: deep={deep_count}, standard={standard_count}, quick={quick_count}"
            )
            structured["strategy"] = {
                "section_count": len(sections),
                "deep": deep_count,
                "standard": standard_count,
                "quick": quick_count,
                "plan_preview": [
                    {
                        "section_idx": idx,
                        "title": sections[idx].get("title", "") if idx < len(sections) else "",
                        "depth": (audit_plan.get(idx) or {}).get("depth", "standard"),
                        "focus_standards": (audit_plan.get(idx) or {}).get("focus_standards", []),
                    }
                    for idx in list(audit_plan.keys())[:20]
                ],
            }
        elif checkpoint_name == "post_audit":
            total_findings = sum(len(r.get("findings", [])) for r in audit_results)
            critical = sum(
                1 for r in audit_results
                for f in r.get("findings", [])
                if "critical" in str(f).lower() or "严重" in str(f)
            )
            summary_lines.append(f"已审核 {len(audit_results)}/{len(sections)} 个章节")
            summary_lines.append(f"发现 {total_findings} 个问题（含 {critical} 个严重问题）")
            if state.get("contradictions"):
                summary_lines.append(f"交叉验证发现 {len(state['contradictions'])} 处矛盾")
            structured["overview"] = {
                "audited": len(audit_results),
                "total": len(sections),
                "total_findings": total_findings,
                "critical": critical,
                "contradictions_count": len(state.get("contradictions") or []),
                "sections": [
                    {
                        "section_idx": r.get("section_idx"),
                        "title": r.get("title", ""),
                        "depth": r.get("depth", ""),
                        "score": r.get("score"),
                        "confidence": r.get("confidence"),
                        "findings_count": len(r.get("findings", [])),
                        "critical_count": sum(
                            1 for f in r.get("findings", [])
                            if "critical" in str(f).lower() or "严重" in str(f)
                        ),
                    }
                    for r in audit_results[:50]
                ],
            }
        elif checkpoint_name == "post_completion":
            # 报告已生成；进入"完成后自由对话"循环。
            # 用户可继续提问（intent=question/chat），或回复"结束/exit/完成"退出。
            summary_lines.append("审核报告已生成。")
            summary_lines.append("您可以继续提问或要求修改报告内容；回复 \"结束\" / \"exit\" 即可关闭会话。")
            structured["completion"] = {
                "section_count": len(sections),
                "audited_count": len(audit_results),
                "has_report": bool(state.get("final_report")),
            }

        summary = "\n".join(summary_lines) if summary_lines else f"Stage: {stage_name}"

        # Python 3.10 的 asyncio 不支持 context 参数传递 contextvars，
        # 需要手动设置 var_child_runnable_config 以便 interrupt() → get_config() 能读取到配置。
        import time as _time
        _token = var_child_runnable_config.set(config)
        try:
            feedback = interrupt({
                "type": "checkpoint",
                "checkpoint": checkpoint_name,
                "stage": stage_name,
                "summary": summary,
                "section_count": len(sections),
                "audited_count": len(audit_results),
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


def _summarize_for_checkpoint(audit_results: list, sections: list) -> str:
    """生成 checkpoint 摘要信息"""
    total_findings = sum(len(r.get("findings", [])) for r in audit_results)
    parts = []
    parts.append(f"已完成 {len(audit_results)}/{len(sections)} 个章节的审核")
    parts.append(f"共发现 {total_findings} 个问题项")
    return "\n".join(parts)


async def _re_audit_sections_node(state: AuditState, llm: ChatOpenAI, retriever) -> dict:
    """重新审核指定章节 — 针对用户请求的章节进行单章重审"""
    sections = state.get("sections", [])
    re_audit_indices = state.get("re_audit_sections", [])
    conversation_context = state.get("conversation_context", "")
    knowledge_context = state.get("knowledge_context", "")
    doc_type = state.get("document_type", "")
    audit_plan = state.get("audit_plan", {})
    existing_results = list(state.get("audit_results", []))
    revision_requests = list(state.get("revision_requests", []))
    re_audit_cycle_count = state.get("re_audit_cycle_count", 0)

    if not re_audit_indices:
        return {"re_audit_cycle_count": re_audit_cycle_count + 1}

    # Filter to valid indices
    valid_indices = [i for i in re_audit_indices if 0 <= i < len(sections)]
    if not valid_indices:
        return {"re_audit_cycle_count": re_audit_cycle_count + 1}

    logger.info(f"re_audit: 重新审核章节 {valid_indices}, cycle={re_audit_cycle_count + 1}")

    semaphore = asyncio.Semaphore(AUDIT_CONCURRENCY)

    async def re_audit_one(idx: int) -> dict:
        async with semaphore:
            section = sections[idx]
            depth_config = audit_plan.get(idx, {})
            depth = depth_config.get("depth", "standard")
            focus_standards = list(depth_config.get("focus_standards", []))

            # Override standard if user specified one
            for req in revision_requests:
                req_indices = req.get("section_idx", [])
                if isinstance(req_indices, int):
                    req_indices = [req_indices]
                if idx in req_indices and req.get("standard_override"):
                    focus_standards = [req["standard_override"]]

            section_knowledge = _retrieve_section_knowledge(section, doc_type, retriever)

            section_llm = create_llm()

            system_prompt = _build_audit_system_prompt(
                section=section, depth=depth, focus_standards=focus_standards,
                doc_type=doc_type, knowledge_context=knowledge_context,
                conversation_context=conversation_context,
                round_count=re_audit_cycle_count,
                section_knowledge=section_knowledge,
            )

            content_text = section.get("content", "")
            user_prompt = (
                f"## Re-Audit Section (User Requested)\n"
                f"**Title**: {section.get('title', 'Untitled')}\n"
                f"**Breadcrumb**: {section.get('breadcrumb', '')}\n"
                f"**Content**:\n{content_text[:8000]}\n\n"
                f"Re-audit this section with extra scrutiny. The user has specifically "
                f"requested this section be re-examined."
            )

            try:
                response = await section_llm.ainvoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ])
                result_content = response.content if hasattr(response, "content") else str(response)
            except Exception as e:
                logger.error(f"re_audit LLM 失败 (section {idx}): {e}")
                result_content = f"[LLM Error: {e}]"

            metadata = _extract_json(result_content, {"confidence": 3, "key_findings": []})
            return {
                "section_idx": idx, "title": section.get("title", ""),
                "breadcrumb": section.get("breadcrumb", ""),
                "depth": depth, "findings": metadata.get("key_findings", []),
                "confidence": int(metadata.get("confidence", 3)),
                "round_count": re_audit_cycle_count + 1,
                "result_content": result_content,
            }

    try:
        new_results = await asyncio.gather(*[re_audit_one(i) for i in valid_indices])
    except Exception as e:
        logger.error(f"re_audit 并发执行异常: {e}")
        new_results = []
        for i in valid_indices:
            try:
                r = await re_audit_one(i)
                new_results.append(r)
            except Exception as e2:
                logger.error(f"re_audit section {i} fallback 失败: {e2}")

    # Merge results: replace existing results for re-audited sections
    new_by_idx = {r["section_idx"]: r for r in new_results}
    merged = []
    for r in existing_results:
        idx = r.get("section_idx")
        if idx in new_by_idx:
            merged.append(new_by_idx[idx])
        else:
            merged.append(r)
    # Add any new sections that didn't exist before
    for idx, r in new_by_idx.items():
        if not any(m.get("section_idx") == idx for m in merged):
            merged.append(r)
    merged.sort(key=lambda r: int(r.get("section_idx", 0)))

    global_context = []
    for r in merged:
        title = r.get("title", "")
        for finding in r.get("findings", [])[:2]:
            global_context.append(f"[{r['section_idx']}: {title}] {finding}")

    logger.info(f"re_audit: 完成 {len(valid_indices)} 个章节重审, 合并后共 {len(merged)} 个结果")

    return {
        "audit_results": merged,
        "global_context": global_context,
        "re_audit_cycle_count": re_audit_cycle_count + 1,
        "re_audit_sections": [],
        "revision_requests": [],
    }


# ============== Phase 3.1: Tool-calling Agent 审核单章节 ==============

# 缓存 create_agent 实例，避免每次重新构建
_section_audit_agent_cache = None


def _get_section_audit_agent(llm: ChatOpenAI, retriever):
    """惰性构建并缓存一个使用 agent_tools.py 工具的审核 agent。

    使用 langchain.agents.create_agent —— LLM 自主决定调用 rag_search /
    check_regulation / assess_completeness 等工具，最后输出 JSON。
    """
    global _section_audit_agent_cache
    if _section_audit_agent_cache is not None:
        return _section_audit_agent_cache

    try:
        from langchain.agents import create_agent
    except ImportError:
        try:
            from langgraph.prebuilt import create_react_agent as create_agent  # type: ignore
        except ImportError:
            logger.warning("create_agent / create_react_agent 不可用，退回直接 LLM 调用")
            return None

    try:
        from agent_tools import create_audit_tools
    except ImportError as e:
        logger.warning(f"agent_tools 导入失败: {e}")
        return None

    tools = create_audit_tools(retriever, llm)
    try:
        _section_audit_agent_cache = create_agent(model=llm, tools=tools)
    except TypeError:
        # 旧版本 API：create_react_agent(llm, tools)
        try:
            _section_audit_agent_cache = create_agent(llm, tools)
        except Exception as e:
            logger.warning(f"create_agent 构建失败: {e}")
            return None
    return _section_audit_agent_cache


async def _audit_section_with_agent(
    section: dict, depth: str, focus_standards: list, doc_type: str,
    knowledge_context: str, section_knowledge: str,
    llm: ChatOpenAI, retriever,
) -> str:
    """通过 tool-calling agent 审核单个章节，返回原始 LLM 文本。

    若 agent 不可用或失败，返回空字符串以让上层退回到默认直接调用 LLM 路径。
    """
    agent = _get_section_audit_agent(llm, retriever)
    if agent is None:
        return ""

    system_prompt = _build_audit_system_prompt(
        section=section, depth=depth, focus_standards=focus_standards,
        doc_type=doc_type, knowledge_context=knowledge_context,
        section_knowledge=section_knowledge,
    )
    user_prompt = (
        f"## Current Section\n"
        f"**Title**: {section.get('title', 'Untitled')}\n"
        f"**Breadcrumb**: {section.get('breadcrumb', '')}\n"
        f"**Content**:\n{section.get('content', '')[:8000]}\n\n"
        f"请按需调用工具收集证据，最终以 JSON 形式输出 confidence + key_findings。"
    )
    try:
        result = await agent.ainvoke({
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        })
        msgs = result.get("messages", []) if isinstance(result, dict) else []
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and getattr(m, "content", ""):
                content = m.content
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content)
                return str(content)
    except Exception as e:
        logger.warning(f"tool-calling agent 审核失败，退回直接调用: {e}")
    return ""


# ============== Phase 3.3: Send API 子图并行审核 ==============

_send_api_subgraph_cache = None


def _get_send_api_audit_subgraph(llm: ChatOpenAI, retriever):
    """构建并缓存一个用 Send API fan-out 的子图。

    子图状态结构（使用 reducer 自动汇集 results）:
      - sections_packets: list[dict]       (输入：每个 dict 描述一个待审章节)
      - results: Annotated[list, operator.add]  (输出：单章节结果汇总)
      - shared: dict                        (上下文：doc_type / knowledge_context)
    """
    global _send_api_subgraph_cache
    if _send_api_subgraph_cache is not None:
        return _send_api_subgraph_cache

    try:
        from langgraph.types import Send  # type: ignore
        from typing import TypedDict as _TD, Annotated as _Ann
        import operator as _op
    except ImportError as e:
        logger.warning(f"Send API 依赖不可用: {e}")
        return None

    class _SubState(_TD):
        sections_packets: list
        shared: dict
        results: _Ann[list, _op.add]

    async def _audit_one(state: dict) -> dict:
        """单章节审核 worker（Send 调用时只看到自己那份 packet）"""
        packet = state.get("packet", {})
        shared = state.get("shared", {})
        idx = packet["section_idx"]
        section = packet["section"]
        depth = packet.get("depth", "standard")
        focus_standards = packet.get("focus_standards", [])
        section_knowledge = packet.get("section_knowledge", "")

        section_llm = create_llm()
        system_prompt = _build_audit_system_prompt(
            section=section, depth=depth, focus_standards=focus_standards,
            doc_type=shared.get("doc_type", ""),
            knowledge_context=shared.get("knowledge_context", ""),
            section_knowledge=section_knowledge,
        )
        user_prompt = (
            f"## Current Section\n"
            f"**Title**: {section.get('title', 'Untitled')}\n"
            f"**Breadcrumb**: {section.get('breadcrumb', '')}\n"
            f"**Content**:\n{section.get('content', '')[:8000]}"
        )
        try:
            response = await section_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            result_content = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"Send-API audit_one 失败 (section {idx}): {e}")
            result_content = f"[LLM Error: {e}]"

        metadata = _extract_json(result_content, {"confidence": 3, "key_findings": []})
        return {
            "results": [{
                "section_idx": idx, "title": section.get("title", ""),
                "breadcrumb": section.get("breadcrumb", ""),
                "depth": depth, "findings": metadata.get("key_findings", []),
                "confidence": int(metadata.get("confidence", 3)),
                "round_count": 1, "result_content": result_content,
            }]
        }

    def _dispatcher(state: _SubState):
        """fan-out: 为每个 packet 发送一个 Send 到 audit_one"""
        packets = state.get("sections_packets", []) or []
        shared = state.get("shared", {})
        return [Send("audit_one", {"packet": p, "shared": shared}) for p in packets]

    sub = StateGraph(_SubState)
    sub.add_node("audit_one", _audit_one)
    sub.add_conditional_edges("__start__", _dispatcher, ["audit_one"])
    sub.add_edge("audit_one", END)
    try:
        _send_api_subgraph_cache = sub.compile()
    except Exception as e:
        logger.warning(f"Send API 子图编译失败: {e}")
        return None
    return _send_api_subgraph_cache


async def _run_send_api_audit_batch(
    section_indices: list, sections: list, audit_plan: dict, doc_type: str,
    knowledge_context: str, llm: ChatOpenAI, retriever,
) -> list:
    """通过 Send API 子图并行审核一批章节，返回结果列表。失败返回空列表。"""
    subgraph = _get_send_api_audit_subgraph(llm, retriever)
    if subgraph is None:
        return []

    packets = []
    for idx in section_indices:
        section = sections[idx]
        depth_config = audit_plan.get(idx, {})
        packets.append({
            "section_idx": idx,
            "section": section,
            "depth": depth_config.get("depth", "standard"),
            "focus_standards": depth_config.get("focus_standards", []),
            "section_knowledge": _retrieve_section_knowledge(section, doc_type, retriever),
        })

    try:
        out = await subgraph.ainvoke({
            "sections_packets": packets,
            "shared": {"doc_type": doc_type, "knowledge_context": knowledge_context},
            "results": [],
        })
        results = out.get("results", []) if isinstance(out, dict) else []
        results.sort(key=lambda r: int(r.get("section_idx", 0)))
        return results
    except Exception as e:
        logger.warning(f"Send API 子图执行失败，退回 asyncio.gather: {e}")
        return []


async def _batched_audit_with_interrupt_node(state: AuditState, llm: ChatOpenAI, retriever, *, config: RunnableConfig) -> dict:
    """批量审核节点 — 每 N 章中断一次，允许用户查看进度"""
    sections = state.get("sections", [])
    audit_plan = state.get("audit_plan", {})
    knowledge_context = state.get("knowledge_context", "")
    doc_type = state.get("document_type", "")
    total_steps = state.get("total_agent_steps", 0)
    skip_sections = set(state.get("skip_sections", []))

    if not sections:
        return {
            "audit_results": [],
            "global_context": [],
            "messages": [AIMessage(content="[No sections to audit]")],
        }

    semaphore = asyncio.Semaphore(AUDIT_CONCURRENCY)
    all_results = []

    async def audit_one_section(idx: int) -> dict:
        async with semaphore:
            section = sections[idx]
            depth_config = audit_plan.get(idx, {})
            depth = depth_config.get("depth", "standard")
            focus_standards = depth_config.get("focus_standards", [])

            section_knowledge = _retrieve_section_knowledge(section, doc_type, retriever)

            section_llm = create_llm()
            system_prompt = _build_audit_system_prompt(
                section=section, depth=depth, focus_standards=focus_standards,
                doc_type=doc_type, knowledge_context=knowledge_context,
                section_knowledge=section_knowledge,
            )

            content_text = section.get("content", "")
            user_prompt = (
                f"## Current Section\n"
                f"**Title**: {section.get('title', 'Untitled')}\n"
                f"**Breadcrumb**: {section.get('breadcrumb', '')}\n"
                f"**Content**:\n{content_text[:8000]}"
            )

            result_content = ""
            # Phase 3.1: 优先尝试 tool-calling agent；失败则退回直接 LLM 调用
            if USE_TOOL_CALLING_AUDIT:
                result_content = await _audit_section_with_agent(
                    section=section, depth=depth, focus_standards=focus_standards,
                    doc_type=doc_type, knowledge_context=knowledge_context,
                    section_knowledge=section_knowledge, llm=section_llm, retriever=retriever,
                )

            if not result_content:
                try:
                    response = await section_llm.ainvoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ])
                    result_content = response.content if hasattr(response, "content") else str(response)
                except Exception as e:
                    logger.error(f"batch_audit LLM 失败 (section {idx}): {e}")
                    result_content = f"[LLM Error: {e}]"

            metadata = _extract_json(result_content, {"confidence": 3, "key_findings": []})
            return {
                "section_idx": idx, "title": section.get("title", ""),
                "breadcrumb": section.get("breadcrumb", ""),
                "depth": depth, "findings": metadata.get("key_findings", []),
                "confidence": int(metadata.get("confidence", 3)),
                "round_count": 1, "result_content": result_content,
            }

    all_indices = [i for i in range(len(sections)) if i not in skip_sections]
    batch_size = AUDIT_BATCH_SIZE
    logger.info(f"batched_audit: {len(all_indices)} 章节, batch_size={batch_size}, 跳过 {len(skip_sections)} 节")

    for batch_start in range(0, len(all_indices), batch_size):
        batch_indices = all_indices[batch_start:batch_start + batch_size]

        batch_results = []
        # Phase 3.3: 优先尝试通过 LangGraph Send API fan-out 并行审核
        if USE_SEND_API_AUDIT:
            send_results = await _run_send_api_audit_batch(
                section_indices=batch_indices, sections=sections,
                audit_plan=audit_plan, doc_type=doc_type,
                knowledge_context=knowledge_context, llm=llm, retriever=retriever,
            )
            if send_results:
                batch_results = send_results
                all_results.extend(batch_results)

        if not batch_results:
            try:
                batch_results = await asyncio.gather(*[audit_one_section(i) for i in batch_indices])
                all_results.extend(batch_results)
            except Exception as e:
                logger.error(f"batch {batch_start // batch_size} 并发异常: {e}")
                for i in batch_indices:
                    try:
                        r = await audit_one_section(i)
                        all_results.append(r)
                    except Exception as e2:
                        logger.error(f"section {i} fallback 失败: {e2}")
                        all_results.append({
                            "section_idx": i,
                            "title": sections[i].get("title", ""),
                            "breadcrumb": sections[i].get("breadcrumb", ""),
                            "depth": "standard", "findings": [],
                            "confidence": 1, "round_count": 1,
                            "result_content": f"[Error: {e2}]",
                        })

        # Allow user interaction between batches
        completed = batch_start + len(batch_indices)
        if completed < len(all_indices):
            summary = _summarize_for_checkpoint(all_results, sections)
            _token = var_child_runnable_config.set(config)
            try:
                interrupt({
                    "type": "checkpoint",
                    "checkpoint": "progress",
                    "stage": "audit_progress",
                    "completed": completed,
                    "total": len(all_indices),
                    "summary": summary,
                })
            finally:
                var_child_runnable_config.reset(_token)

    all_results.sort(key=lambda r: int(r.get("section_idx", 0)))

    global_context = []
    for r in all_results:
        title = r.get("title", "")
        for finding in r.get("findings", [])[:2]:
            global_context.append(f"[{r['section_idx']}: {title}] {finding}")

    logger.info(f"batched_audit: 完成 {len(all_results)} 章节审核")

    return {
        "audit_results": all_results,
        "global_context": global_context,
        "total_agent_steps": total_steps + len(all_indices),
        "current_stage": "audit",
        "messages": [AIMessage(content=f"[已完成 {len(all_results)} 个章节的审核]")],
    }


def _route_after_post_audit(state: AuditState) -> str:
    """post_audit checkpoint 后的路由: 根据 intent 决定下一步"""
    feedback = state.get("user_feedback", "").strip().lower()
    re_audit_sections = state.get("re_audit_sections", [])
    re_audit_cycle_count = state.get("re_audit_cycle_count", 0)

    # Simple keyword-based routing as initial filter
    re_audit_keywords = ["重审", "重新审核", "re-audit", "reaudit", "再审", "重新审"]
    skip_keywords = ["跳过", "skip", "忽略", "ignore"]
    approve_keywords = ["继续", "好的", "没问题", "ok", "yes", "approve", "continue", "可以", "通过", "同意", "确认"]

    if re_audit_sections and re_audit_cycle_count < MAX_RE_AUDIT_CYCLES:
        return "re_audit"
    if any(kw in feedback for kw in re_audit_keywords) and re_audit_cycle_count < MAX_RE_AUDIT_CYCLES:
        return "re_audit"
    return "generate_report"


# ----- Phase 1.4: 报告生成后的"自由对话"环 -----

_END_KEYWORDS = ("结束", "退出", "完成", "exit", "quit", "done", "bye", "再见")


def _route_after_completion(state: AuditState) -> str:
    """post_completion checkpoint 之后的路由：用户想退出还是继续聊天。"""
    feedback = (state.get("user_feedback", "") or "").strip().lower()
    if not feedback:
        return "end"
    if any(kw in feedback for kw in _END_KEYWORDS):
        return "end"
    return "chat"


async def _chat_response_node(state: AuditState, llm: ChatOpenAI) -> dict:
    """
    审核完成后的自由问答节点。

    - 输入：state.user_feedback（用户最新提问）
    - 上下文：final_report（截断到 2000 字以控制 token） + summary + 最近 6 条 messages
    - 输出：AIMessage 追加到 messages；当前阶段标记为 "post_completion"
    """
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
            SystemMessage(content=system_prompt),
            *history[-6:],
            HumanMessage(content=question),
        ])
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"chat_response 失败: {e}")
        answer = f"抱歉，处理您的问题时出错：{e}"

    return {
        "current_stage": "post_completion",
        "messages": [HumanMessage(content=question), AIMessage(content=answer)],
    }


# ============== Phase 2: Conversation Memory Summarization ==============

# 当 messages 超过该阈值时触发摘要节点；摘要后保留最近 6 条原始消息
_SUMMARIZE_TRIGGER_LEN = 16
_SUMMARIZE_KEEP_RECENT = 6


def _route_before_chat(state: AuditState) -> str:
    """根据 messages 长度判断是否需要先摘要再回答用户问题。"""
    history = state.get("messages", []) or []
    if len(history) >= _SUMMARIZE_TRIGGER_LEN:
        return "summarize"
    return "chat"


async def _summarize_conversation_node(state: AuditState, llm: ChatOpenAI) -> dict:
    """
    把较早的对话消息压缩为一段中文摘要，并通过 RemoveMessage 修剪历史。

    触发条件：len(messages) >= _SUMMARIZE_TRIGGER_LEN（默认 16）。
    保留：最近 _SUMMARIZE_KEEP_RECENT（默认 6）条原始消息。
    其余消息：合并到 state.summary 字段，再用 RemoveMessage 从 messages 中移除。
    """
    try:
        from langchain_core.messages import RemoveMessage
    except ImportError:
        # 老版本 langchain_core 没有 RemoveMessage —— 退化为只更新 summary
        RemoveMessage = None  # type: ignore

    history = state.get("messages", []) or []
    if len(history) < _SUMMARIZE_TRIGGER_LEN:
        return {}

    prev_summary = (state.get("summary", "") or "").strip()
    to_summarize = history[:-_SUMMARIZE_KEEP_RECENT]

    # 把 messages 转成纯文本输入
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
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        new_summary = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.warning(f"summarize_conversation 失败，跳过本次摘要: {e}")
        return {}

    update: dict = {"summary": new_summary.strip()}

    # 用 RemoveMessage 修剪 messages，只保留最近 _SUMMARIZE_KEEP_RECENT 条
    if RemoveMessage is not None:
        remove_ops = []
        for m in to_summarize:
            mid = getattr(m, "id", None)
            if mid:
                remove_ops.append(RemoveMessage(id=mid))
        if remove_ops:
            update["messages"] = remove_ops

    return update


# ============== Async wrappers for sync nodes (conversation graph) ==============

def _ensure_event_loop():
    """Ensure the current thread has an event loop (required on Windows when
    running inside asyncio.to_thread, since langchain may call get_event_loop)."""
    import asyncio as _asyncio
    try:
        _asyncio.get_event_loop()
    except RuntimeError:
        _asyncio.set_event_loop(_asyncio.new_event_loop())


async def _plan_strategy_async(state: AuditState, llm: ChatOpenAI, retriever) -> dict:
    """plan_strategy_node 的 async 包装 — 通过线程池执行同步调用，避免阻塞事件循环"""

    def _run():
        _ensure_event_loop()
        return plan_strategy_node(state, llm, retriever)

    return await asyncio.to_thread(_run)


async def _analyze_outline_async(state: AuditState, llm: ChatOpenAI) -> dict:
    """analyze_outline_node 的 async 包装（LLM 模式下不阻塞事件循环）"""

    def _run():
        _ensure_event_loop()
        return analyze_outline_node(state, llm)

    return await asyncio.to_thread(_run)


async def _cross_validate_async(state: AuditState, llm: ChatOpenAI) -> dict:
    """cross_validate_node 的 async 包装"""

    def _run():
        _ensure_event_loop()
        return cross_validate_node(state, llm)

    return await asyncio.to_thread(_run)


async def _generate_report_async(state: AuditState, llm: ChatOpenAI) -> dict:
    """generate_report_node 的 async 包装"""

    def _run():
        _ensure_event_loop()
        return generate_report_node(state, llm)

    return await asyncio.to_thread(_run)


async def _classify_intent_async(feedback: str, llm: ChatOpenAI) -> dict:
    """_classify_user_intent 的 async 包装"""

    def _run():
        _ensure_event_loop()
        return _classify_user_intent(feedback, llm)

    return await asyncio.to_thread(_run)


def build_conversation_graph(llm: ChatOpenAI, retriever) -> StateGraph:
    """
    构建对话式审核 StateGraph

    Topology:
      plan_strategy → checkpoint:strategy_review (interrupt)
                   → batched_audit (每 N 章 interrupt 报告进度)
                   → cross_validate
                   → checkpoint:post_audit (interrupt, 用户可请求重审/跳过/继续)
                   → [if re_audit] → re_audit_sections → cross_validate → back to post_audit
                   → [if approve] → generate_report → END
    """
    from functools import partial

    workflow = StateGraph(AuditState)

    # Nodes — use async wrappers for sync operations
    plan_node = partial(_plan_strategy_async, llm=llm, retriever=retriever)
    batch_audit_node = partial(_batched_audit_with_interrupt_node, llm=llm, retriever=retriever)
    re_audit_node = partial(_re_audit_sections_node, llm=llm, retriever=retriever)
    cross_val_node = partial(_cross_validate_async, llm=llm)
    report_node = partial(_generate_report_async, llm=llm)
    outline_node = partial(_analyze_outline_async, llm=llm)
    chat_node = partial(_chat_response_node, llm=llm)
    summarize_node = partial(_summarize_conversation_node, llm=llm)

    # Checkpoint nodes (use interrupt to pause for user input)
    strategy_checkpoint = _create_checkpoint_node("strategy_review", "审核策略确认")
    post_audit_checkpoint = _create_checkpoint_node("post_audit", "审核结果确认")
    post_completion_checkpoint = _create_checkpoint_node("post_completion", "报告生成完成")

    # Add all nodes
    workflow.add_node("analyze_outline", outline_node)
    workflow.add_node("plan_strategy", plan_node)
    workflow.add_node("checkpoint_strategy", strategy_checkpoint)
    workflow.add_node("batched_audit", batch_audit_node)
    workflow.add_node("cross_validate", cross_val_node)
    workflow.add_node("checkpoint_post_audit", post_audit_checkpoint)
    workflow.add_node("re_audit_sections", re_audit_node)
    workflow.add_node("generate_report", report_node)
    workflow.add_node("checkpoint_post_completion", post_completion_checkpoint)
    workflow.add_node("chat_response", chat_node)
    workflow.add_node("summarize_conversation", summarize_node)

    workflow.set_entry_point("analyze_outline")

    # Edges
    workflow.add_edge("analyze_outline", "plan_strategy")
    workflow.add_edge("plan_strategy", "checkpoint_strategy")
    workflow.add_edge("checkpoint_strategy", "batched_audit")
    workflow.add_edge("batched_audit", "cross_validate")
    workflow.add_edge("cross_validate", "checkpoint_post_audit")

    # Conditional routing after post_audit checkpoint
    workflow.add_conditional_edges(
        "checkpoint_post_audit",
        _route_after_post_audit,
        {
            "re_audit": "re_audit_sections",
            "generate_report": "generate_report",
        },
    )
    workflow.add_edge("re_audit_sections", "cross_validate")
    # 报告生成后进入"完成后自由对话"环：
    #   generate_report → post_completion (interrupt) → [chat | summarize→chat | END]
    #                     chat_response → post_completion (循环回去)
    # 当 messages 长度超过 _SUMMARIZE_TRIGGER_LEN 时，先经过 summarize_conversation
    # 节点把早期消息压缩到 state.summary，再进入 chat_response 节点。
    def _route_post_completion(state: AuditState) -> str:
        decision = _route_after_completion(state)
        if decision == "end":
            return "end"
        # 用户要继续对话：判断是否需要先摘要
        return _route_before_chat(state)

    workflow.add_edge("generate_report", "checkpoint_post_completion")
    workflow.add_conditional_edges(
        "checkpoint_post_completion",
        _route_post_completion,
        {
            "summarize": "summarize_conversation",
            "chat": "chat_response",
            "end": END,
        },
    )
    workflow.add_edge("summarize_conversation", "chat_response")
    workflow.add_edge("chat_response", "checkpoint_post_completion")

    logger.info("Conversation Graph: 对话模式 "
                f"(batch_size={AUDIT_BATCH_SIZE}, max_re_audit={MAX_RE_AUDIT_CYCLES})")

    return workflow


def compile_conversation_graph(
    llm: Optional[ChatOpenAI] = None,
    retriever=None,
    checkpointer: Optional[object] = None,
) -> object:
    """
    编译对话式审核 graph

    Args:
        llm: ChatOpenAI 实例
        retriever: RAGRetriever 实例
        checkpointer: 必须提供 checkpointer（推荐 SqliteSaver）用于 interrupt 持久化

    Returns:
        CompiledGraph (带 checkpointer)
    """
    if llm is None:
        llm = create_llm()
    if checkpointer is None:
        logger.warning("对话模式未提供 checkpointer，使用 MemorySaver（服务重启后状态丢失）")
        checkpointer = MemorySaver()

    workflow = build_conversation_graph(llm, retriever)
    return workflow.compile(checkpointer=checkpointer)
