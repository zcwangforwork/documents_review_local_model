"""
Agent 工具集 — LangChain @tool 封装
将现有 rag_retriever / doc_processor 的原子能力暴露为 Agent 工具
"""
import json
import logging
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Module-level globals set by create_audit_tools factory
_retriever = None
_llm_client = None
_DOC_TYPE_LABELS: Dict[str, str] = {}


def _get_doc_type_label(doc_type: str) -> str:
    if not doc_type:
        return ""
    if _DOC_TYPE_LABELS:
        return _DOC_TYPE_LABELS.get(doc_type, doc_type)
    if _retriever and hasattr(_retriever, "_DOC_TYPE_LABELS"):
        return _retriever._DOC_TYPE_LABELS.get(doc_type, doc_type)
    return doc_type


def _format_search_results(docs: List[Dict[str, Any]]) -> str:
    """将检索结果格式化为 LLM 可读的文本"""
    if not docs:
        return "[未在知识库中找到直接相关的参考内容]"
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("source", "unknown")
        text = doc.get("text", "")[:1500]
        parts.append(f"--- 参考 {i} (来源: {source}) ---\n{text}")
    return "\n\n".join(parts)


# ============ 工具1: 文档大纲解析 ============

@tool
def parse_document(text: str) -> str:
    """解析文档结构，返回树状大纲和审核单元列表。
    这是 Agent 审核任何文档的第一步必调用工具。

    Args:
        text: 文档全文文本

    Returns:
        JSON 字符串，包含 outline 和 sections
    """
    from doc_processor import parse_document_outline, flatten_to_audit_units

    outline = parse_document_outline(text)
    sections = flatten_to_audit_units(outline)

    serializable_sections = []
    for sec in sections:
        title, content, level, breadcrumb = sec
        serializable_sections.append({
            "title": title,
            "content_preview": content[:300] + "..." if len(content) > 300 else content,
            "level": level,
            "breadcrumb": breadcrumb,
            "content_length": len(content),
        })

    result = {
        "outline": outline,
        "sections": serializable_sections,
        "section_count": len(sections),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============ 工具2: RAG 检索 ============

@tool
def rag_search(query: str, top_k: int = 5) -> str:
    """从医疗器械法规知识库中检索相关条款和模板。
    在每个小节审核前调用，获取相关法规要求作为审核依据。
    建议 query 格式: "ISO 14971 风险控制措施要求"

    Args:
        query: 检索查询文本
        top_k: 返回文档数量，默认5

    Returns:
        检索到的相关文档片段文本
    """
    if _retriever is None:
        return "[ERROR] 知识库检索器未初始化"

    try:
        docs = _retriever.search_sync(query, top_k=top_k)
        return _format_search_results(docs)
    except Exception as e:
        logger.error(f"rag_search 失败: {e}")
        return f"[检索出错: {str(e)}]"


# ============ 工具3: 法规条款精确查询 ============

@tool
def check_regulation(standard: str, clause: str) -> str:
    """精确查询指定标准的具体条款原文。
    当审核中发现某条款引用不明确时使用。

    Args:
        standard: 标准编号，如 'ISO 14971:2019', 'IEC 62304', 'ISO 13485:2016'
        clause: 条款号，如 '§5.7', 'Section 7.3.2'

    Returns:
        检索到的条款原文或相关文档片段
    """
    if _retriever is None:
        return "[ERROR] 知识库检索器未初始化"

    query = f"{standard} {clause} 条款原文 要求"
    try:
        docs = _retriever.search_sync(query, top_k=5)
        return _format_search_results(docs)
    except Exception as e:
        logger.error(f"check_regulation 失败: {e}")
        return f"[查询出错: {str(e)}]"


# ============ 工具4: 完整性快速评估 ============

@tool
def assess_completeness(section_text: str, doc_type: str) -> str:
    """快速评估一个章节的完整度，返回1-5分评分和缺口摘要。
    在制定审核策略前调用，决定审核深度。

    Args:
        section_text: 章节文本内容
        doc_type: 文档类型 (risk_management/design_dev/software_compliance/
                  registration/production_quality/system_construction)

    Returns:
        JSON 字符串，包含 score(1-5), gaps, assessment
    """
    if _llm_client is None:
        return json.dumps({"score": 3, "gaps": [], "assessment": "LLM 未初始化，使用默认评分"})

    doc_type_label = _get_doc_type_label(doc_type)

    prompt = f"""You are a medical device documentation expert. Quickly assess the completeness of this document section.

Document type: {doc_type_label or doc_type}

## Section Content
{section_text[:4000]}

## Task
Rate completeness on a 1-5 scale:
- 5: Well-structured, covers all expected elements, specific and actionable
- 4: Good coverage, minor gaps
- 3: Moderate, some expected elements missing
- 2: Sparse, significant gaps
- 1: Nearly empty or completely inadequate

## Output (JSON only, no markdown)
{{"score": <int 1-5>, "gaps": ["<gap description>", ...], "assessment": "<one sentence summary>"}}"""

    try:
        from langchain_core.messages import HumanMessage
        response = _llm_client.invoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)

        # Try to extract JSON
        json_match = __import__("re").search(r'\{.*\}', content, __import__("re").DOTALL)
        if json_match:
            return json_match.group(0)
        return json.dumps({"score": 3, "gaps": [], "assessment": content[:200]})
    except Exception as e:
        logger.error(f"assess_completeness 失败: {e}")
        return json.dumps({"score": 3, "gaps": [str(e)], "assessment": "评估过程出错"})


# ============ 工具5: 专项审核 ============

@tool
def audit_section(section_text: str, context: str, depth: str, doc_type: str) -> str:
    """对指定章节执行专项审核。
    根据 doc_type 选择对应的法规检查清单，逐条核对。

    Args:
        section_text: 章节全文
        context: 全局上下文（前几节发现的问题、需重点关注的标准）
        depth: 审核深度 (quick/standard/deep)
        doc_type: 文档类型

    Returns:
        审核结果文本，包含发现、评分、合规状态
    """
    if _llm_client is None:
        return "[ERROR] LLM 客户端未初始化"

    if _retriever is None:
        return "[ERROR] 知识库检索器未初始化"

    from langchain_core.messages import HumanMessage, SystemMessage

    # 检索相关法规
    try:
        docs = _retriever.search_sync(section_text[:2000], top_k=5)
        knowledge_context = _format_search_results(docs)
    except Exception as e:
        knowledge_context = f"[知识库检索失败: {e}]"

    doc_type_label = _get_doc_type_label(doc_type)

    depth_guidance = {
        "quick": "快速审核：仅进行基本合规检查，重点关注是否覆盖核心条款。",
        "standard": "标准审核：逐条对照检查清单进行全面审核。",
        "deep": "深度审核：详尽分析每条要求的符合程度，必须引用具体标准条款号。",
    }

    system_prompt = f"""You are a medical device documentation audit expert specializing in insulin pump systems.

## Audit Context
- Document type: {doc_type_label or doc_type}
- Audit depth: {depth} — {depth_guidance.get(depth, 'standard')}

## Reference Knowledge
{knowledge_context}

## Audit Principles
1. Compare section content against regulatory requirements
2. Identify gaps, inconsistencies, and missing elements
3. Rate severity of each finding: critical / major / minor / suggestion
4. Cite specific standard clauses in findings
5. Provide actionable corrective recommendations

## Output Format
Output the audit result with these sections:
1. **原文摘要**: Brief summary of section content
2. **标准要求**: Key regulatory requirements from knowledge base
3. **发现项**: Numbered findings with severity (critical/major/minor/suggestion)
4. **差距分析**: Detailed gap analysis with standard clause references
5. **修改建议**: Specific, actionable recommendations
6. **严重度评级**: Overall severity (🔴严重缺失 / 🟡需要修改 / 🟢基本符合)
7. **量化评分**: Score 0-10 in dimensions: completeness(30%), compliance(25%), traceability(20%), consistency(15%), actionability(10%)

If this is a re-audit, also note: {context if context else 'N/A (first audit)'}"""

    try:
        response = _llm_client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"## Section to Audit\n\n{section_text[:8000]}")
        ])
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.error(f"audit_section 工具失败: {e}")
        return f"[审核过程出错: {str(e)}]"


# ============ 工具工厂函数 ============

def create_audit_tools(retriever_instance, llm_client):
    """
    创建绑定到具体实例的工具列表。
    因为 @tool 装饰器生成的函数是静态的，需要通过模块级全局变量注入外部依赖。

    Args:
        retriever_instance: RAGRetriever 实例
        llm_client: ChatOpenAI 实例

    Returns:
        List[BaseTool]: 工具列表
    """
    global _retriever, _llm_client, _DOC_TYPE_LABELS
    _retriever = retriever_instance
    _llm_client = llm_client

    # 缓存 doc_type labels
    if hasattr(retriever_instance, "_DOC_TYPE_LABELS"):
        _DOC_TYPE_LABELS = retriever_instance._DOC_TYPE_LABELS

    return [
        parse_document,
        rag_search,
        check_regulation,
        assess_completeness,
        audit_section,
    ]


# ============ 工具列表（Phase 1） ============
PHASE1_TOOLS = [
    parse_document,
    rag_search,
    check_regulation,
    assess_completeness,
    audit_section,
]
