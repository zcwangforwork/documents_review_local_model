"""多 Agent 包基础单元测试

测试 multi_agent 包的核心组件：
- state.py: MultiAgentState 类型 + 初始状态工厂
- supervisor_graph.py: build/compile 流程 + Send API 派发逻辑
- agents.py: _build_chapter_summary 摘要生成 helper

不调用真实 LLM/RAG，使用 dummy 替身。

运行方式:
    cd backend
    python -m pytest test_multi_agent.py -v
或:
    python test_multi_agent.py
"""
from __future__ import annotations

import sys
import unittest
from typing import List

sys.path.insert(0, ".")


class TestMultiAgentState(unittest.TestCase):
    """state.py — 状态工厂与字段约束"""

    def test_initial_state_required_fields(self):
        from multi_agent import make_multi_agent_initial_state

        s = make_multi_agent_initial_state(
            document_text="hello",
            document_type="design_dev",
            outline=[{"title": "ch1"}],
            sections=[{"title": "1.1", "content": "x"}],
        )
        self.assertEqual(s["document_text"], "hello")
        self.assertEqual(s["document_type"], "design_dev")
        self.assertEqual(s["chapter_results"], [])
        self.assertEqual(s["finished"], False)
        self.assertEqual(s["total_chapters"], 0)
        self.assertEqual(s["total_sections"], 0)
        self.assertEqual(s["conversation_mode"], False)

    def test_initial_state_conversation_mode(self):
        from multi_agent import make_multi_agent_initial_state

        s = make_multi_agent_initial_state(
            document_text="x",
            document_type="risk_management",
            conversation_mode=True,
        )
        self.assertEqual(s["conversation_mode"], True)
        self.assertEqual(s["current_stage"], "intake")
        self.assertEqual(s["outline"], [])
        self.assertEqual(s["sections"], [])


class TestSupervisorGraph(unittest.TestCase):
    """supervisor_graph.py — 图构建与编译"""

    def setUp(self):
        class DummyLLM:
            def invoke(self, *a, **k):
                return type("R", (), {"content": '{"confidence": 3, "key_findings": []}'})()

            async def ainvoke(self, *a, **k):
                return type("R", (), {"content": '{"confidence": 3, "key_findings": []}'})()

        class DummyRetriever:
            def retrieve(self, *a, **k):
                return []

        self.llm = DummyLLM()
        self.retriever = DummyRetriever()

    def test_build_graph_has_expected_nodes(self):
        from multi_agent import build_supervisor_graph

        graph = build_supervisor_graph(llm=self.llm, retriever=self.retriever)
        # StateGraph stores nodes in a private dict; check via compile()
        compiled = graph.compile()
        node_names = set(compiled.nodes.keys())
        self.assertIn("analyze_structure", node_names)
        self.assertIn("audit_chapter", node_names)
        self.assertIn("synthesize_report", node_names)

    def test_compile_supervisor_graph_no_checkpointer(self):
        from multi_agent import compile_supervisor_graph

        compiled = compile_supervisor_graph(llm=self.llm, retriever=self.retriever)
        self.assertIsNotNone(compiled)
        self.assertTrue(hasattr(compiled, "ainvoke"))
        self.assertTrue(hasattr(compiled, "astream"))


class TestDispatchChapters(unittest.TestCase):
    """supervisor_graph._dispatch_chapters — Send API fan-out 逻辑"""

    def test_dispatch_with_valid_chapters(self):
        from multi_agent.supervisor_graph import _dispatch_chapters
        from langgraph.types import Send

        state = {
            "chapter_structure": [
                {
                    "chapter_idx": 0,
                    "chapter_title": "第一章",
                    "subsection_indices": [0, 1],
                },
                {
                    "chapter_idx": 1,
                    "chapter_title": "第二章",
                    "subsection_indices": [2],
                },
            ],
            "sections": [
                {"title": "1.1", "content": "a"},
                {"title": "1.2", "content": "b"},
                {"title": "2.1", "content": "c"},
            ],
            "document_type": "design_dev",
        }
        sends = _dispatch_chapters(state)
        self.assertEqual(len(sends), 2)
        for s in sends:
            self.assertIsInstance(s, Send)
            self.assertEqual(s.node, "audit_chapter")
            self.assertIn("chapter_idx", s.arg)
            self.assertIn("subsections", s.arg)

    def test_dispatch_empty_chapter_structure(self):
        from multi_agent.supervisor_graph import _dispatch_chapters

        state = {
            "chapter_structure": [],
            "sections": [],
            "document_type": "design_dev",
        }
        sends = _dispatch_chapters(state)
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0].node, "synthesize_report")
        self.assertTrue(sends[0].arg.get("_skip"))

    def test_dispatch_skips_chapters_without_subsections(self):
        from multi_agent.supervisor_graph import _dispatch_chapters

        state = {
            "chapter_structure": [
                {
                    "chapter_idx": 0,
                    "chapter_title": "空章节",
                    "subsection_indices": [],
                },
            ],
            "sections": [{"title": "x", "content": "y"}],
            "document_type": "design_dev",
        }
        sends = _dispatch_chapters(state)
        # 所有章节均无可审小节 → 跳过到 synthesize_report
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0].node, "synthesize_report")


class TestChapterSummary(unittest.TestCase):
    """agents._build_chapter_summary — 章节摘要文本生成"""

    def test_empty_subsections(self):
        from multi_agent.agents import _build_chapter_summary

        out = _build_chapter_summary("空章节", [])
        self.assertIn("空章节", out)
        self.assertIn("无可审核小节", out)

    def test_summary_aggregates_findings(self):
        from multi_agent.agents import _build_chapter_summary

        results = [
            {
                "title": "1.1",
                "breadcrumb": "Ch1 > 1.1",
                "confidence": 4,
                "findings": ["[Critical] 严重缺陷示例", "minor 微问题"],
            },
            {
                "title": "1.2",
                "breadcrumb": "Ch1 > 1.2",
                "confidence": 3,
                "findings": ["🟡 需修改某处"],
            },
        ]
        out = _build_chapter_summary("第一章", results)
        self.assertIn("第一章", out)
        self.assertIn("小节数: 2", out)
        # 至少一个严重项 (critical) + 一个 major (🟡)
        self.assertIn("🔴严重", out)
        self.assertIn("🟡需修改", out)


def main():
    unittest.main(verbosity=2, exit=False)


if __name__ == "__main__":
    main()
