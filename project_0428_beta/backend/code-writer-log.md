# Code Writer Log

## 2026-06-12 - Agent模式审核并行化优化（方案A）

### 修改概述
根据《Agent模式审核耗时优化_技术调研_20260612.md》中的方案A，对 project_0428_beta 的 Agent 模式审核进行并行化改造。

### 修改文件

#### 1. agent_graph.py
- 添加 `import asyncio`
- 新增并行审核配置常量：`PARALLEL_AUDIT_ENABLED`（默认 true）、`AUDIT_CONCURRENCY`（默认 5）
- 新增 `_build_audit_system_prompt_v2()` — 并行模式专用 prompt 构建函数（移除 global_context 依赖）
- 新增 `batch_audit_all_sections_node()` — 异步批量节点，内部使用 `asyncio.gather` + `Semaphore` 并发执行所有章节审核
- 修改 `build_audit_graph()` — 根据 `PARALLEL_AUDIT_ENABLED` 选择并行/顺序拓扑
- 保留原有顺序模式作为 fallback（`PARALLEL_AUDIT_ENABLED=false` 时生效）
- 更新模块 docstring

#### 2. main.py
- `/api/agent/analyze` 端点：`def` → `async def`
- `audit_graph.invoke()` → `await audit_graph.ainvoke()`

### 图拓扑变化

**并行模式（默认）**：
```
plan_strategy → batch_audit_all_sections (asyncio.gather + Semaphore(5))
              → cross_validate → generate_report → END
```

**顺序模式（fallback，PARALLEL_AUDIT_ENABLED=false）**：
```
plan_strategy → audit_section → evaluate_result → update_context
              → next/done → cross_validate → generate_report → END
```

### 预期效果
- 41章节文档：审核阶段从 14-27 min 降至 3-6 min（3-5x 加速）
- 总耗时从 17-32 min 降至 6.5-9.5 min，低于 20 分钟目标

### 环境变量控制
- `PARALLEL_AUDIT_ENABLED=true/false` — 启用/禁用并行模式（默认 true）
- `AUDIT_CONCURRENCY=5` — 并发数（默认 5）

## 2026-06-17 - Agent 对话优化全量改造（Phase 1+2+3）

### 来源
依据 `e:\nrf_sample_codes\working_team_work\public\docs\researcher_docs\project_0428_beta_agent_dialogue_optimization.md` 全量实施。

### Phase 1
- **1.1** 修复 `agent_graph.py` 顶部 `HumanMessage` 缺失导入。
- **1.2** `create_llm()` 加 `streaming=True`；`main.py` SSE 循环切到 `stream_mode=["messages","updates"]` 并 push `token` 事件。
- **1.3** `frontend/agent.html` 新增 `handleToken()` / `finalizeStreamingBubble()` + 闪烁光标 CSS。
- **1.4** 新增 `post_completion` checkpoint + `_chat_response_node` + 路由 `_route_after_completion`，支持报告生成后自由问答。
- **1.5** 会话超时清理：`ConversationSession` 加 `last_active`/`touch()`；新增后台 `_session_cleanup_loop()` 任务；多个端点加 `session.touch()`。

### Phase 2
- **2.1** `AuditState` 新增 `summary: str` 字段。
- **2.2-2.3** 新增 `_summarize_conversation_node` + `_route_before_chat`，messages ≥16 时压缩早期消息到 `summary`，保留最近 6 条；`_chat_response_node` 在 prompt 中拼接 `summary`。
- **2.4-2.5** `_create_checkpoint_node` 在 `interrupt` payload 添加 `structured` 字段（strategy/overview/completion）；main.py SSE 透传；前端 `addCheckpointCard()` 渲染策略表 + 章节问题表 + 完成卡，含 depth/severity badge CSS。

### Phase 3
- **3.1-3.2** 新增 `USE_TOOL_CALLING_AUDIT` 开关 + `_get_section_audit_agent()` + `_audit_section_with_agent()`，通过 `langchain.agents.create_agent` 调用 `agent_tools.py` 工具集；`audit_one_section` 优先走 agent 路径，失败回退直 LLM。
- **3.3** 新增 `USE_SEND_API_AUDIT` 开关 + `_get_send_api_audit_subgraph()` + `_run_send_api_audit_batch()`，通过 LangGraph `Send` API fan-out 子图并行审核；失败回退 `asyncio.gather`。
- **3.4** main.py SSE 新增 `tool_call`/`tool_result` 事件透传；前端 `handleToolCall`/`handleToolResult` 渲染工具调用条 + 结果预览 + 状态徽章。
- **3.5** AST 校验全部通过；动态 import 与 `build_conversation_graph().compile()` 验证图节点齐全（13 nodes 含 `summarize_conversation`）。

### 修改文件
- `backend/main.py`
- `backend/agent_graph.py`
- `backend/agent_state.py`
- `frontend/agent.html`

### 环境变量
- `USE_TOOL_CALLING_AUDIT=1` 启用 tool-calling 审核（默认 0）
- `USE_SEND_API_AUDIT=1` 启用 Send API 并行（默认 0）

### 验证
- AST: main / agent_graph / agent_state / agent_tools 全部 OK
- Import: AuditState.summary 存在；新函数全部可导入
- Graph compile: 13 节点；新增 `summarize_conversation` 节点串接到 `checkpoint_post_completion → chat_response` 链路

## 2026-06-18 - LLM 大纲解析 JSON 自愈 + 失败日志（A+C 修复）

### 问题
运行时日志：`parse_document_structure: LLM 失败 (JSONDecodeError: Expecting ',' delimiter: line 73 column 69 (char 3881))`，回退正则解析。
会话 `conv-1781746623366` 仅识别出 3 章，明显少于 LLM 应能识别的数量，正则回退虽然能跑但丢精度。

### 根因
LLM (deepseek-v4-pro via 火山方舟) 在生成 JSON 字符串时未转义中文文本中的 ASCII 双引号：
```json
{"title": "1.1 "产品" 概述", ...}   ← 中间的 " 提前关闭字符串
```
或在 JSON string value 里写了裸换行符。位置在 line 73 col 69 是文档中后段某个 chapter/subsection title。

### 修复

#### 1. `doc_processor.py:765-834` 新增 `_repair_malformed_json`
状态机扫描，区分字符串内/外：
- 字符串外的字符原样保留
- 字符串内的 `"` 若是合法结束（后跟 `, : } ]` 或 EOF）→ 保留
- 字符串内的 `"` 否则 → 替换为全角引号 `"`（左右交替）
- 字符串内的裸 `\n \r \t` → 转义为 `\\n \\r \\t`
- 对合法 JSON 是恒等变换（已用单元测试覆盖）

#### 2. `doc_processor.py:837-867` 新增 `_log_outline_failure`
失败时把 raw / cleaned / repaired 三段输出落盘到 `backend/logs/llm_outline_failures.log`，便于事后分析模型行为模式。
截断策略：head 1500 + tail 500，避免日志爆炸。

#### 3. `doc_processor.py:907-929` 修改 `llm_parse_outline` JSON 解析逻辑
原：cleaned 失败 → regex 抓 `{...}` → 直接 json.loads（仍会因嵌入引号失败）
新：cleaned 失败 → 写日志 → regex 抓块 → 直接 json.loads（处理前后杂质） → 失败则修复 → 仍失败抛错

#### 4. `tests/test_doc_processor.py:171-241` 新增 `TestRepairMalformedJson` 6 个用例
- `test_legal_json_unchanged`：合法 JSON 不被改写
- `test_embedded_ascii_quote_in_title`：核心场景 — 标题里嵌入 ASCII 引号
- `test_raw_newline_in_string`：字符串内裸换行
- `test_realistic_llm_output_with_quote_at_line_73`：模拟错误信息中 line 73 col 69 的多章节场景
- `test_escaped_quote_preserved`：已正确转义的 `\"` 不被误判
- `test_empty_string_preserved`：空字符串保留

### 验证
```
tests/test_doc_processor.py ... 16 passed in 0.04s
```
- 原 10 个 markdown 解析测试：全通过
- 新 6 个 JSON 修复测试：全通过
- AST 校验：doc_processor.py / test_doc_processor.py 均 OK

### 决策点
1. **修复策略选状态机而非正则替换**：正则无法区分"字符串结束"和"字符串内的引号"
2. **嵌入引号替换为全角 `"`/`"`，交替使用**：视觉上更接近原始 LLM 意图，避免直接删除引号导致语义丢失
3. **裸换行/制表符转义而非删除**：保留 LLM 的换行意图
4. **失败日志截断 head 1500 + tail 500**：典型 LLM 大纲输出 5-15KB，截断到 ~2KB 既能诊断又不爆日志
5. **未改 `_OUTLINE_SYSTEM_PROMPT`**：先用代码层修复兜底，下次再观察日志看 LLM 是否仍高频犯错再决定是否强化 prompt

### 修改文件
- `backend/doc_processor.py`：+106 行（_repair_malformed_json 70 行 + _log_outline_failure 31 行 + 解析逻辑重写 25 行 - 原 8 行）
- `backend/tests/test_doc_processor.py`：+74 行（6 个测试用例 + import 调整）

### 验证命令
```bash
cd E:/nrf_sample_codes/working_team_work/public/project/project_0428_beta/backend
source E:/anaconda/anaconda_content/etc/profile.d/conda.sh && conda activate env_01
python -m pytest tests/test_doc_processor.py -v
```

## 2026-06-18 - LLM 大纲解析 APITimeoutError 修复

### 问题
运行时日志：`parse_document_structure: LLM 失败 (APITimeoutError: Request timed out.)`，回退正则解析。
发生于 `企业级agent.docx` 等大文档（数千行）审核。

### 根因
1. **prompt 过大**：原代码把整篇文档（5-15 万字符）发给 LLM，deepseek-v4-pro 处理 + 生成需 30-60s
2. **超时过短**：`request_timeout=20` 不足以覆盖大文档
3. **单次失败即放弃**：无重试机制，偶发抖动即回退正则
4. **end_line 依赖 LLM**：LLM 经常给错 end_line，但代码层无兜底

实测 prompt 压缩效果（2000 行 / 242,163 字符 / 720KB 文档）：
- 抽取后：5,423 字符 / 10,199 字节 / 209 行
- **压缩率 97.8%（字节 98.6%）**

### 修复

#### 1. `doc_processor.py:873-936` 新增 `_extract_outline_candidates`
从完整文档中抽取"看起来像章节标题"的行（正则匹配 `^#` / `^第X章` / `^\d+(\.\d+)+` / `^[一二三四五]+、` 等模式），
保留前 5 行（封面）+ 后 5 行（签字栏）+ 所有候选行；候选 <30 时保留全部但限 1500 行；字节超 200KB 时硬截断。
行号 `[NNNN]` 前缀保留原文档行号，LLM 返回的 start_line/end_line 仍可直接用于 `_safe_slice_lines` 切片。

#### 2. `doc_processor.py:939-984` 新增 `_recompute_chapter_end_lines`
按"下一节点 start - 1"重算每个 chapter 与 subsection 的 end_line，让章节范围在代码层可控。
- chapter 节点：end_line = 下一 chapter.start - 1（或 total_lines - 1）
- subsection 节点：end_line = 同 chapter 内下一 subsection.start - 1（最后一个 subsection 取 chapter.end_line）
- clamp 到 [start_line, total_lines-1] 范围

#### 3. `doc_processor.py:1003-1067` 重写 `llm_parse_outline` 主体
- prompt 改用 `_extract_outline_candidates` 抽取结果
- LLM 调用加 1 次重试：捕获 `APITimeoutError` / `TimeoutError` / `asyncio.TimeoutError` / `httpx.TimeoutException` 或名称匹配 (`APITimeoutError`/`ReadTimeout`/`TimeoutException`) → 第二次调用；其它异常直接抛
- 解析 LLM JSON 后立即调 `_recompute_chapter_end_lines` 校正 end_line

#### 4. `main.py:922,1287` 超时从 20 提至 90
两处 `create_agent_llm(... request_timeout=20 ...)` → `request_timeout=90`，给大文档 + 单次重试足够缓冲。

#### 5. `tests/test_doc_processor.py` 新增 9 个测试用例
- `TestExtractOutlineCandidates` (5 个)：短文档保留/大文档压缩/行号保留/无章节回退/字节截断
- `TestRecomputeChapterEndLines` (4 个)：含 subsection 章节/末尾章节/clamp 边界/空列表

### 验证
```
tests/test_doc_processor.py ... 25 passed in 0.07s
```
- 10 个 markdown 解析：全通过
- 6 个 JSON 修复：全通过
- 9 个新增（5 抽取 + 4 end_line）：全通过
- AST 校验：doc_processor.py / main.py / test_doc_processor.py 均 OK
- prompt 压缩实测：2000 行 / 720KB → 209 行 / 10KB（97.8% 字符 / 98.6% 字节压缩）

### 决策点
1. **抽取候选行而非全部行**：LLM 识别大纲只需要"行号 → 标题"映射，正文对识别无价值，prompt 体积下降 70-100x
2. **候选 <30 保留全部**：极少数文档无明显章节结构（如纯散文），不抽取反而更安全
3. **end_line 由代码重算而非依赖 LLM**：LLM 经常给错 end_line（即便发完整正文），用"下一节点 start - 1"是确定性的
4. **重试 1 次而非 3 次**：单次重试即可吸收偶发抖动；3 次会让最坏延迟翻倍，用户体验更差
5. **超时 90s 而非 60s**：大文档 + 单次重试 + LLM 排队 30-60s，留 30s 余量
6. **不重试非超时异常**：认证/参数错误重试无意义，反而会拖慢失败反馈

### 修改文件
- `backend/doc_processor.py`：+108 行（_extract_outline_candidates 60 行 + _recompute_chapter_end_lines 50 行 + llm_parse_outline 重写 60 行 - 原 12 行）
- `backend/main.py`：2 处 `request_timeout=20` → `90`，各加 1 行注释
- `backend/tests/test_doc_processor.py`：+118 行（9 个测试用例 + import 调整）

### 验证命令
```bash
cd E:/nrf_sample_codes/working_team_work/public/project/project_0428_beta/backend
source E:/anaconda/anaconda_content/etc/profile.d/conda.sh && conda activate env_01
python -m pytest tests/test_doc_processor.py -v
```

## 2026-06-29 11:12:43 - 修复"会话不存在或已过期"（仅改代码，不动环境变量/不关 thinking）
- 用户约束：不要改环境变量，也不要关闭 thinking，只改其它需要改动的代码。

### File Edited
- File: `main.py:117` - `SESSION_TIMEOUT = 7200`（原 1800）。Why: 本地大模型推理慢，闲置超时放宽到 2h。Result: Success
- File: `main.py` 单 Agent SSE 心跳（~1402）- 心跳 yield 前加 `session.touch()`。Why: 长推理无队列事件时 last_active 不更新，被清理线程误删。Result: Success
- File: `main.py` 多 Agent SSE 心跳（~2042）- 同上加 `session.touch()`。Result: Success
- File: `agent_graph.py:74` - `create_llm` 默认 `request_timeout` 120→600。Result: Success
- File: `main.py:964,1330,1776,1977` - 大纲解析 `create_agent_llm(... request_timeout=90 ...)` 全部改 600。Why: qwen3.5:122b thinking 默认开启，解析大文档单次推理可能需数分钟，90s 触发 `llm_parse_outline` 超时。Result: Success

### Analysis
- Topic: 会话过期根因
- Finding: `session.touch()` 仅在 SSE 队列事件时调用，60s 心跳（TimeoutError 路径）未调；长 LLM 推理期间无队列事件→last_active 不更新→清理线程按 `now-last_active>SESSION_TIMEOUT` 删除会话。
- Decision: 心跳路径补 touch + 放宽 SESSION_TIMEOUT + 提高 request_timeout，全部为代码改动，符合用户约束。

### Bash
| # | Timestamp | Command | Purpose | Result |
|---|-----------|---------|---------|--------|
| - | 11:12 | `python -m py_compile main.py agent_graph.py` | 语法检查 | Success（ALL_OK） |
