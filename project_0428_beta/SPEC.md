# 医疗器械体系文件审核 Agent - 规格文档

## 1. 项目概述

**项目名称**: 医疗器械体系文件审核 Agent
**项目类型**: RAG 增强的企业文档审核系统
**核心功能**: 用户上传体系文件（Word/PDF），结合预处理的法规知识库，通过大模型给出文件修改建议
**目标用户**: 医疗器械企业质量管理人员、注册工程师、体系审核员

## 2. 功能规格

### 2.1 核心功能

- [x] **文档上传**: 支持 .docx 和 .pdf 格式文件上传
- [x] **文本提取**: 从 Word 和 PDF 中自动提取文本内容
- [x] **智能审核**: 结合知识库进行 RAG 增强分析，输出审核结果
- [x] **修改建议**: 指出缺失内容、需要修改的地方、具体修改建议
- [x] **法规依据**: 关联 ISO 13485、ISO 14971、IEC 62304、MDR、NMPA GMP 等标准条款

### 2.2 用户交互

- [x] 文件拖拽上传
- [x] 上传进度显示
- [x] 多模式处理（智能审核/仅提取/添加到对话）
- [x] 聊天式对话界面
- [x] 会话历史管理

### 2.3 知识库

- [x] 预处理的 develop_documents 目录文件
- [x] ChromaDB 向量存储
- [x] 本地持久化

## 3. 技术架构

### 3.1 后端

- **框架**: FastAPI + httpx
- **向量库**: ChromaDB
- **文档处理**: python-docx, pdfplumber
- **Embedding**: MiniMax embed-multilingual-v2

### 3.2 前端

- **框架**: 原生 HTML + JavaScript + TailwindCSS
- **UI**: 深色主题，医疗器械行业风格

### 3.3 API 端点

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/` | 前端页面 |
| GET | `/health` | 健康检查 |
| GET | `/info` | 服务信息 |
| POST | `/api/chat` | 通用聊天 |
| POST | `/api/upload` | 上传文档 |
| POST | `/api/analyze` | RAG 分析 |
| POST | `/api/clear` | 清除会话 |
| GET | `/api/history/{session_id}` | 获取历史 |
| GET | `/api/vectorstore/status` | 向量库状态 |

## 4. 文件结构

```
project/
├── backend/
│   ├── main.py              # FastAPI 主程序
│   ├── doc_processor.py    # 文档处理模块
│   ├── vector_store.py     # ChromaDB 向量库
│   ├── rag_retriever.py    # RAG 检索逻辑
│   ├── build_knowledge_base.py  # 知识库预处理脚本
│   └── requirements.txt    # Python 依赖
├── frontend/
│   └── index.html          # 前端页面
├── develop_documents/      # 知识库源文件
├── SPEC.md
└── README.md
```

## 5. System Prompt

审核 Agent 使用专门的 System Prompt，聚焦于：
- ISO 13485:2016 质量管理体系
- ISO 14971:2019 风险管理
- IEC 62304 软件生命周期
- EU MDR / FDA QMSR / NMPA GMP

输出格式：
1. 缺失内容清单
2. 需要修改的地方
3. 具体修改建议
4. 关联法规条款依据

## 6. 验证方式

1. `conda run -n env_01 python backend/build_knowledge_base.py --docs-dir ../develop_documents` 预处理知识库
2. `conda run -n env_01 python backend/main.py` 启动服务
3. 访问 http://localhost:8000
4. 上传体系文件，选择"智能审核"模式
5. 查看审核结果

## 7. 已知限制

- Embedding API 需要有效的 MiniMax API Key
- 向量库初次构建需要处理大量文档，耗时较长
- PDF 扫描件（图片）无法提取文本
