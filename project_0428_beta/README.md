# 医疗器械体系文件审核 Agent

基于 FastAPI + MiniMax-M2.7 + ChromaDB RAG 的医疗器械企业体系文件智能审核系统。

## 功能特性

- **文档上传审核**: 支持 Word (.docx) 和 PDF 文件上传
- **RAG 智能检索**: 结合预处理的知识库进行相关文档检索
- **体系文件审核**: 指出缺失内容、需要修改的地方、具体修改建议
- **法规条款关联**: 关联 ISO 13485、ISO 14971、IEC 62304、MDR、NMPA GMP 等标准
- **Web 对话界面**: 深色主题，现代化 UI，支持拖拽上传

## 技术栈

- **后端**: Python 3.10+ / FastAPI / httpx / ChromaDB
- **前端**: HTML / JavaScript / TailwindCSS
- **模型**: MiniMax-M2.7 (对话) + embed-multilingual-v2 (向量化)
- **知识库**: ChromaDB 本地向量存储

## 项目结构

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
├── develop_documents/      # 知识库源文件（医疗器械体系模板）
├── SPEC.md                 # 规格文档
└── README.md
```

## 快速部署

### 1. 环境要求

- Python 3.10+
- Conda 环境（env_01）
- MiniMax Token Plan API Key

### 2. 安装依赖

```bash
conda activate env_01
cd backend
pip install -r requirements.txt
```

### 3. 预处理知识库

```bash
cd backend
python build_knowledge_base.py --docs-dir ../develop_documents --api-key YOUR_API_KEY
```

### 4. 启动服务

```bash
python main.py
```

服务启动后访问：**http://localhost:8000/**

### 5. 使用

1. 访问 http://localhost:8000/
2. 点击右上角"上传体系文件"按钮
3. 拖拽或选择 Word/PDF 文件
4. 选择处理模式：
   - **智能审核（推荐）**: 使用 RAG 技术结合知识库进行深度分析
   - **仅提取文本**: 提取文档文本内容
   - **添加到对话**: 将文档添加到对话上下文

## 审核范围

本系统可审核以下类型的体系文件：

- 质量手册
- 程序文件（SOP）
- 设计开发文件（DHF）
- 风险管理文档（ISO 14971）
- 软件生命周期文档（IEC 62304）
- 采购/供应商管理文件
- 生产/检验文件
- 不良事件/CAPA/召回文件

参考法规标准：
- ISO 13485:2016 医疗器械质量管理体系
- ISO 14971:2019 医疗器械风险管理
- IEC 62304 医疗器械软件生命周期过程
- EU MDR 2017/745 欧盟医疗器械法规
- FDA 21 CFR Part 820 质量体系法规
- NMPA 医疗器械生产质量管理规范

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `GET /` | GET | 前端页面 |
| `GET /health` | GET | 健康检查（含向量库状态） |
| `POST /api/chat` | POST | 通用聊天接口 |
| `POST /api/upload` | POST | 上传文档（multipart/form-data） |
| `POST /api/analyze` | POST | RAG 智能审核 |
| `POST /api/clear` | POST | 清除会话历史 |
| `GET /api/history/{session_id}` | GET | 获取会话历史 |
| `GET /api/vectorstore/status` | GET | 向量库状态 |

### 上传文档示例

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@体系文件.docx" \
  -F "question=请审核这份文件" \
  -F "session_id=default"
```

## 注意事项

1. **API Key 安全**: 不要将 API Key 提交到公共仓库
2. **知识库构建**: 首次使用需运行 `build_knowledge_base.py` 预处理文档
3. **审核结果**: 本系统输出仅供参考，请结合法规专家意见使用
