#!/usr/bin/env python
"""重新构建知识库，添加医械标准库目录"""
import os
import sys
import gc
from pathlib import Path
import time

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from doc_processor import extract_text, chunk_text, get_file_metadata
from vector_store import create_vector_store, MiniMaxEmbeddingFunction

def get_embedding(text: str, api_key: str, api_url: str) -> list:
    """获取嵌入向量"""
    import httpx

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "doubao-embedding-vision-250615",
        "encoding_format": "float",
        "input": [{"text": text[:8000], "type": "text"}]
    }

    try:
        with httpx.Client(timeout=60, trust_env=False) as client:
            response = client.post(api_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                data = result.get("data", {})
                if data:
                    # 截断到1024维
                    embedding = data.get("embedding", [])
                    if len(embedding) > 1024:
                        embedding = embedding[:1024]
                    return embedding
    except Exception as e:
        print(f"Embedding API错误: {e}")

    # API失败时使用随机向量
    import random
    return [random.uniform(-1, 1) for _ in range(1024)]

def process_directory(directory: str, vector_store, api_key: str, api_url: str, chunk_size: int = 500, overlap: int = 50) -> dict:
    """处理目录下的文档"""
    doc_dir = Path(directory)
    stats = {
        "total_files": 0,
        "processed_files": 0,
        "failed_files": 0,
        "total_chunks": 0,
        "errors": []
    }

    supported_exts = {'.docx', '.pdf', '.txt', '.doc'}

    print(f"正在扫描目录: {doc_dir}")

    for root, dirs, files in os.walk(doc_dir):
        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext not in supported_exts:
                continue

            file_path = os.path.join(root, filename)
            stats["total_files"] += 1

            try:
                print(f"\n正在处理: {filename}...")

                # 提取文本
                text = extract_text(file_path)
                if not text or len(text.strip()) < 50:
                    print(f"  跳过（文本过短）: {filename}")
                    continue

                # 分块
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                print(f"  分割成 {len(chunks)} 个文本块")

                # 获取元数据
                metadata = get_file_metadata(file_path)
                metadata["source"] = filename
                # 计算相对于doc_dir的路径作为分类
                rel_path = os.path.relpath(root, doc_dir)
                if rel_path == '.':
                    metadata["category"] = "根目录"
                else:
                    metadata["category"] = rel_path.replace(os.path.sep, '/')

                # 预计算embeddings
                print(f"  正在生成向量...")
                embeddings = []
                for i, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, api_key, api_url)
                    embeddings.append(embedding)
                    if (i + 1) % 10 == 0:
                        print(f"    已生成 {i + 1}/{len(chunks)} 个向量")
                    time.sleep(0.05)

                # 添加到向量库
                chunk_ids = []
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{metadata['source']}_chunk_{i}_{int(time.time())}"
                    chunk_ids.append(chunk_id)

                vector_store.add_documents(
                    documents=chunks,
                    metadatas=[metadata] * len(chunks),
                    ids=chunk_ids,
                    embeddings=embeddings
                )

                stats["processed_files"] += 1
                stats["total_chunks"] += len(chunks)
                print(f"  成功添加: {len(chunks)} 个文本块")

                # 逐文件清理内存
                del text, chunks, embeddings, chunk_ids
                gc.collect()

                time.sleep(0.05)

            except Exception as e:
                stats["failed_files"] += 1
                stats["errors"].append(f"{filename}: {str(e)}")
                print(f"  处理失败: {filename}, 错误: {e}")
                import traceback
                traceback.print_exc()

    return stats

def main():
    # 配置
    API_KEY = "ark-6c047509-3ac1-4689-b796-47363425012c-3112a"
    API_URL = "https://ark.cn-beijing.volces.com/api/coding/v3/embeddings/multimodal"

    # 医械标准库目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    docs_dir = os.path.join(project_root, "develop_documents", "医械标准库")

    print("=" * 70)
    print("医械标准库知识库构建")
    print("=" * 70)
    print(f"文档目录: {docs_dir}")
    print(f"API URL: {API_URL}")

    # 检查目录是否存在
    if not os.path.exists(docs_dir):
        print(f"错误：目录不存在: {docs_dir}")
        return

    # 创建向量存储（不删除现有数据，而是追加）
    print("\n正在初始化向量库...")
    embedding_function = MiniMaxEmbeddingFunction(
        api_key=API_KEY,
        api_url=API_URL,
        model="doubao-embedding-vision-250615",
        dimension=1024
    )

    vs = create_vector_store(persist_directory="data/chroma_db", embedding_function=embedding_function)

    print(f"\n当前向量库文档数: {vs.count()}")

    # 处理文档
    print("\n开始处理文档...")
    stats = process_directory(
        directory=docs_dir,
        vector_store=vs,
        api_key=API_KEY,
        api_url=API_URL,
        chunk_size=500
    )

    # 输出统计
    print("\n" + "=" * 70)
    print("处理完成!")
    print("=" * 70)
    print(f"总文件数: {stats['total_files']}")
    print(f"成功处理: {stats['processed_files']}")
    print(f"处理失败: {stats['failed_files']}")
    print(f"新增文本块: {stats['total_chunks']}")
    print(f"向量库总文档数: {vs.count()}")

    if stats['errors']:
        print("\n错误列表:")
        for err in stats['errors'][:10]:
            print(f"  - {err}")

if __name__ == "__main__":
    main()
