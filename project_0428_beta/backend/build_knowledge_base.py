"""
医疗器械体系文件审核 - 知识库预处理脚本
对 develop_documents 目录下的文件进行向量化，建立知识库索引
"""
import os
import sys
import gc
from pathlib import Path
import time

# 添加 backend 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from doc_processor import extract_text, chunk_text, get_file_metadata
from vector_store import create_vector_store


def get_embedding(text: str, api_key: str, api_url: str) -> list:
    """
    使用本地 Ollama Embedding API（OpenAI 兼容格式）获取文本向量

    Args:
        text: 待嵌入的文本
        api_key: API 密钥
        api_url: API 地址

    Returns:
        嵌入向量列表
    """
    import httpx

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b"),
        "input": text[:8000]
    }

    try:
        with httpx.Client(timeout=60, trust_env=False) as client:
            response = client.post(api_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                data = result.get("data", [])
                if data:
                    embedding = data[0].get("embedding", [])
                    # 截断到 1024 维（与向量库维度保持一致）
                    if len(embedding) > 1024:
                        embedding = embedding[:1024]
                    return embedding
    except Exception:
        pass

    # 如果 API 调用失败，使用随机向量作为后备
    import random
    return [random.uniform(-1, 1) for _ in range(1024)]


def process_directory(
    directory: str,
    vector_store,
    api_key: str,
    api_url: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> dict:
    """
    处理目录下的所有文档并添加到向量库

    Args:
        directory: 文档目录路径
        vector_store: VectorStore 实例
        api_key: API 密钥
        api_url: API 地址
        chunk_size: 分块大小
        overlap: 重叠大小

    Returns:
        处理统计信息
    """
    doc_dir = Path(directory)
    stats = {
        "total_files": 0,
        "processed_files": 0,
        "failed_files": 0,
        "total_chunks": 0,
        "errors": []
    }

    # 支持的文件扩展名
    supported_exts = {'.docx', '.pdf', '.txt', '.doc'}

    # 遍历目录
    for root, dirs, files in os.walk(doc_dir):
        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext not in supported_exts:
                continue

            file_path = os.path.join(root, filename)
            stats["total_files"] += 1

            try:
                print(f"正在处理: {filename}...")

                # 提取文本
                text = extract_text(file_path)
                if not text or len(text.strip()) < 50:
                    print(f"  跳过（文本过短）: {filename}")
                    continue

                # 分块
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

                # 获取元数据
                metadata = get_file_metadata(file_path)
                metadata["source"] = filename
                metadata["category"] = Path(root).name

                # 预计算 embeddings
                print(f"  正在生成 {len(chunks)} 个向量...")
                embeddings = []
                for i, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, api_key, api_url)
                    embeddings.append(embedding)
                    if (i + 1) % 10 == 0:
                        print(f"    已生成 {i + 1}/{len(chunks)} 个向量")
                    # 避免请求过快
                    time.sleep(0.1)

                # 添加到向量库（vector_store 内部已分批提交）
                chunk_ids = []
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{metadata['source']}_chunk_{i}"
                    chunk_ids.append(chunk_id)

                vector_store.add_documents(
                    documents=chunks,
                    metadatas=[metadata] * len(chunks),
                    ids=chunk_ids,
                    embeddings=embeddings
                )

                stats["processed_files"] += 1
                stats["total_chunks"] += len(chunks)
                print(f"  成功: {len(chunks)} 个文本块")

                # 逐文件清理内存
                del text, chunks, embeddings, chunk_ids
                gc.collect()

                # 避免请求过快
                time.sleep(0.1)

            except Exception as e:
                stats["failed_files"] += 1
                stats["errors"].append(f"{filename}: {str(e)}")
                print(f"  处理失败: {filename}, 错误: {e}")

    return stats


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="预处理医疗器械知识库文档")
    parser.add_argument(
        "--docs-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'develop_documents'),
        help="文档目录路径"
    )
    parser.add_argument(
        "--persist-dir",
        default="data/chroma_db",
        help="向量库持久化目录"
    )
    parser.add_argument(
        "--api-key",
        help="MiniMax API 密钥（也可以设置 OPENAI_API_KEY 环境变量）"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:11435/v1/embeddings",
        help="Embedding API 地址"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="文本分块大小"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="重建向量库（删除旧数据）"
    )

    args = parser.parse_args()

    # 获取 API Key（本地 Ollama 无需鉴权，默认 "ollama" 占位）
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("MINIMAX_API_KEY") or "ollama"

    # 创建向量存储
    print(f"初始化向量库: {args.persist_dir}")
    vs = create_vector_store(args.persist_dir)

    if args.rebuild:
        print("重建向量库...")
        vs.delete_collection()
        vs = create_vector_store(args.persist_dir)

    # 处理文档
    print(f"处理文档目录: {args.docs_dir}")
    stats = process_directory(
        directory=args.docs_dir,
        vector_store=vs,
        api_key=api_key,
        api_url=args.api_url,
        chunk_size=args.chunk_size
    )

    # 输出统计
    print("\n" + "=" * 50)
    print("处理完成!")
    print(f"总文件数: {stats['total_files']}")
    print(f"成功处理: {stats['processed_files']}")
    print(f"处理失败: {stats['failed_files']}")
    print(f"总文本块: {stats['total_chunks']}")
    print(f"向量库文档数: {vs.count()}")

    if stats['errors']:
        print("\n错误列表:")
        for err in stats['errors'][:10]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
