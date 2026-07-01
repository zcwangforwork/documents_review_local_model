#!/usr/bin/env python
"""检查向量库中的文档内容"""
import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from vector_store import create_vector_store, MiniMaxEmbeddingFunction

def main():
    # 使用绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "data", "chroma_db")

    # API配置
    API_KEY = "ark-6c047509-3ac1-4689-b796-47363425012c-3112a"
    EMBEDDING_URL = "https://ark.cn-beijing.volces.com/api/coding/v3/embeddings/multimodal"
    EMBEDDING_MODEL = "doubao-embedding-vision-250615"

    print("=" * 70)
    print("检查向量库内容")
    print("=" * 70)

    # 创建 embedding function
    embedding_function = MiniMaxEmbeddingFunction(
        api_key=API_KEY,
        api_url=EMBEDDING_URL,
        model=EMBEDDING_MODEL,
        dimension=1024
    )

    # 创建向量存储
    vs = create_vector_store(persist_directory=db_path, embedding_function=embedding_function)

    print(f"\n向量库中文档总数: {vs.count()}")

    # 分页获取所有文档（避免一次性加载10000条导致OOM）
    print("\n正在分页获取所有文档...")
    sources = {}
    categories = {}
    ids_sample = []
    documents_sample = []
    metadatas_sample = []

    total_count = vs.count()
    page_size = 2000
    offset = 0
    while offset < total_count:
        batch = vs.collection.get(
            limit=page_size,
            offset=offset,
            include=['documents', 'metadatas']
        )
        batch_metas = batch.get('metadatas', [])
        for meta in batch_metas:
            if meta:
                source = meta.get('source', '未知')
                category = meta.get('category', '未知')
                sources[source] = sources.get(source, 0) + 1
                categories[category] = categories.get(category, 0) + 1

        if len(ids_sample) < 10:
            ids_sample.extend(batch.get('ids', [])[:10 - len(ids_sample)])
            metadatas_sample.extend(batch_metas[:10 - len(metadatas_sample)])
            documents_sample.extend(batch.get('documents', [])[:10 - len(documents_sample)])

        offset += page_size

    print(f"\n实际获取到的文档数: {total_count}")

    print("\n" + "=" * 70)
    print("文档来源统计:")
    print("=" * 70)
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count} 个文本块")

    print("\n" + "=" * 70)
    print("分类统计:")
    print("=" * 70)
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count} 个文本块")

    # 检查医械标准库相关的文档
    print("\n" + "=" * 70)
    print("医械标准库相关文档:")
    print("=" * 70)

    medical_standard_docs = []
    for source in sources.keys():
        if '医械' in source or '标准' in source or 'YY' in source or 'GB' in source or 'ISO' in source:
            medical_standard_docs.append((source, sources[source]))

    if medical_standard_docs:
        for source, count in sorted(medical_standard_docs):
            print(f"  {source}: {count} 个文本块")
        print(f"\n医械标准库相关文档总数: {sum(count for _, count in medical_standard_docs)} 个文本块")
    else:
        print("  未找到明确标注为医械标准库的文档")

    # 显示部分文档预览
    print("\n" + "=" * 70)
    print("文档预览（前10个）:")
    print("=" * 70)

    for i in range(min(10, len(ids_sample))):
        print(f"\n{i+1}. ID: {ids_sample[i]}")
        if i < len(metadatas_sample) and metadatas_sample[i]:
            print(f"   来源: {metadatas_sample[i].get('source', '未知')}")
            print(f"   分类: {metadatas_sample[i].get('category', '未知')}")
        if i < len(documents_sample) and documents_sample[i]:
            preview = documents_sample[i][:200].replace('\n', ' ')
            print(f"   内容预览: {preview}...")

    print("\n" + "=" * 70)
    print("检查完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()
