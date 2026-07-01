#!/usr/bin/env python
"""全面检查向量库内容"""
import os
import sys
from pathlib import Path
from collections import defaultdict

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from vector_store import create_vector_store

def main():
    print("=" * 70)
    print("向量库全面检查")
    print("=" * 70)

    # 创建向量存储
    vs = create_vector_store(persist_directory="data/chroma_db")

    total_count = vs.count()
    print(f"\n向量库总文档数: {total_count}")

    # 分页获取所有文档，避免一次性加载过多导致 OOM
    print("\n正在分页获取文档...")
    sources = defaultdict(int)
    categories = defaultdict(int)
    ids_sample = []
    metadatas_sample = []
    documents_sample = []

    page_size = 2000  # 每页2000条，降低内存压力
    offset = 0
    total_fetched = 0

    while offset < total_count:
        batch = vs.collection.get(
            limit=page_size,
            offset=offset,
            include=['documents', 'metadatas']
        )
        batch_ids = batch.get('ids', [])
        batch_metas = batch.get('metadatas', [])
        batch_docs = batch.get('documents', [])

        for i, meta in enumerate(batch_metas):
            if meta:
                source = meta.get('source', '未知')
                category = meta.get('category', '未知')
                sources[source] += 1
                categories[category] += 1

        # 保留前100条样本用于预览
        if len(ids_sample) < 100:
            ids_sample.extend(batch_ids[:100 - len(ids_sample)])
            metadatas_sample.extend(batch_metas[:100 - len(metadatas_sample)])
            documents_sample.extend(batch_docs[:100 - len(documents_sample)])

        total_fetched += len(batch_ids)
        offset += page_size
        print(f"  已获取 {min(total_fetched, total_count)}/{total_count} 条...")

    print(f"实际获取文档数: {total_fetched}")

    ids = ids_sample
    metadatas = metadatas_sample
    documents = documents_sample

    print("\n正在统计...")
    for i, meta in enumerate(metadatas):
        if meta:
            source = meta.get('source', '未知')
            category = meta.get('category', '未知')

            sources[source] += 1
            categories[category] += 1

    # 显示分类统计
    print("\n" + "=" * 70)
    print("分类统计（按类别）:")
    print("=" * 70)
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} 个文本块")

    # 显示来源统计（前50个）
    print("\n" + "=" * 70)
    print("来源统计（前50个文件）:")
    print("=" * 70)
    sorted_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)
    for i, (source, count) in enumerate(sorted_sources[:50]):
        print(f"  {i + 1}. {source}: {count} 个文本块")

    # 查找医械标准库相关的所有文件
    print("\n" + "=" * 70)
    print("医械标准库目录下的所有文件:")
    print("=" * 70)

    medical_related = []
    for source, count in sorted_sources:
        if any(keyword in source for keyword in ['YY', 'GB', 'GBT', 'ISO', 'IEC', '医疗', '器械', '标准']):
            medical_related.append((source, count))

    if medical_related:
        print(f"\n找到 {len(medical_related)} 个相关文件\n")
        for i, (source, count) in enumerate(medical_related[:100]):
            print(f"  {i + 1}. {source}: {count} 个文本块")

        total = sum(count for _, count in medical_related)
        print(f"\n总计: {len(medical_related)} 个文件，{total} 个文本块")
    else:
        print("  未找到相关文件")

    # 显示一些预览
    print("\n" + "=" * 70)
    print("文档预览（按ID前缀查找）:")
    print("=" * 70)

    preview_count = 0
    for i, (doc_id, meta) in enumerate(zip(ids, metadatas)):
        if preview_count >= 15:
            break

        if meta:
            if 'medical' in doc_id.lower() or '医械' in meta.get('category', '') or 'YY' in meta.get('source', ''):
                source = meta.get('source', '未知')
                category = meta.get('category', '未知')
                print(f"\n{preview_count + 1}. ID: {doc_id[:60]}...")
                print(f"   来源: {source}")
                print(f"   分类: {category}")
                if i < len(documents) and documents[i]:
                    preview = documents[i][:100].replace('\n', ' ')
                    print(f"   预览: {preview}...")
                preview_count += 1

    print("\n" + "=" * 70)
    print(f"检查完成！总文件数: {len(sources)}, 总文本块: {sum(sources.values())}")
    print("=" * 70)

if __name__ == "__main__":
    main()
