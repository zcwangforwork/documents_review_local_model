#!/usr/bin/env python
"""对比文档目录和向量库内容"""
import os
import sys
from pathlib import Path
from collections import defaultdict

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from vector_store import create_vector_store

def scan_directory(base_dir):
    """扫描目录下所有支持的文件"""
    supported_exts = {'.docx', '.pdf', '.txt', '.doc'}
    all_files = []

    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext not in supported_exts:
                continue

            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, base_dir)
            all_files.append({
                'filename': filename,
                'full_path': full_path,
                'rel_path': rel_path,
                'directory': os.path.relpath(root, base_dir)
            })

    return sorted(all_files, key=lambda x: x['rel_path'])

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    docs_dir = os.path.join(project_root, "develop_documents")

    print("=" * 80)
    print("文档目录与向量库对比检查")
    print("=" * 80)
    print(f"文档目录: {docs_dir}")

    if not os.path.exists(docs_dir):
        print(f"\n错误: 目录不存在: {docs_dir}")
        return

    # 扫描目录
    print("\n正在扫描文档目录...")
    dir_files = scan_directory(docs_dir)
    print(f"目录中找到 {len(dir_files)} 个支持的文档文件\n")

    # 加载向量库
    print("正在加载向量库...")
    vs = create_vector_store(persist_directory="data/chroma_db")
    print(f"向量库总文档数: {vs.count()} 个文本块\n")

    # 分页获取向量库中的文档信息（避免一次性加载100000条导致OOM）
    print("正在分页获取向量库中的文档信息...")
    sources_in_db = defaultdict(int)

    total_count = vs.count()
    page_size = 2000
    offset = 0
    while offset < total_count:
        batch = vs.collection.get(
            limit=page_size,
            offset=offset,
            include=['metadatas']
        )
        for meta in batch.get('metadatas', []):
            if meta:
                source = meta.get('source', '未知')
                sources_in_db[source] += 1
        offset += page_size
        print(f"  已获取 {min(offset, total_count)}/{total_count} 条...")

    print(f"向量库中有 {len(sources_in_db)} 个不同的来源文件\n")

    # 对比
    print("=" * 80)
    print("对比结果:")
    print("=" * 80)

    # 整理目录中的文件
    filenames_in_dir = set()
    dir_structure = defaultdict(list)

    for f in dir_files:
        filenames_in_dir.add(f['filename'])
        dir_structure[f['directory']].append(f['filename'])

    # 检查已摄入的
    ingested_files = []
    missing_files = []

    for f in dir_files:
        if f['filename'] in sources_in_db:
            ingested_files.append(f)
        else:
            missing_files.append(f)

    print(f"\n目录中总文件数: {len(dir_files)}")
    print(f"已摄入向量库: {len(ingested_files)} 个文件")
    print(f"未摄入: {len(missing_files)} 个文件\n")

    if missing_files:
        print("=" * 80)
        print("未摄入的文件列表:")
        print("=" * 80)
        for i, f in enumerate(missing_files[:100], 1):
            print(f"  {i}. {f['rel_path']}")

        if len(missing_files) > 100:
            print(f"\n  ... 还有 {len(missing_files) - 100} 个文件")

    # 显示目录结构统计
    print("\n" + "=" * 80)
    print("目录结构统计:")
    print("=" * 80)

    for directory in sorted(dir_structure.keys()):
        files = dir_structure[directory]
        ingested_count = sum(1 for f in files if f in ingested_files)
        print(f"\n目录: {directory if directory else '(根目录)'}")
        print(f"  文件数: {len(files)}")
        print(f"  已摄入: {ingested_count}")
        print(f"  未摄入: {len(files) - ingested_count}")

        if len(files) - ingested_count > 0:
            missing_in_dir = [f for f in files if f not in ingested_files]
            if missing_in_dir:
                print("  未摄入文件:")
                for f in missing_in_dir[:10]:
                    print(f"    - {f}")
                if len(missing_in_dir) > 10:
                    print(f"    ... 还有 {len(missing_in_dir) - 10} 个")

    # 统计已摄入的详细信息
    print("\n" + "=" * 80)
    print("已摄入文件详细信息（前50个）:")
    print("=" * 80)

    sorted_ingested = sorted(
        [(f['filename'], sources_in_db[f['filename']]) for f in ingested_files],
        key=lambda x: x[1], reverse=True
    )

    for i, (filename, count) in enumerate(sorted_ingested[:50], 1):
        print(f"  {i}. {filename}: {count} 个文本块")

    total_blocks_ingested = sum(count for _, count in sorted_ingested)
    print(f"\n已摄入文件总文本块数: {total_blocks_ingested}")

    # 检查是否有向量库中有但目录中没有的文件
    print("\n" + "=" * 80)
    print("向量库中有但目录中没有的文件:")
    print("=" * 80)

    extra_files = []
    for source in sources_in_db.keys():
        if source not in filenames_in_dir:
            extra_files.append((source, sources_in_db[source]))

    if extra_files:
        print(f"\n找到 {len(extra_files)} 个额外的文件:\n")
        for i, (filename, count) in enumerate(extra_files[:50], 1):
            print(f"  {i}. {filename}: {count} 个文本块")
    else:
        print("\n向量库中的文件与目录中的文件完全匹配")

    print("\n" + "=" * 80)
    print("检查完成!")
    print("=" * 80)
    print(f"\n总文件数: {len(dir_files)}")
    print(f"已摄入: {len(ingested_files)} ({len(ingested_files)/len(dir_files)*100:.1f}%)")
    print(f"未摄入: {len(missing_files)} ({len(missing_files)/len(dir_files)*100:.1f}%)")

if __name__ == "__main__":
    main()
