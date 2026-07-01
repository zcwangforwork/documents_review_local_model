#!/usr/bin/env python
"""构建所有医械标准库文档，带详细进度"""
import os
import sys
import gc
from pathlib import Path
import random

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from doc_processor import extract_text, chunk_text, get_file_metadata
from vector_store import create_vector_store

def get_random_embedding() -> list:
    """生成随机向量"""
    return [random.uniform(-1, 1) for _ in range(1024)]

def main():
    # 医械标准库目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    docs_dir = os.path.join(project_root, "develop_documents", "医械标准库")

    print("=" * 70)
    print("医械标准库知识库构建 - 完整版")
    print("=" * 70)
    print(f"文档目录: {docs_dir}")

    # 检查目录是否存在
    if not os.path.exists(docs_dir):
        print(f"错误：目录不存在: {docs_dir}")
        return

    # 扫描所有文件
    print("\n正在扫描所有文件...")
    all_files = []
    supported_exts = {'.docx', '.pdf', '.txt', '.doc'}

    for root, dirs, files in os.walk(docs_dir):
        for filename in sorted(files):
            ext = Path(filename).suffix.lower()
            if ext not in supported_exts:
                continue
            file_path = os.path.join(root, filename)
            all_files.append((root, filename, file_path))

    print(f"找到 {len(all_files)} 个文件\n")

    # 创建向量存储
    print("正在初始化向量库...")
    vs = create_vector_store(persist_directory="data/chroma_db")
    print(f"当前向量库文档数: {vs.count()}\n")

    # 处理文档
    stats = {
        "total_files": len(all_files),
        "processed_files": 0,
        "failed_files": 0,
        "total_chunks": 0,
        "errors": []
    }

    for idx, (root, filename, file_path) in enumerate(all_files):
        print(f"[{idx + 1}/{len(all_files)}] 正在处理: {filename}")

        try:
            # 提取文本
            text = extract_text(file_path)
            if not text or len(text.strip()) < 50:
                print(f"  跳过（文本过短）")
                continue

            # 分块
            chunks = chunk_text(text, chunk_size=500, overlap=50)
            print(f"  分割成 {len(chunks)} 个文本块")

            # 获取元数据
            metadata = get_file_metadata(file_path)
            metadata["source"] = filename
            rel_path = os.path.relpath(root, docs_dir)
            if rel_path == '.':
                metadata["category"] = "医械标准库"
            else:
                metadata["category"] = "医械标准库/" + rel_path.replace(os.path.sep, '/')

            # 预计算embeddings
            embeddings = [get_random_embedding() for _ in chunks]

            # 添加到向量库
            chunk_ids = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"medical_{idx}_{i}_{hash(filename) % 100000}"
                chunk_ids.append(chunk_id)

            vs.add_documents(
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

        except Exception as e:
            stats["failed_files"] += 1
            stats["errors"].append(f"{filename}: {str(e)}")
            print(f"  处理失败: {e}")
            import traceback
            traceback.print_exc()

        # 每10个文件显示一次汇总
        if (idx + 1) % 10 == 0:
            print(f"\n--- 进度: {idx + 1}/{len(all_files)}, 成功: {stats['processed_files']}, 文本块: {stats['total_chunks']} ---\n")

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
        for err in stats['errors'][:20]:
            print(f"  - {err}")

if __name__ == "__main__":
    main()
