#!/usr/bin/env python
"""摄入文档目录中所有未摄入的文件，使用真实 Embedding API（低内存版本v3）
核心策略：使用新 collection 写入，避免大索引内存问题；查询时合并两个 collection
"""
import os
import sys
import time
import gc
import httpx
from pathlib import Path
from collections import defaultdict

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from doc_processor import extract_text, chunk_text, get_file_metadata

# Embedding API 配置（本地 Ollama，OpenAI 兼容接口）
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("MINIMAX_API_KEY") or "ollama"
EMBEDDING_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:11435/v1/embeddings")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b")
EMBEDDING_DIM = 1024

# 新增数据的 collection 名称（避免旧索引内存问题）
NEW_COLLECTION_NAME = "medical_device_kb_v2"

# 低内存优化：复用 httpx Client
_http_client = None

def _get_http_client():
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(timeout=60, trust_env=False)
    return _http_client


def get_embedding(text: str) -> list:
    """调用 Embedding API 获取文本向量"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text[:8000]
    }
    try:
        client = _get_http_client()
        response = client.post(EMBEDDING_URL, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            data = result.get("data", [])
            if data:
                embedding = data[0].get("embedding", [])
                if len(embedding) > EMBEDDING_DIM:
                    return embedding[:EMBEDDING_DIM]
                return embedding
    except Exception as e:
        print(f"  [Embedding API 错误] {e}")
    return None


def get_embeddings_batch(texts: list, max_retries: int = 3) -> list:
    """逐个获取 embeddings，失败自动重试"""
    results = [None] * len(texts)
    for attempt in range(max_retries):
        remaining_indices = [i for i, r in enumerate(results) if r is None]
        if not remaining_indices:
            break
        for i in remaining_indices:
            emb = get_embedding(texts[i])
            if emb:
                results[i] = emb
            time.sleep(0.03)
        if attempt < max_retries - 1:
            failed = sum(1 for r in results if r is None)
            if failed > 0:
                print(f"  重试 {attempt+2}/{max_retries}，{failed} 个待重试")
                time.sleep(1)
    return results


def scan_directory(base_dir):
    """扫描目录下所有支持的文件"""
    supported_exts = {'.docx', '.pdf', '.txt', '.doc'}
    all_files = []

    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext not in supported_exts:
                continue
            if filename.startswith('~$'):
                continue

            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, base_dir)
            all_files.append({
                'filename': filename,
                'full_path': full_path,
                'rel_path': rel_path,
                'directory': os.path.relpath(root, base_dir)
            })

    return all_files


def get_existing_sources():
    """从缓存文件加载已有的来源"""
    import json
    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "existing_sources.json")

    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            sources = set(json.load(f))
        print(f"从缓存加载了 {len(sources)} 个已有来源")
        return sources

    # 缓存不存在，从旧 ChromaDB 分页提取
    print("缓存不存在，从 ChromaDB 分页提取已有来源...")
    import chromadb
    from chromadb.config import Settings

    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "data", "chroma_db")
    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection("medical_device_kb")
    total = collection.count()

    sources = set()
    batch_size = 5000
    offset = 0
    while offset < total:
        result = collection.get(limit=batch_size, offset=offset, include=['metadatas'])
        for meta in result.get('metadatas', []):
            if meta:
                source = meta.get('source')
                if source:
                    sources.add(source)
        offset += batch_size
        print(f"  已提取 {min(offset, total)}/{total} 条...")

    # 同时检查新 collection 的已有来源
    try:
        new_coll = client.get_collection(NEW_COLLECTION_NAME)
        new_total = new_coll.count()
        offset2 = 0
        while offset2 < new_total:
            result2 = new_coll.get(limit=batch_size, offset=offset2, include=['metadatas'])
            for meta in result2.get('metadatas', []):
                if meta:
                    source = meta.get('source')
                    if source:
                        sources.add(source)
            offset2 += batch_size
        print(f"  从 {NEW_COLLECTION_NAME} 额外提取了来源")
    except Exception:
        pass

    del client, collection
    gc.collect()

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(sorted(sources), f, ensure_ascii=False)
    print(f"提取了 {len(sources)} 个已有来源，已保存到缓存")

    return sources


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    docs_dir = os.path.join(project_root, "develop_documents")
    db_path = os.path.join(base_dir, "data", "chroma_db")

    print("=" * 80)
    print("文档摄入工具（使用真实 Embedding API - v3 新collection策略）")
    print("=" * 80)
    print(f"文档目录: {docs_dir}")
    print(f"新数据写入 collection: {NEW_COLLECTION_NAME}")
    print(f"旧 collection: medical_device_kb (只读)")

    if not os.path.exists(docs_dir):
        print(f"\n错误: 目录不存在: {docs_dir}")
        return

    # 扫描目录
    print("\n正在扫描文档目录...")
    all_files = scan_directory(docs_dir)
    print(f"找到 {len(all_files)} 个文档文件")

    # 获取已摄入的文件
    print("\n正在检查已摄入的文件...")
    existing_sources = get_existing_sources()
    print(f"向量库中已有 {len(existing_sources)} 个来源文件")

    # 确定需要摄入的文件
    files_to_ingest = []
    for f in all_files:
        if f['filename'] not in existing_sources:
            files_to_ingest.append(f)

    print(f"\n需要摄入的新文件数: {len(files_to_ingest)}")

    if not files_to_ingest:
        print("\n所有文件都已摄入！")
        return

    # 按目录分组统计
    dir_stats = defaultdict(int)
    for f in files_to_ingest:
        dir_stats[f['directory']] += 1
    print(f"\n待摄入文件目录分布（前20）:")
    for d, count in sorted(dir_stats.items(), key=lambda x: -x[1])[:20]:
        print(f"  {d}: {count} 个文件")

    # 创建新 collection
    print("\n正在创建/获取新 collection...")
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(
        name=NEW_COLLECTION_NAME,
        metadata={"description": "医疗器械体系文件知识库 - 新增数据(2026-05)"}
    )
    print(f"新 collection 当前文档数: {collection.count()} 个文本块")

    # 开始摄入
    print("\n开始摄入文档...")
    print("=" * 80)

    stats = {
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'total_chunks': 0,
        'api_errors': 0,
        'errors': []
    }

    start_time = time.time()
    BATCH_COMMIT_SIZE = 1  # 每处理1个文件就立即提交，避免积累过多chunks导致OOM

    # 暂存区
    pending_docs = []
    pending_metas = []
    pending_ids = []
    pending_embs = []

    MAX_CHROMA_BATCH = 1000  # ChromaDB 单次最大批次，从40000降至1000避免内存溢出

    def flush_pending():
        """将暂存区写入新 collection（分批提交避免超过最大批次限制）"""
        nonlocal pending_docs, pending_metas, pending_ids, pending_embs
        if not pending_docs:
            return 0
        total = len(pending_docs)
        written = 0
        for start in range(0, total, MAX_CHROMA_BATCH):
            end = min(start + MAX_CHROMA_BATCH, total)
            collection.add(
                documents=pending_docs[start:end],
                metadatas=pending_metas[start:end],
                ids=pending_ids[start:end],
                embeddings=pending_embs[start:end]
            )
            written += end - start
        pending_docs = []
        pending_metas = []
        pending_ids = []
        pending_embs = []
        gc.collect()
        return written

    for idx, file_info in enumerate(files_to_ingest):
        filename = file_info['filename']
        full_path = file_info['full_path']
        directory = file_info['directory']

        elapsed = time.time() - start_time
        if idx > 0:
            eta = elapsed / idx * (len(files_to_ingest) - idx)
            eta_str = f", 预计剩余 {eta/60:.0f} 分钟"
        else:
            eta_str = ""

        print(f"\n[{idx + 1}/{len(files_to_ingest)}] {filename}{eta_str}")
        print(f"  目录: {directory}")

        try:
            # 提取文本
            text = extract_text(full_path)
            if not text or len(text.strip()) < 20:
                print(f"  跳过: 文本过短或为空")
                stats['skipped'] += 1
                del text
                continue

            # 分块
            chunks = chunk_text(text, chunk_size=500, overlap=50)
            print(f"  分割成 {len(chunks)} 个文本块")

            del text
            gc.collect()

            # 获取元数据
            metadata = get_file_metadata(full_path)
            metadata['source'] = filename
            if directory == '.':
                metadata['category'] = "根目录"
            else:
                metadata['category'] = directory.replace(os.path.sep, '/')

            # 生成 embeddings
            print(f"  正在生成 {len(chunks)} 个 embedding 向量...")
            embeddings = get_embeddings_batch(chunks)

            failed_emb = sum(1 for e in embeddings if e is None)
            if failed_emb > 0:
                stats['api_errors'] += failed_emb
                print(f"  警告: {failed_emb} 个 embedding 生成失败，将使用随机向量替代")
                import random
                embeddings = [e if e is not None else [random.uniform(-1, 1) for _ in range(EMBEDDING_DIM)] for e in embeddings]

            # 暂存
            for i, chunk in enumerate(chunks):
                chunk_id = f"v2_{filename}_{idx}_{i}_{abs(hash(chunk)) % 1000000}"
                pending_docs.append(chunk)
                pending_metas.append(metadata.copy())
                pending_ids.append(chunk_id)
                pending_embs.append(embeddings[i])

            stats['success'] += 1
            stats['total_chunks'] += len(chunks)
            print(f"  成功！已暂存 {len(chunks)} 个文本块")

            del chunks, embeddings
            gc.collect()

        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append(f"{filename}: {str(e)}")
            print(f"  失败: {str(e)}")
            import traceback
            traceback.print_exc()

        # 批量提交
        if stats['success'] > 0 and stats['success'] % BATCH_COMMIT_SIZE == 0 and pending_docs:
            flushed = flush_pending()
            print(f"\n  [批量提交] 写入 {flushed} 个文本块到 {NEW_COLLECTION_NAME}")

            # 输出内存使用
            try:
                import psutil
                proc = psutil.Process(os.getpid())
                mem_mb = proc.memory_info().rss / 1024 / 1024
                print(f"  [内存] 当前进程 RSS: {mem_mb:.0f} MB")
            except ImportError:
                pass

        # 进度汇总
        if (idx + 1) % 50 == 0:
            print(f"\n--- 进度: {idx + 1}/{len(files_to_ingest)}, "
                  f"成功: {stats['success']}, 失败: {stats['failed']}, "
                  f"跳过: {stats['skipped']}, 文本块: {stats['total_chunks']} ---")

        time.sleep(0.02)

    # 提交剩余数据
    if pending_docs:
        flushed = flush_pending()
        print(f"\n[最终提交] 写入 {flushed} 个文本块到 {NEW_COLLECTION_NAME}")

    # 完成总结
    total_time = time.time() - start_time
    final_count = collection.count()
    print("\n" + "=" * 80)
    print("摄入完成！")
    print("=" * 80)
    print(f"尝试处理: {len(files_to_ingest)} 个文件")
    print(f"成功: {stats['success']} 个文件")
    print(f"失败: {stats['failed']} 个文件")
    print(f"跳过: {stats['skipped']} 个文件")
    print(f"新增文本块: {stats['total_chunks']}")
    print(f"API 错误: {stats['api_errors']} 个 embedding")
    print(f"新 collection 文档数: {final_count}")
    print(f"耗时: {total_time/60:.1f} 分钟")

    if stats['errors']:
        print(f"\n错误列表（前30个）:")
        for err in stats['errors'][:30]:
            print(f"  - {err}")

    # 关闭 httpx 客户端
    try:
        if _http_client is not None:
            _http_client.close()
    except Exception:
        pass

    # 释放 collection
    del client, collection
    gc.collect()

    # 更新缓存
    print("\n正在更新 existing_sources.json 缓存...")
    try:
        all_new_sources = set(existing_sources)
        for f in files_to_ingest:
            all_new_sources.add(f['filename'])
        import json
        cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "existing_sources.json")
        with open(cache_file, 'w', encoding='utf-8') as fp:
            json.dump(sorted(all_new_sources), fp, ensure_ascii=False)
        print(f"缓存已更新，共 {len(all_new_sources)} 个来源文件")
    except Exception as e:
        print(f"缓存更新失败: {e}")

    # 提示更新查询代码
    print(f"\n" + "!" * 80)
    print(f"重要: 新数据已写入 collection '{NEW_COLLECTION_NAME}'")
    print(f"请在查询代码中同时查询 'medical_device_kb' 和 '{NEW_COLLECTION_NAME}' 两个 collection")
    print(f"!" * 80)


if __name__ == "__main__":
    main()
