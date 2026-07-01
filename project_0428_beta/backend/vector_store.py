"""
医疗器械体系文件审核 - 向量存储模块
使用 ChromaDB 管理知识库向量，支持本地持久化
"""
import os
import sqlite3 as _sqlite3
from typing import List, Optional, Dict, Any
from pathlib import Path
import httpx

# ===== SQLite 内存限制 Monkey-Patch =====
# 拦截所有 sqlite3 连接并在打开后应用内存限制 PRAGMA，
# 避免 ChromaDB 的 2.8GB 数据库将大量数据加载到内存中导致 OOM
_original_connect = _sqlite3.connect

def _patched_connect(database, *args, **kwargs):
    conn = _original_connect(database, *args, **kwargs)
    try:
        conn.execute("PRAGMA cache_size = -4000")       # 缓存限制 4MB
        conn.execute("PRAGMA mmap_size = 0")             # 禁用内存映射
        conn.execute("PRAGMA temp_store = FILE")         # 临时数据写入文件
        conn.execute("PRAGMA synchronous = NORMAL")      # 减少同步开销
    except Exception:
        pass
    return conn

_sqlite3.connect = _patched_connect


class MiniMaxEmbeddingFunction:
    """自定义的 Embedding Function，使用本地 Ollama（OpenAI 兼容接口）"""

    def __init__(self, api_key: str, api_url: str, model: str = "qwen3-embedding:4b", dimension: int = 1024):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.dimension = dimension

    def __call__(self, input: List[str]) -> List[List[float]]:
        """调用 Embedding API（OpenAI 兼容格式）"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        embeddings = []
        with httpx.Client(timeout=60, trust_env=False) as client:
            for text in input:
                payload = {
                    "model": self.model,
                    "input": text[:8000]
                }
                try:
                    response = client.post(self.api_url, headers=headers, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        data = result.get("data", [])
                        if data:
                            embedding = data[0].get("embedding", [])
                            # 如果需要，截断到指定维度
                            if len(embedding) > self.dimension:
                                embedding = embedding[:self.dimension]
                            elif len(embedding) < self.dimension:
                                # 如果太短，用随机数填充
                                import random
                                while len(embedding) < self.dimension:
                                    embedding.append(random.uniform(-1, 1))
                            embeddings.append(embedding)
                        else:
                            # 失败时使用随机向量
                            import random
                            embeddings.append([random.uniform(-1, 1) for _ in range(self.dimension)])
                    else:
                        import random
                        embeddings.append([random.uniform(-1, 1) for _ in range(self.dimension)])
                except Exception:
                    import random
                    embeddings.append([random.uniform(-1, 1) for _ in range(self.dimension)])

        return embeddings


class VectorStore:
    """ChromaDB 向量存储管理类"""

    # 查询时使用的目标 collection 列表（优先级从高到低）
    # 使用 insulin_pump_kb collection 进行检索
    QUERY_COLLECTIONS = ["insulin_pump_kb"]

    def __init__(self, persist_directory: str = "data/chroma_db_insulin_pump", embedding_function=None):
        """
        初始化向量存储

        Args:
            persist_directory: 持久化存储路径
            embedding_function: 自定义 embedding 函数（用于查询时生成 query embedding）
        """
        import chromadb
        from chromadb.config import Settings

        # 获取 backend 目录的绝对路径
        base_dir = Path(__file__).parent
        full_path = base_dir / persist_directory

        self.persist_directory = str(full_path)
        self.collection_name = "insulin_pump_kb"
        self.embedding_function = embedding_function

        # 确保目录存在
        os.makedirs(self.persist_directory, exist_ok=True)

        # 初始化 ChromaDB 客户端（添加内存优化设置）
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # 获取主 collection（不传 embedding_function，避免初始化时加载 HNSW 索引）
        # 主 collection 仅用于写入/计数等操作，查询走 QUERY_COLLECTIONS 列表
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "医疗器械体系文件知识库",
                    "hnsw:space": "cosine",
                    # HNSW 内存优化参数：
                    # M=4 相比默认16降低75%边存储，是 RAG 场景可接受的最低值
                    "hnsw:M": 4,
                    # construction_ef 仅影响构建时的索引质量，降低可减少构建内存峰值
                    "hnsw:construction_ef": 50,
                    # search_ef 影响查询精度，30 在召回率与内存间取得平衡
                    "hnsw:search_ef": 30,
                }
            )

    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        添加文档到向量库（自动分批，避免单次提交过大导致 ChromaDB OOM）

        Args:
            documents: 文档文本列表
            metadatas: 元数据列表
            ids: ID 列表（可选，自动生成）
            embeddings: 预计算的 embedding 列表（可选）

        Returns:
            生成的 ID 列表
        """
        if ids is None:
            ids = [f"doc_{i}_{hash(text) % 100000}" for i, text in enumerate(documents)]

        total = len(documents)
        batch_size = 500  # 每批最多500条，避免 ChromaDB 内存溢出

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            if embeddings is not None:
                self.collection.add(
                    embeddings=embeddings[start:end],
                    metadatas=metadatas[start:end],
                    documents=documents[start:end],
                    ids=ids[start:end]
                )
            else:
                self.collection.add(
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                    ids=ids[start:end]
                )

        return ids

    def _get_query_embeddings(self, query_texts: List[str]) -> Optional[List[List[float]]]:
        """
        使用 embedding_function 将查询文本转为向量

        Args:
            query_texts: 查询文本列表

        Returns:
            embedding 列表，如果 embedding_function 不可用则返回 None
        """
        if self.embedding_function is None:
            return None
        try:
            return self.embedding_function(query_texts)
        except Exception as e:
            print(f"生成 query embedding 失败: {e}")
            return None

    def query(
        self,
        query_texts: Optional[List[str]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        query_embeddings: Optional[List[List[float]]] = None
    ) -> Dict[str, Any]:
        """
        查询相似文档（从 QUERY_COLLECTIONS 列表中的 collection 检索并合并结果）

        从 QUERY_COLLECTIONS 中配置的 collection（默认 insulin_pump_kb）检索。

        Args:
            query_texts: 查询文本列表
            n_results: 返回结果数量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件
            query_embeddings: 预计算的查询 embedding 列表（可选）

        Returns:
            查询结果字典
        """
        # 如果没有预计算的 embedding，使用 embedding_function 生成
        if query_embeddings is None and query_texts is not None:
            query_embeddings = self._get_query_embeddings(query_texts)

        # 限制每次查询返回结果数，避免加载过多 HNSW 索引数据
        safe_n_results = min(n_results, 10)

        # 构建查询参数
        if query_embeddings is not None:
            query_kwargs = {
                "query_embeddings": query_embeddings,
                "n_results": safe_n_results,
                "where": where,
                "where_document": where_document,
            }
        else:
            query_kwargs = {
                "query_texts": query_texts,
                "n_results": safe_n_results,
                "where": where,
                "where_document": where_document,
            }

        # 遍历 QUERY_COLLECTIONS 列表，收集各 collection 的检索结果
        all_items = []

        for coll_name in self.QUERY_COLLECTIONS:
            try:
                coll = self.client.get_collection(coll_name)
                coll_count = coll.count()
                if coll_count == 0:
                    print(f"[VectorStore] 跳过空 collection: {coll_name}")
                    continue
                results = coll.query(**query_kwargs)
                if results and results.get('ids') and results['ids'][0]:
                    for i in range(len(results['ids'][0])):
                        item = {'id': results['ids'][0][i]}
                        if 'documents' in results and results['documents']:
                            item['document'] = results['documents'][0][i]
                        if 'metadatas' in results and results['metadatas']:
                            item['metadata'] = results['metadatas'][0][i]
                        else:
                            item['metadata'] = {}
                        if 'distances' in results and results['distances']:
                            item['distance'] = results['distances'][0][i]
                        else:
                            item['distance'] = float('inf')
                        all_items.append(item)
            except Exception as e:
                print(f"[VectorStore] 查询 collection '{coll_name}' 失败: {e}")
                continue

        # 如果没有结果，返回空
        if not all_items:
            return {
                'ids': [[]],
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }

        # 按距离排序，取 top-n
        all_items.sort(key=lambda x: x['distance'])

        merged = {
            'ids': [[]],
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        for item in all_items[:n_results]:
            merged['ids'][0].append(item['id'])
            merged['documents'][0].append(item.get('document', ''))
            merged['metadatas'][0].append(item.get('metadata', {}))
            merged['distances'][0].append(item.get('distance', float('inf')))

        return merged

    def get_all_documents(self, limit: int = 1000) -> Dict[str, Any]:
        """
        获取所有文档

        Args:
            limit: 限制返回数量

        Returns:
            文档字典
        """
        total = self.collection.count()
        return self.collection.get(limit=min(limit, total))

    def delete_collection(self):
        """删除 collection"""
        self.client.delete_collection(name=self.collection_name)

    def count(self) -> int:
        """返回所有查询 collection 中的文档块总数"""
        total = 0
        for coll_name in self.QUERY_COLLECTIONS:
            try:
                coll = self.client.get_collection(coll_name)
                total += coll.count()
            except Exception:
                continue
        if total == 0:
            return self.collection.count()
        return total


def create_vector_store(persist_directory: str = "data/chroma_db_insulin_pump", embedding_function=None) -> VectorStore:
    """
    工厂函数：创建向量存储实例

    Args:
        persist_directory: 持久化目录
        embedding_function: 自定义 embedding 函数（可选）

    Returns:
        VectorStore 实例
    """
    return VectorStore(persist_directory=persist_directory, embedding_function=embedding_function)
