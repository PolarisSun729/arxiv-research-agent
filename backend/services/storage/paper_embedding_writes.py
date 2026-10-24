"""论文级向量（arxiv_paper_embeddings）的去重写入。

论文向量 collection 的主键是自增 id，不是 arxiv_id：两处代码同时给同一篇论文补向量时，
各自“查缺失 → 写入”会写出两条向量。Web 进程里有两个写入方（推荐候选物化、论文向量维护任务），
这里用一把进程内锁把“再查一次已有 → 写入”合成一步；embedding 调用在锁外，锁只覆盖向量库读写。
Milvus Lite 只允许一个进程打开，所以进程内锁就足够，不需要跨进程互斥。
"""

import threading
from typing import Any, Dict, List

_PAPER_EMBEDDING_WRITE_LOCK = threading.Lock()


def insert_missing_paper_embeddings(vector_store_service: Any, collection_name: str, items: List[Dict[str, Any]]) -> int:
    """只写入向量库里还没有的论文向量（同一批内重复的 arxiv_id 只写第一条），返回实际写入条数。

    items 的格式与 insert_embeddings 相同：{"embedding": [...], "metadata": {"arxiv_id": ..., ...}}。
    """
    with _PAPER_EMBEDDING_WRITE_LOCK:
        arxiv_ids = [str(item.get("metadata", {}).get("arxiv_id", "") or "").strip() for item in items]
        existing_ids = vector_store_service.get_existing_paper_arxiv_ids(collection_name, [arxiv_id for arxiv_id in arxiv_ids if arxiv_id])
        fresh_items = []
        for item, arxiv_id in zip(items, arxiv_ids):
            if arxiv_id and arxiv_id in existing_ids:
                continue
            if arxiv_id:
                existing_ids.add(arxiv_id)
            fresh_items.append(item)
        if not fresh_items:
            return 0
        return int(vector_store_service.insert_embeddings(collection_name, fresh_items))
