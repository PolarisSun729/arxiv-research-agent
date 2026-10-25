"""论文级向量（arxiv_paper_embeddings）的文件存储：向量文件 + SQLite 元数据，检索时逐一比对。

为什么不用 Milvus Lite:
    Milvus Lite 在 Web 进程内运行，会把数据文件整份读进进程内存，同一份数据还不止存一份；
    7 万篇 2048 维论文向量常驻约 1.2G，再加上 Docling 解析的峰值就超出服务的内存上限。
    论文向量用的本来就是 FLAT 索引（逐一比对），换成自己逐一算余弦相似度，结果一样。

存储格式:
    - 向量文件：每行一篇论文（一个“槽位”），dim 个小端 float32，没有文件头；容量按 _GROW_ROWS 行扩展。
      检索时用内存映射逐块读，读到的页只进页缓存，内存紧张时内核直接丢弃，不计入进程自己的内存。
    - SQLite：每篇论文一行，记录槽位、arxiv_id、向量模长和展示字段；store_meta 记录维度。
      只有 SQLite 里有记录的槽位才有效；删除只删记录，空出的槽位留给后面的新论文。

一致性:
    写入先把向量写进文件并刷盘，再提交 SQLite 记录，最后换上新的内存快照；中途崩溃只会留下没有记录的行，
    重启后它仍是空闲槽位。读取不加锁：开始时取一份快照，写入方只往快照里的空闲槽位写，所以读不到写了一半的数据。
    只允许一个进程打开（文件锁，Linux），和 Milvus Lite 的独占锁一样：写入去重靠的是进程内的锁和内存索引。

对外方法与 VectorStoreService 的论文级方法同名同参，依赖注入层按 PAPER_VECTOR_BACKEND 二选一，调用方不用改。
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from services.storage.vector_store_service import build_paper_embedding_payload, build_paper_payload

logger = logging.getLogger(__name__)

PAPER_EMBEDDING_COLLECTION = "arxiv_paper_embeddings"

_DTYPE = np.dtype("<f4")
# 向量文件每次扩展的行数（2048 维约 64MB），避免每写一批就改一次文件大小。
_GROW_ROWS = 8192
# 检索时每次读入并计算的行数（2048 维约 128MB 的页缓存），控制单次触碰的内存。
_SEARCH_BLOCK_ROWS = 16384
# SQLite 单条语句的参数个数有上限，IN 查询按这个大小分批。
_SQL_BATCH = 500

_METADATA_COLUMNS = ("published_date", "embedding_model", "title", "authors", "categories", "url", "content")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slot INTEGER NOT NULL UNIQUE,
    arxiv_id TEXT NOT NULL UNIQUE,
    norm REAL NOT NULL,
    published_date TEXT NOT NULL DEFAULT '',
    embedding_model TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    authors TEXT NOT NULL DEFAULT '',
    categories TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_paper_vectors_published_date ON paper_vectors(published_date);
CREATE TABLE IF NOT EXISTS store_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class _Snapshot:
    """某一时刻的有效槽位视图；写入方整体替换，读取方拿到后不会再变。"""

    capacity: int
    slot_by_arxiv_id: Dict[str, int]
    arxiv_id_by_slot: Dict[int, str]
    # 长度都是 capacity；空闲槽位的 valid 为 False、norm 为 0。
    valid: np.ndarray
    norms: np.ndarray


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _normalize_ids(arxiv_ids: Iterable[Any]) -> List[str]:
    """去掉空白和空值，保持原顺序去重。"""
    seen: Dict[str, None] = {}
    for arxiv_id in arxiv_ids:
        normalized = str(arxiv_id or "").strip()
        if normalized:
            seen.setdefault(normalized, None)
    return list(seen)


class FilePaperVectorStore:
    """论文级向量的文件存储，只存一个 collection（默认 arxiv_paper_embeddings）。"""

    def __init__(self, vectors_path: str, db_path: str, collection_name: str = PAPER_EMBEDDING_COLLECTION):
        self.vectors_path = str(vectors_path)
        self.db_path = str(db_path)
        self.collection_name = collection_name
        self._write_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._snapshot: Optional[_Snapshot] = None
        self._dim: Optional[int] = None
        self._process_lock_handle: Any = None

    # ---- 对外方法：与 VectorStoreService 的论文级方法同名同参 ----

    def insert_embeddings(self, collection_name: str, items: List[Dict[str, Any]]) -> int:
        """写入向量库里还没有的论文向量，返回实际写入条数；已有的 arxiv_id（含同一批内重复的）跳过。

        items 的格式与 VectorStoreService.insert_embeddings 相同：{"embedding": [...], "metadata": {...}}；
        迁移时可带顶层 "id" 沿用旧主键。
        """
        self._check_collection(collection_name)
        return len(self._insert(items))

    def insert_single_embedding(self, collection_name: str, embedding: List[float], metadata: Dict[str, Any]) -> int:
        """写入单篇论文向量，返回它的主键 id（已存在时返回已有记录的 id）。"""
        self._check_collection(collection_name)
        arxiv_id = str(metadata.get("arxiv_id", "") or "").strip()
        inserted = self._insert([{"embedding": embedding, "metadata": metadata}])
        if inserted:
            return inserted[0]
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT id FROM paper_vectors WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
        return int(row[0]) if row else 0

    def search_similar_papers(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_arxiv_ids: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """按余弦相似度检索最相近的论文，返回结构与 Milvus 论文检索相同（score 与 distance 都是余弦相似度）。"""
        self._check_collection(collection_name)
        snapshot = self._ensure_loaded()
        if not snapshot.slot_by_arxiv_id or top_k <= 0:
            return []
        query = self._as_query(query_vector)
        query_norm = float(np.linalg.norm(query))

        dots = np.zeros(snapshot.capacity, dtype=np.float32)
        matrix = self._open_matrix(snapshot.capacity)
        try:
            for start in range(0, snapshot.capacity, _SEARCH_BLOCK_ROWS):
                dots[start : start + _SEARCH_BLOCK_ROWS] = matrix[start : start + _SEARCH_BLOCK_ROWS] @ query
        finally:
            del matrix

        denominators = snapshot.norms * np.float32(query_norm)
        scores = np.zeros_like(dots)
        np.divide(dots, denominators, out=scores, where=denominators > 0)
        scores[~snapshot.valid] = -np.inf
        for arxiv_id in _normalize_ids(filter_arxiv_ids or []):
            slot = snapshot.slot_by_arxiv_id.get(arxiv_id)
            if slot is not None:
                scores[slot] = -np.inf

        candidate_count = int(np.isfinite(scores).sum())
        k = min(int(top_k), candidate_count)
        if k <= 0:
            return []
        top_slots = np.argpartition(-scores, k - 1)[:k]
        # 分数相同按槽位排，结果稳定。
        top_slots = top_slots[np.lexsort((top_slots, -scores[top_slots]))]

        ranked_ids = [snapshot.arxiv_id_by_slot[int(slot)] for slot in top_slots]
        rows = self._fetch_metadata(ranked_ids)
        results = []
        for slot, arxiv_id in zip(top_slots, ranked_ids):
            row = rows.get(arxiv_id)
            # 快照之后被删除的论文查不到记录，直接跳过。
            if row is None:
                continue
            score = float(scores[slot])
            results.append(build_paper_payload(row, score=score, distance=score))
        return results

    def get_paper_embeddings_by_arxiv_ids(self, collection_name: str, arxiv_ids: List[str]) -> List[Dict[str, Any]]:
        """按 arxiv_id 批量读取论文向量和元数据（按传入顺序），没有向量的论文不出现在结果里。"""
        self._check_collection(collection_name)
        snapshot = self._ensure_loaded()
        wanted = [arxiv_id for arxiv_id in _normalize_ids(arxiv_ids) if arxiv_id in snapshot.slot_by_arxiv_id]
        if not wanted:
            return []
        rows = self._fetch_metadata(wanted)
        matrix = self._open_matrix(snapshot.capacity)
        try:
            results = []
            for arxiv_id in wanted:
                row = rows.get(arxiv_id)
                if row is None:
                    continue
                vector = np.array(matrix[snapshot.slot_by_arxiv_id[arxiv_id]], dtype=np.float32)
                results.append(build_paper_embedding_payload({**row, "vector": vector.tolist()}))
            return results
        finally:
            del matrix

    def get_existing_paper_arxiv_ids(self, collection_name: str, arxiv_ids: List[str]) -> Set[str]:
        """返回 arxiv_ids 中已有向量的那部分。"""
        self._check_collection(collection_name)
        snapshot = self._ensure_loaded()
        return {arxiv_id for arxiv_id in _normalize_ids(arxiv_ids) if arxiv_id in snapshot.slot_by_arxiv_id}

    def list_paper_arxiv_ids(self, collection_name: str) -> Set[str]:
        """返回所有已有向量的 arxiv_id。"""
        self._check_collection(collection_name)
        return set(self._ensure_loaded().slot_by_arxiv_id)

    def delete_paper_embeddings_published_before(
        self,
        collection_name: str,
        cutoff_date: str,
        keep_arxiv_ids: Set[str],
        dry_run: bool = False,
    ) -> int:
        """删除 published_date 早于 cutoff_date 的论文向量，返回删除（dry_run 时为将删除）的条数。

        规则与 Milvus 实现一致：按字符串比较，published_date 为空的保留，keep_arxiv_ids 里的无论多旧都保留。
        """
        self._check_collection(collection_name)
        with self._write_lock:
            snapshot = self._ensure_loaded()
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    "SELECT arxiv_id FROM paper_vectors WHERE published_date < ? AND published_date != ''",
                    (str(cutoff_date),),
                ).fetchall()
                doomed = [row[0] for row in rows if row[0] not in keep_arxiv_ids]
                if dry_run or not doomed:
                    return len(doomed)
                with connection:
                    for batch in _chunks(doomed, _SQL_BATCH):
                        placeholders = ",".join("?" * len(batch))
                        connection.execute(f"DELETE FROM paper_vectors WHERE arxiv_id IN ({placeholders})", list(batch))
            self._snapshot = self._without(snapshot, doomed)
            return len(doomed)

    def close(self) -> None:
        """释放进程文件锁；之后再调用会重新加载。"""
        with self._load_lock:
            self._snapshot = None
            if self._process_lock_handle is not None:
                self._process_lock_handle.close()
                self._process_lock_handle = None

    # ---- 内部实现 ----

    def _check_collection(self, collection_name: str) -> None:
        if collection_name != self.collection_name:
            raise ValueError(f"FilePaperVectorStore only stores '{self.collection_name}', got '{collection_name}'")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    @property
    def _row_bytes(self) -> int:
        return int(self._dim or 0) * _DTYPE.itemsize

    def _open_matrix(self, capacity: int) -> np.memmap:
        return np.memmap(self.vectors_path, dtype=_DTYPE, mode="r", shape=(capacity, int(self._dim)))

    def _as_query(self, query_vector: List[float]) -> np.ndarray:
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if self._dim is not None and query.shape[0] != self._dim:
            raise ValueError(f"Query vector dimension {query.shape[0]} does not match stored dimension {self._dim}")
        return query

    def _ensure_loaded(self) -> _Snapshot:
        snapshot = self._snapshot
        if snapshot is not None:
            return snapshot
        with self._load_lock:
            if self._snapshot is None:
                self._snapshot = self._load()
            return self._snapshot

    def _load(self) -> _Snapshot:
        for path in (self.vectors_path, self.db_path):
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
        self._acquire_process_lock()
        with closing(self._connect()) as connection:
            with connection:
                connection.executescript(_SCHEMA)
            dim_row = connection.execute("SELECT value FROM store_meta WHERE key = 'dim'").fetchone()
            rows = connection.execute("SELECT slot, arxiv_id, norm FROM paper_vectors").fetchall()
        self._dim = int(dim_row[0]) if dim_row else None

        capacity = 0
        if self._dim and os.path.exists(self.vectors_path):
            size = os.path.getsize(self.vectors_path)
            if size % self._row_bytes:
                raise RuntimeError(f"Paper vector file size {size} is not a multiple of row size {self._row_bytes}: {self.vectors_path}")
            capacity = size // self._row_bytes
        valid = np.zeros(capacity, dtype=bool)
        norms = np.zeros(capacity, dtype=np.float32)
        slot_by_arxiv_id: Dict[str, int] = {}
        for slot, arxiv_id, norm in rows:
            if slot >= capacity:
                raise RuntimeError(f"Paper vector record {arxiv_id} points at slot {slot} beyond file capacity {capacity}")
            valid[slot] = True
            norms[slot] = norm
            slot_by_arxiv_id[arxiv_id] = slot
        logger.info("Loaded paper vector store: %s vectors, capacity %s, dim %s", len(rows), capacity, self._dim)
        return _Snapshot(
            capacity=capacity,
            slot_by_arxiv_id=slot_by_arxiv_id,
            arxiv_id_by_slot={slot: arxiv_id for arxiv_id, slot in slot_by_arxiv_id.items()},
            valid=valid,
            norms=norms,
        )

    def _acquire_process_lock(self) -> None:
        """Linux 上用 flock 保证只有一个进程打开；其他平台（本地开发）没有 fcntl，跳过。"""
        if self._process_lock_handle is not None:
            return
        try:
            import fcntl
        except ImportError:
            return
        handle = open(f"{self.vectors_path}.lock", "a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(f"Paper vector store is already opened by another process: {self.vectors_path}") from exc
        self._process_lock_handle = handle

    def _insert(self, items: List[Dict[str, Any]]) -> List[int]:
        """写入新论文向量，返回新记录的 id（顺序与写入顺序一致）。"""
        if not items:
            return []
        vectors = [np.asarray(item.get("embedding", []), dtype=np.float32).reshape(-1) for item in items]
        if not vectors[0].size:
            raise ValueError("Cannot insert empty embeddings")

        with self._write_lock:
            snapshot = self._ensure_loaded()
            dim = int(vectors[0].shape[0])
            if self._dim is not None and dim != self._dim:
                raise ValueError(
                    f"Collection '{self.collection_name}' already uses vector dimension {self._dim}, "
                    f"but the new embeddings have dimension {dim}. "
                    "Rebuild the collection or keep the embedding model/dimension consistent."
                )
            if any(vector.shape[0] != dim for vector in vectors):
                raise ValueError("Embeddings in one batch have different dimensions")

            fresh = []
            batch_ids: Set[str] = set()
            for item, vector in zip(items, vectors):
                metadata = item.get("metadata", {}) or {}
                arxiv_id = str(metadata.get("arxiv_id", "") or "").strip()
                if not arxiv_id:
                    raise ValueError("Paper embedding has no arxiv_id")
                if arxiv_id in snapshot.slot_by_arxiv_id or arxiv_id in batch_ids:
                    continue
                batch_ids.add(arxiv_id)
                fresh.append((item, metadata, arxiv_id, vector))
            if not fresh:
                return []

            if self._dim is None:
                self._dim = dim
                with closing(self._connect()) as connection, connection:
                    connection.execute("INSERT OR REPLACE INTO store_meta (key, value) VALUES ('dim', ?)", (str(dim),))

            free_slots = np.flatnonzero(~snapshot.valid)[: len(fresh)].tolist()
            next_slot = snapshot.capacity
            while len(free_slots) < len(fresh):
                free_slots.append(next_slot)
                next_slot += 1
            capacity = snapshot.capacity
            if next_slot > capacity:
                capacity = -(-next_slot // _GROW_ROWS) * _GROW_ROWS

            # 先写向量并刷盘，再提交记录：中途崩溃只会留下没有记录的行，重启后仍是空闲槽位。
            with open(self.vectors_path, "r+b" if os.path.exists(self.vectors_path) else "w+b") as handle:
                if capacity > snapshot.capacity:
                    handle.truncate(capacity * self._row_bytes)
                for slot, (_item, _metadata, _arxiv_id, vector) in zip(free_slots, fresh):
                    handle.seek(slot * self._row_bytes)
                    handle.write(vector.astype(_DTYPE, copy=False).tobytes())
                handle.flush()
                os.fsync(handle.fileno())

            norms = [float(np.linalg.norm(vector)) for _item, _metadata, _arxiv_id, vector in fresh]
            new_ids: List[int] = []
            with closing(self._connect()) as connection, connection:
                for slot, norm, (item, metadata, arxiv_id, _vector) in zip(free_slots, norms, fresh):
                    values = [str(metadata.get(column, "") or "") for column in _METADATA_COLUMNS]
                    cursor = connection.execute(
                        f"INSERT INTO paper_vectors (id, slot, arxiv_id, norm, {', '.join(_METADATA_COLUMNS)}) "
                        f"VALUES (?, ?, ?, ?, {', '.join('?' * len(_METADATA_COLUMNS))})",
                        [item.get("id"), slot, arxiv_id, norm, *values],
                    )
                    new_ids.append(int(cursor.lastrowid))

            self._snapshot = self._with(snapshot, capacity, [(arxiv_id, slot, norm) for slot, norm, (_i, _m, arxiv_id, _v) in zip(free_slots, norms, fresh)])
            return new_ids

    def _fetch_metadata(self, arxiv_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        columns = ("id", "arxiv_id", *_METADATA_COLUMNS)
        rows: Dict[str, Dict[str, Any]] = {}
        with closing(self._connect()) as connection:
            for batch in _chunks(list(arxiv_ids), _SQL_BATCH):
                placeholders = ",".join("?" * len(batch))
                for values in connection.execute(
                    f"SELECT {', '.join(columns)} FROM paper_vectors WHERE arxiv_id IN ({placeholders})",
                    list(batch),
                ):
                    row = dict(zip(columns, values))
                    rows[row["arxiv_id"]] = row
        return rows

    @staticmethod
    def _with(snapshot: _Snapshot, capacity: int, added: List[tuple]) -> _Snapshot:
        valid = np.zeros(capacity, dtype=bool)
        norms = np.zeros(capacity, dtype=np.float32)
        valid[: snapshot.capacity] = snapshot.valid
        norms[: snapshot.capacity] = snapshot.norms
        slot_by_arxiv_id = dict(snapshot.slot_by_arxiv_id)
        arxiv_id_by_slot = dict(snapshot.arxiv_id_by_slot)
        for arxiv_id, slot, norm in added:
            valid[slot] = True
            norms[slot] = norm
            slot_by_arxiv_id[arxiv_id] = slot
            arxiv_id_by_slot[slot] = arxiv_id
        return _Snapshot(capacity, slot_by_arxiv_id, arxiv_id_by_slot, valid, norms)

    @staticmethod
    def _without(snapshot: _Snapshot, removed: List[str]) -> _Snapshot:
        valid = snapshot.valid.copy()
        norms = snapshot.norms.copy()
        slot_by_arxiv_id = dict(snapshot.slot_by_arxiv_id)
        arxiv_id_by_slot = dict(snapshot.arxiv_id_by_slot)
        for arxiv_id in removed:
            slot = slot_by_arxiv_id.pop(arxiv_id, None)
            if slot is None:
                continue
            arxiv_id_by_slot.pop(slot, None)
            valid[slot] = False
            norms[slot] = 0.0
        return _Snapshot(snapshot.capacity, slot_by_arxiv_id, arxiv_id_by_slot, valid, norms)
