"""把 Milvus Lite 里的论文向量（arxiv_paper_embeddings）导出成文件存储（PAPER_VECTOR_BACKEND=file 用的那份）。

为什么需要:
    论文向量从 Milvus Lite 移到「向量文件 + SQLite」后，已经生成的约 7 万条向量原样搬过去，不重新调用 embedding。

怎么运行:
    Milvus Lite 只允许一个进程打开，Web 服务运行时先把数据目录拷一份再导出，不用停服务：
        cp -a <shared>/backend/06-database/milvus_lite.db /tmp/milvus_export.db && rm -f /tmp/milvus_export.db/LOCK
        python 07-arxiv-tools/export_paper_vectors.py --milvus-db /tmp/milvus_export.db
    默认写到 PAPER_VECTOR_FILE_PATH / PAPER_VECTOR_DB_PATH（06-database/paper_vectors.f32 / .db）。
    目标文件已存在时直接退出，不覆盖。先写到 .tmp 文件，校验全部通过后才改成正式文件名。
    拷贝之后维护任务新写入 Milvus 的论文，切换到文件存储后由维护任务下一轮自动补上。

校验:
    1. 每批写入后从文件存储读回，向量逐位相等、元数据一致；
    2. 条数与 arxiv_id 集合和 Milvus 一致（Milvus 里同一篇的重复向量只保留第一条，空 arxiv_id 的行跳过，都会报出来）；
    3. 随机抽 --verify-queries 篇论文的向量作查询，两边 top --verify-top-k 的分数逐名次相差 < 1e-5，
       arxiv_id 集合相同（第 k 名分数并列时允许边界上的论文不同）。

内存:
    Milvus Lite 打开时会把数据整份读进内存（7 万条约 1.2–1.5G），和 Web 进程叠加。
    脚本打开前关掉 Milvus Lite 的后台合并（只读用不着，合并峰值是数据的数倍）；
    并把自己的 oom_score_adj 调到最大，万一内存不够先被杀的是它，而不是 Web 服务。

退出码: 0 = 完成；1 = 参数或环境问题（目标已存在、collection 不存在等）；2 = 校验不通过（.tmp 文件保留供排查）。
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.storage.paper_vector_store import PAPER_EMBEDDING_COLLECTION, FilePaperVectorStore  # noqa: E402
from utils.config import PAPER_VECTOR_CONFIG  # noqa: E402

logger = logging.getLogger("export_paper_vectors")

_FIELDS = ["id", "arxiv_id", "title", "authors", "categories", "published_date", "url", "embedding_model", "content", "vector"]
_SCORE_TOLERANCE = 1e-5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export paper embeddings from Milvus Lite into the file-backed paper vector store.")
    parser.add_argument("--milvus-db", required=True, help="Milvus Lite 数据目录（建议用拷贝出来的副本）")
    parser.add_argument("--collection", default=PAPER_EMBEDDING_COLLECTION)
    parser.add_argument("--vectors", default=PAPER_VECTOR_CONFIG["vectors_path"], help="输出的向量文件")
    parser.add_argument("--db", default=PAPER_VECTOR_CONFIG["db_path"], help="输出的 SQLite 元数据库")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--verify-queries", type=int, default=20)
    parser.add_argument("--verify-top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261025)
    return parser


def _prefer_as_oom_victim() -> None:
    try:
        with open("/proc/self/oom_score_adj", "w", encoding="ascii") as handle:
            handle.write("1000")
    except OSError:
        pass


def _disable_milvus_lite_compaction() -> bool:
    """导出只读不写，关掉 Milvus Lite 打开后的后台合并，返回是否生效。

    副本里的数据文件可能正好满足合并条件，打开后后台会把十几个文件整份读进内存合并成一个，
    峰值是数据的数倍（生产试跑时 7 万行直接被 OOM 杀掉）。只读场景合并没有任何用处。
    """
    try:
        from milvus_lite.engine.compaction import CompactionManager
    except ImportError:
        return False
    if not hasattr(CompactionManager, "maybe_compact"):
        return False
    CompactionManager.maybe_compact = lambda self, *args, **kwargs: False
    return True


def _peak_rss_mb() -> float:
    import resource  # 只在 Linux 上有；脚本只在生产服务器运行

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def _vector(row: Dict[str, Any]) -> np.ndarray:
    value = row.get("vector") or []
    # 部分 pymilvus 版本会把向量包成二维 [[...]]。
    if len(value) and isinstance(value[0], (list, tuple)):
        value = value[0]
    return np.asarray(value, dtype=np.float32)


def export_rows(client: Any, collection: str, store: FilePaperVectorStore, batch_size: int, query_count: int, seed: int) -> Dict[str, Any]:
    """逐批从 Milvus 读出写入文件存储，并读回逐条比对；返回统计和抽样的查询向量。"""
    rng = random.Random(seed)
    seen: set = set()
    duplicates = empty_ids = mismatches = 0
    sampled: List[np.ndarray] = []
    total = 0
    iterator = client.query_iterator(collection_name=collection, batch_size=batch_size, filter="id >= 0", output_fields=_FIELDS)
    try:
        while True:
            rows = iterator.next()
            if not rows:
                break
            items = []
            for row in rows:
                arxiv_id = str(row.get("arxiv_id") or "").strip()
                if not arxiv_id:
                    empty_ids += 1
                    continue
                if arxiv_id in seen:
                    duplicates += 1
                    continue
                seen.add(arxiv_id)
                vector = _vector(row)
                items.append({"id": int(row["id"]), "embedding": vector, "metadata": {**row, "arxiv_id": arxiv_id}})
                # 蓄水池抽样，均匀抽出查询向量。
                total += 1
                if len(sampled) < query_count:
                    sampled.append(vector)
                else:
                    pick = rng.randrange(total)
                    if pick < query_count:
                        sampled[pick] = vector
            store.insert_embeddings(collection, items)
            readback = {row["arxiv_id"]: row for row in store.get_paper_embeddings_by_arxiv_ids(collection, [item["metadata"]["arxiv_id"] for item in items])}
            for item in items:
                metadata = item["metadata"]
                stored = readback.get(metadata["arxiv_id"])
                if (
                    stored is None
                    or stored["embedding_id"] != item["id"]
                    or not np.array_equal(np.asarray(stored["vector"], dtype=np.float32), item["embedding"])
                    or stored["published_date"] != str(metadata.get("published_date") or "")
                    or stored["title"] != str(metadata.get("title") or "")
                ):
                    mismatches += 1
            logger.info("exported %s papers", len(seen))
    finally:
        iterator.close()
    return {"exported": len(seen), "duplicates": duplicates, "empty_ids": empty_ids, "readback_mismatches": mismatches, "queries": sampled}


def compare_search(client: Any, collection: str, store: FilePaperVectorStore, queries: List[np.ndarray], top_k: int) -> int:
    """用同一批查询向量对比两边的检索结果，返回不一致的查询数。"""
    failed = 0
    for index, query in enumerate(queries):
        hits = client.search(
            collection_name=collection,
            data=[query.tolist()],
            limit=top_k,
            anns_field="vector",
            output_fields=["arxiv_id"],
            search_params={"metric_type": "COSINE", "params": {}},
        )[0]
        milvus = [(str(hit["entity"]["arxiv_id"] if "entity" in hit else hit["arxiv_id"]), float(hit["distance"])) for hit in hits]
        mine = [(paper["arxiv_id"], float(paper["score"])) for paper in store.search_similar_papers(collection, query.tolist(), top_k=top_k)]
        score_gap = max((abs(a[1] - b[1]) for a, b in zip(milvus, mine)), default=0.0)
        same_length = len(milvus) == len(mine)
        boundary = milvus[-1][1] if milvus else 0.0
        # 第 k 名分数并列时，边界上取哪一篇两边可以不同，只比较严格高于边界分数的部分。
        strict_m = {arxiv_id for arxiv_id, score in milvus if score > boundary + _SCORE_TOLERANCE}
        strict_f = {arxiv_id for arxiv_id, score in mine if score > boundary + _SCORE_TOLERANCE}
        ok = same_length and score_gap < _SCORE_TOLERANCE and strict_m == strict_f
        if not ok:
            failed += 1
            logger.error("search mismatch query=%s same_length=%s score_gap=%.2e only_milvus=%s only_file=%s", index, same_length, score_gap, sorted(strict_m - strict_f)[:5], sorted(strict_f - strict_m)[:5])
        else:
            logger.info("search match query=%s top=%s score_gap=%.2e", index, len(mine), score_gap)
    return failed


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for target in (args.vectors, args.db):
        if os.path.exists(target):
            logger.error("Target already exists, refusing to overwrite: %s", target)
            return 1
    tmp_vectors, tmp_db = f"{args.vectors}.tmp", f"{args.db}.tmp"
    for leftover in (tmp_vectors, tmp_db, f"{tmp_vectors}.lock"):
        if os.path.exists(leftover):
            os.remove(leftover)

    _prefer_as_oom_victim()
    if not _disable_milvus_lite_compaction():
        logger.error("Cannot disable Milvus Lite compaction (milvus_lite layout changed?); refusing to open %s", args.milvus_db)
        return 1
    from pymilvus import MilvusClient

    client = MilvusClient(uri=args.milvus_db)
    if not client.has_collection(collection_name=args.collection):
        logger.error("Collection %s not found in %s", args.collection, args.milvus_db)
        return 1
    client.load_collection(collection_name=args.collection)

    store = FilePaperVectorStore(vectors_path=tmp_vectors, db_path=tmp_db, collection_name=args.collection)
    try:
        stats = export_rows(client, args.collection, store, args.batch_size, args.verify_queries, args.seed)
        milvus_ids = set()
        iterator = client.query_iterator(collection_name=args.collection, batch_size=5000, filter='arxiv_id != ""', output_fields=["arxiv_id"])
        try:
            while True:
                rows = iterator.next()
                if not rows:
                    break
                milvus_ids.update(str(row["arxiv_id"]).strip() for row in rows)
        finally:
            iterator.close()
        stored_ids = store.list_paper_arxiv_ids(args.collection)
        failed_queries = compare_search(client, args.collection, store, stats["queries"], args.verify_top_k)
    finally:
        store.close()
        client.close()

    ids_match = milvus_ids == stored_ids
    logger.info(
        "paper_vector_export exported=%s milvus_unique_ids=%s duplicates_skipped=%s empty_ids_skipped=%s readback_mismatches=%s ids_match=%s failed_queries=%s/%s",
        stats["exported"],
        len(milvus_ids),
        stats["duplicates"],
        stats["empty_ids"],
        stats["readback_mismatches"],
        ids_match,
        failed_queries,
        len(stats["queries"]),
    )
    if not ids_match or stats["readback_mismatches"] or failed_queries:
        logger.error("Verification failed; keeping %s and %s for inspection", tmp_vectors, tmp_db)
        return 2
    os.replace(tmp_vectors, args.vectors)
    os.replace(tmp_db, args.db)
    if os.path.exists(f"{tmp_vectors}.lock"):
        os.remove(f"{tmp_vectors}.lock")
    logger.info("paper_vector_export done vectors=%s db=%s peak_rss_mb=%s", args.vectors, args.db, _peak_rss_mb())
    return 0


if __name__ == "__main__":
    sys.exit(main())
