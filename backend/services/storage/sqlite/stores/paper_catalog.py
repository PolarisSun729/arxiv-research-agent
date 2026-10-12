import json
from typing import Any, Dict, List, Optional, Sequence

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import logger, sql_placeholders

# 单篇详情/批量详情读取的完整列（含 embedding_model）。
_PAPER_DETAIL_COLUMNS = (
    "arxiv_id", "title", "authors", "abstract", "categories",
    "published_date", "url", "embedding_id", "embedding_model", "created_at",
)
# 列表类接口（全部论文、按分类搜索）历史上不返回 embedding_model，保持响应结构不变。
_PAPER_LIST_COLUMNS = tuple(column for column in _PAPER_DETAIL_COLUMNS if column != "embedding_model")
# 以 JSON 文本存储、读取时需要反序列化为列表的列。
_PAPER_JSON_COLUMNS = {"authors", "categories"}

# SQLite 旧版本最多 999 个绑定参数；批量查询按 500 分批，给其他条件留余量。
_BATCH_QUERY_SIZE = 500


def paper_has_display_metadata(paper: Dict[str, Any]) -> bool:
    """判断本地论文记录是否足以支撑列表卡片展示，缺字段时允许详情接口回源修复。"""
    title = str(paper.get("title") or "").strip()
    abstract = str(paper.get("abstract") or paper.get("summary") or "").strip()
    return bool(title and abstract)


class PaperCatalogStore(BaseSqliteStore):
    """arxiv_papers 表的内部实现；只负责论文目录和 embedding 元数据引用。"""

    def add_paper(self, paper: Dict[str, Any]) -> bool:
        """插入或刷新一篇论文的元数据，返回是否写入成功。

        同一 arxiv_id 重复写入时只覆盖元数据；embedding 引用只有在新值非空时才更新，
        防止“仅刷新元数据”的调用把已经建好的向量索引引用擦掉。
        """
        try:
            normalized_paper = {
                "arxiv_id": self._serialize_paper_db_value(paper.get("arxiv_id")),
                "title": self._serialize_paper_db_value(paper.get("title")),
                "authors": self._serialize_paper_db_value(paper.get("authors")),
                "abstract": self._serialize_paper_db_value(paper.get("abstract")),
                "categories": self._serialize_paper_db_value(paper.get("categories")),
                "published_date": self._serialize_paper_db_value(paper.get("published_date")),
                "url": self._serialize_paper_db_value(paper.get("url")),
                "embedding_id": self._serialize_paper_db_value(paper.get("embedding_id")),
                "embedding_model": self._serialize_paper_db_value(paper.get("embedding_model")),
            }

            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 元数据刷新不应擦掉已建好的向量索引；只有新 embedding_id 有效时才更新向量引用。
                cursor.execute('''
                    INSERT INTO arxiv_papers
                    (arxiv_id, title, authors, abstract, categories, published_date, url, embedding_id, embedding_model, embedded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(arxiv_id) DO UPDATE SET
                        title = excluded.title,
                        authors = excluded.authors,
                        abstract = excluded.abstract,
                        categories = excluded.categories,
                        published_date = excluded.published_date,
                        url = excluded.url,
                        embedding_id = CASE
                            WHEN NULLIF(excluded.embedding_id, '') IS NOT NULL THEN excluded.embedding_id
                            ELSE arxiv_papers.embedding_id
                        END,
                        embedding_model = CASE
                            WHEN NULLIF(excluded.embedding_id, '') IS NOT NULL THEN excluded.embedding_model
                            ELSE arxiv_papers.embedding_model
                        END,
                        embedded_at = CASE
                            WHEN NULLIF(excluded.embedding_id, '') IS NOT NULL THEN excluded.embedded_at
                            ELSE arxiv_papers.embedded_at
                        END
                ''', (
                    normalized_paper["arxiv_id"],
                    normalized_paper["title"],
                    normalized_paper["authors"],
                    normalized_paper["abstract"],
                    normalized_paper["categories"],
                    normalized_paper["published_date"],
                    normalized_paper["url"],
                    normalized_paper["embedding_id"],
                    normalized_paper["embedding_model"],
                ))

                conn.commit()
                logger.info(f"Paper added: {normalized_paper.get('arxiv_id')}")
                return True
        except Exception as e:
            logger.error(f"Error adding paper: {str(e)}")
            return False

    @staticmethod
    def _serialize_paper_db_value(value: Any) -> str:
        """把论文字段统一转成 TEXT：None 写空串，列表/字典写 JSON，其余去掉首尾空白。"""
        if value is None:
            return ""

        if isinstance(value, (list, tuple, set)):
            return json.dumps(list(value), ensure_ascii=False)

        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)

        return str(value).strip()

    # 反序列化使用 JsonFieldCodec._deserialize_paper_db_value（经 BaseSqliteStore 继承），
    # 这里曾有一份逐行相同的副本，已删除以免两处行为分叉。

    def _paper_from_row(self, row: Sequence[Any], columns: Sequence[str] = _PAPER_DETAIL_COLUMNS) -> Dict[str, Any]:
        """按列名把查询行映射成字典；authors/categories 从 JSON 文本还原为列表。"""
        return {
            column: self._deserialize_paper_db_value(value) if column in _PAPER_JSON_COLUMNS else value
            for column, value in zip(columns, row)
        }

    def get_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """读取单篇论文详情；不存在或读库失败时返回 None。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT {', '.join(_PAPER_DETAIL_COLUMNS)} FROM arxiv_papers WHERE arxiv_id = ?",
                    (arxiv_id,),
                )

                row = cursor.fetchone()
                if row:
                    return self._paper_from_row(row)
                return None
        except Exception as e:
            logger.error(f"Error getting paper: {str(e)}")
            return None

    def get_papers_by_ids(self, arxiv_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """批量读取本地论文，返回 arxiv_id → 论文；本地没有的 ID 不出现在结果中，排序由调用方决定。"""
        unique_ids = list(dict.fromkeys(arxiv_id for arxiv_id in arxiv_ids if arxiv_id))
        papers: Dict[str, Dict[str, Any]] = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 分批查询，调用方传入较多 ID 也不会触发 SQLite 绑定参数上限。
            for start in range(0, len(unique_ids), _BATCH_QUERY_SIZE):
                batch = unique_ids[start:start + _BATCH_QUERY_SIZE]
                cursor.execute(
                    f"SELECT {', '.join(_PAPER_DETAIL_COLUMNS)} "
                    f"FROM arxiv_papers WHERE arxiv_id IN ({sql_placeholders(batch)})",
                    batch,
                )
                for row in cursor.fetchall():
                    papers[row[0]] = self._paper_from_row(row)
        return papers

    def search_papers_by_category(self, category: str) -> List[Dict[str, Any]]:
        """按分类做子串匹配（categories 以 JSON 文本存储，LIKE 足以命中单个分类名）。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT {', '.join(_PAPER_LIST_COLUMNS)} FROM arxiv_papers WHERE categories LIKE ?",
                    (f'%{category}%',),
                )
                return [self._paper_from_row(row, _PAPER_LIST_COLUMNS) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error searching papers by category: {str(e)}")
            return []

    def get_all_papers(self) -> List[Dict[str, Any]]:
        """返回本地全部论文，按发表日期倒序。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT {', '.join(_PAPER_LIST_COLUMNS)} FROM arxiv_papers ORDER BY published_date DESC"
                )
                return [self._paper_from_row(row, _PAPER_LIST_COLUMNS) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting all papers: {str(e)}")
            return []

    def get_total_paper_count(self) -> int:
        """返回本地论文总数，读库失败时按 0 处理。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM arxiv_papers')
                row = cursor.fetchone()
                return int(row[0] or 0) if row else 0
        except Exception as e:
            logger.error(f"Error getting total paper count: {str(e)}")
            return 0

    def delete_paper(self, arxiv_id: str) -> bool:
        """删除论文目录记录；只删本地目录，不负责清理向量库中的对应向量。"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM arxiv_papers WHERE arxiv_id = ?', (arxiv_id,))
                conn.commit()
                logger.info(f"Paper deleted: {arxiv_id}")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting paper: {str(e)}")
            return False

    def update_paper_embedding(self, arxiv_id: str, embedding_id: int, embedding_model: Optional[str] = None) -> bool:
        """
        更新论文在向量库中的 embedding 元数据。

        这里只保存向量库引用，不负责写入真实向量；调用方需要先完成向量写入，
        再把 embedding_id 和 embedding_model 回填到本地论文目录表。
        未传 embedding_model 时保留表中原有模型名，只刷新 embedding_id。
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if embedding_model:
                    cursor.execute('''
                        UPDATE arxiv_papers
                        SET embedding_id = ?, embedding_model = ?, embedded_at = CURRENT_TIMESTAMP
                        WHERE arxiv_id = ?
                    ''', (str(embedding_id), embedding_model, arxiv_id))
                else:
                    cursor.execute('''
                        UPDATE arxiv_papers
                        SET embedding_id = ?, embedded_at = CURRENT_TIMESTAMP
                        WHERE arxiv_id = ?
                    ''', (str(embedding_id), arxiv_id))

                conn.commit()
                logger.info(f"Paper {arxiv_id} embedding updated with id: {embedding_id}")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating paper embedding: {str(e)}")
            return False
