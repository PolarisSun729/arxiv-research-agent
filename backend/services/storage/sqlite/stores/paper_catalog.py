import json
from typing import Any, Dict, List, Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import logger


class PaperCatalogStore(BaseSqliteStore):
    """arxiv_papers 表的内部实现；只负责论文目录和 embedding 元数据引用。"""

    def add_paper(self, paper: Dict[str, Any]) -> bool:
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

    def _serialize_paper_db_value(self, value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, (list, tuple, set)):
            return json.dumps(list(value), ensure_ascii=False)

        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)

        return str(value).strip()

    def _deserialize_paper_db_value(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value

        text = value.strip()
        if not text:
            return ""

        if text.startswith("[") or text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value

        return value

    def get_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT arxiv_id, title, authors, abstract, categories, published_date, url, embedding_id, embedding_model, created_at
                    FROM arxiv_papers WHERE arxiv_id = ?
                ''', (arxiv_id,))

                row = cursor.fetchone()
                if row:
                    return {
                        'arxiv_id': row[0],
                        'title': row[1],
                        'authors': self._deserialize_paper_db_value(row[2]),
                        'abstract': row[3],
                        'categories': self._deserialize_paper_db_value(row[4]),
                        'published_date': row[5],
                        'url': row[6],
                        'embedding_id': row[7],
                        'embedding_model': row[8],
                        'created_at': row[9]
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting paper: {str(e)}")
            return None

    def search_papers_by_category(self, category: str) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT arxiv_id, title, authors, abstract, categories, published_date, url, embedding_id, created_at
                    FROM arxiv_papers WHERE categories LIKE ?
                ''', (f'%{category}%',))

                results = []
                for row in cursor.fetchall():
                    results.append({
                        'arxiv_id': row[0],
                        'title': row[1],
                        'authors': self._deserialize_paper_db_value(row[2]),
                        'abstract': row[3],
                        'categories': self._deserialize_paper_db_value(row[4]),
                        'published_date': row[5],
                        'url': row[6],
                        'embedding_id': row[7],
                        'created_at': row[8]
                    })
                return results
        except Exception as e:
            logger.error(f"Error searching papers by category: {str(e)}")
            return []

    def get_all_papers(self) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT arxiv_id, title, authors, abstract, categories, published_date, url, embedding_id, created_at
                    FROM arxiv_papers ORDER BY published_date DESC
                ''')

                results = []
                for row in cursor.fetchall():
                    results.append({
                        'arxiv_id': row[0],
                        'title': row[1],
                        'authors': self._deserialize_paper_db_value(row[2]),
                        'abstract': row[3],
                        'categories': self._deserialize_paper_db_value(row[4]),
                        'published_date': row[5],
                        'url': row[6],
                        'embedding_id': row[7],
                        'created_at': row[8]
                    })
                return results
        except Exception as e:
            logger.error(f"Error getting all papers: {str(e)}")
            return []

    def get_total_paper_count(self) -> int:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM arxiv_papers')
                row = cursor.fetchone()
                return int(row[0] or 0) if row else 0
        except Exception as e:
            logger.error(f"Error getting total paper count: {str(e)}")
            return 0

    def get_today_new_paper_count(self) -> int:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT COUNT(*)
                    FROM arxiv_papers
                    WHERE date(created_at, 'localtime') = date('now', 'localtime')
                    '''
                )
                row = cursor.fetchone()
                return int(row[0] or 0) if row else 0
        except Exception as e:
            logger.error(f"Error getting today's new paper count: {str(e)}")
            return 0

    def delete_paper(self, arxiv_id: str) -> bool:
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

    def update_paper_embedding(self, arxiv_id: str, embedding_id: int, embedding_model: str = None) -> bool:
        """
        更新论文在向量库中的 embedding 元数据。

        这里只保存向量库引用，不负责写入真实向量；调用方需要先完成向量写入，
        再把 embedding_id 和 embedding_model 回填到本地论文目录表。
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
