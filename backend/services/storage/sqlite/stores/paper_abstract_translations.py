from typing import Optional

from services.storage.sqlite.base import BaseSqliteStore
from services.storage.sqlite.shared import logger


class PaperAbstractTranslationStore(BaseSqliteStore):
    """paper_abstract_translations 表：按 (论文, 目标语言) 缓存摘要译文，所有用户共享。

    缓存只是省钱提速：读失败按未命中处理、写失败只记日志，都不影响本次翻译返回给用户。
    """

    def get_translation(self, arxiv_id: str, target_lang: str, source_hash: str) -> Optional[str]:
        """返回与当前原文哈希一致的译文；未命中、原文已变或读库失败时返回 None。"""
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT translated_text FROM paper_abstract_translations "
                    "WHERE arxiv_id = ? AND target_lang = ? AND source_hash = ?",
                    (arxiv_id, target_lang, source_hash),
                ).fetchone()
            return str(row[0]) if row and row[0] else None
        except Exception as e:
            logger.error(f"Error getting abstract translation: {str(e)}")
            return None

    def save_translation(
        self,
        arxiv_id: str,
        target_lang: str,
        source_hash: str,
        translated_text: str,
    ) -> None:
        """写入或覆盖一篇论文的译文；同一论文同一语言只保留最新原文对应的一份。"""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    '''
                    INSERT INTO paper_abstract_translations
                    (arxiv_id, target_lang, source_hash, translated_text, created_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(arxiv_id, target_lang) DO UPDATE SET
                        source_hash = excluded.source_hash,
                        translated_text = excluded.translated_text,
                        created_at = excluded.created_at
                    ''',
                    (arxiv_id, target_lang, source_hash, translated_text),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving abstract translation: {str(e)}")
