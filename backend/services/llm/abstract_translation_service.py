"""论文摘要的中文翻译：按论文共享缓存，未命中时用小模型流式翻译。

摘要原文以服务端存储为准（本地论文表优先，其次 OAI 镜像库），不接受客户端传入的文本，
否则这个接口会变成一个可以翻译任意内容的免费 LLM 代理。
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Iterator

from core.errors import AppError, ErrorCode

logger = logging.getLogger(__name__)

TARGET_LANG = "zh"
TASK_TYPE = "abstract_translation"

_PROMPT_TEMPLATE = (
    "把下面的 arXiv 论文摘要翻译成简体中文。要求：准确流畅的学术中文；专业术语首次出现时在括号里保留英文"
    "（如「多模态实体链接（MEL）」）；模型名、数据集名、方法名保持原文；公式与 LaTeX 记号原样保留；"
    "只输出译文，不要任何解释或前后缀。\n\n{abstract}"
)


def normalize_abstract(text: Any) -> str:
    """arXiv 摘要带硬换行和多余空白；规范化后再翻译和算哈希，排版差异不会导致缓存失效。"""
    return " ".join(str(text or "").split())


class AbstractTranslationService:
    def __init__(self, *, generation_service, translation_store, paper_catalog_store, oai_database_service) -> None:
        self.generation_service = generation_service
        self.translation_store = translation_store
        self.paper_catalog_store = paper_catalog_store
        self.oai_database_service = oai_database_service

    def resolve_abstract(self, arxiv_id: str) -> str:
        """取出待翻译的摘要；两个库都没有时抛 PAPER_NOT_FOUND，让路由在开流之前返回 404。"""
        for source in (self.paper_catalog_store, self.oai_database_service):
            paper = source.get_paper(arxiv_id) if source is not None else None
            abstract = normalize_abstract((paper or {}).get("abstract"))
            if abstract:
                return abstract
        raise AppError(ErrorCode.PAPER_NOT_FOUND, context={"arxiv_id": arxiv_id, "stage": "abstract_translation"})

    def stream_translation(self, arxiv_id: str, abstract: str) -> Iterator[Dict[str, Any]]:
        """产出 delta（译文增量）事件，最后产出一条带完整译文的 done 事件；命中缓存时只有 done。"""
        source_hash = hashlib.sha256(abstract.encode("utf-8")).hexdigest()
        cached = self.translation_store.get_translation(arxiv_id, TARGET_LANG, source_hash)
        if cached:
            yield {"event": "done", "data": {"arxiv_id": arxiv_id, "translation": cached, "cached": True}}
            return

        parts = []
        try:
            for delta in self.generation_service.stream_with_qwen(
                _PROMPT_TEMPLATE.format(abstract=abstract),
                task_type=TASK_TYPE,
                enable_thinking=False,
            ):
                parts.append(delta)
                yield {"event": "delta", "data": {"text": delta}}
        except Exception as exc:
            logger.warning("Abstract translation failed: arxiv_id=%s error=%s", arxiv_id, exc)
            raise AppError(
                ErrorCode.LLM_GENERATION_FAILED,
                message="摘要翻译失败，请稍后重试。",
                detail={"stage": "abstract_translation"},
                context={"arxiv_id": arxiv_id},
            ) from exc

        translation = "".join(parts).strip()
        if not translation:
            raise AppError(
                ErrorCode.LLM_GENERATION_FAILED,
                message="摘要翻译失败，请稍后重试。",
                detail={"stage": "abstract_translation", "reason": "empty_output"},
                context={"arxiv_id": arxiv_id},
            )
        # 只缓存完整跑完的译文；用户中途断开时生成器在 yield 处被关闭，走不到这里，不会缓存半截。
        self.translation_store.save_translation(arxiv_id, TARGET_LANG, source_hash, translation)
        yield {"event": "done", "data": {"arxiv_id": arxiv_id, "translation": translation, "cached": False}}
