import json
import re
from typing import Any, List


PROFILE_TOPIC_BLOCKLIST = {
    "analysis",
    "approach",
    "framework",
    "method",
    "methods",
    "model",
    "models",
    "paper",
    "papers",
    "system",
    "systems",
}
PROFILE_ARXIV_ID_PATTERN = re.compile(r"^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)$")
PROFILE_ARXIV_CATEGORY_PATTERN = re.compile(r"^[a-z-]+(?:\.[A-Z]{2})?$")
PROFILE_URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)


def normalize_profile_list_value(values: Any, limit: int = 30) -> List[str]:
    if values is None:
        return []
    source = values if isinstance(values, list) else [values]
    normalized: List[str] = []
    for item in source:
        text = str(item or "").strip()
        if text and text not in normalized:
            normalized.append(text)
        if len(normalized) >= limit:
            break
    return normalized


def looks_like_profile_arxiv_id(value: str) -> bool:
    text = str(value or "").strip()
    if text.startswith(("http://", "https://")):
        text = text.rstrip("/").rsplit("/", 1)[-1]
    return bool(text and PROFILE_ARXIV_ID_PATTERN.match(text))


def looks_like_profile_arxiv_category(value: str) -> bool:
    text = str(value or "").strip()
    return bool(text and PROFILE_ARXIV_CATEGORY_PATTERN.match(text) and ("." in text or text.startswith("cs.")))


def looks_like_profile_paper_title(value: str) -> bool:
    text = str(value or "").strip()
    words = [part for part in re.split(r"\s+", text) if part]
    if len(text) > 60 or len(words) > 6:
        return True
    title_joiners = {"for", "with", "of", "using", "via", "towards", "toward", "based"}
    lower_words = {word.strip(".,:;!?()[]{}").lower() for word in words}
    if len(words) >= 4 and lower_words & title_joiners:
        return True
    return any(marker in text for marker in (":", "?", "!", " -- ", " - "))


def normalize_profile_topics(values: Any, limit: int = 30) -> List[str]:
    """清洗画像主题，过滤论文标题、arXiv id/category 和泛化过弱的占位词。"""
    normalized: List[str] = []
    for item in normalize_profile_list_value(values, limit=limit * 3):
        text = str(item or "").strip()
        if not text:
            continue
        if PROFILE_URL_PATTERN.search(text) or looks_like_profile_arxiv_id(text):
            continue
        if looks_like_profile_arxiv_category(text) or text.lower().startswith("cs."):
            continue
        if looks_like_profile_paper_title(text) or text.lower() in PROFILE_TOPIC_BLOCKLIST:
            continue
        if text not in normalized:
            normalized.append(text)
        if len(normalized) >= limit:
            break
    return normalized


def normalize_profile_categories(values: Any, limit: int = 20) -> List[str]:
    """清洗 arXiv 类别字段，兼容旧库中把类别列表序列化成字符串的记录。"""
    categories: List[str] = []
    source: List[str] = []
    for item in normalize_profile_list_value(values, limit=limit * 4):
        if item.startswith("["):
            try:
                parsed = json.loads(item)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                source.extend(normalize_profile_list_value(parsed, limit=limit * 4))
                continue
        source.extend([part.strip() for part in re.split(r"[,\s]+", item) if part.strip()])
    for item in source:
        if not looks_like_profile_arxiv_category(item):
            continue
        if item not in categories:
            categories.append(item)
        if len(categories) >= limit:
            break
    return categories


def normalize_profile_papers(values: Any, limit: int = 20) -> List[str]:
    normalized: List[str] = []
    for item in normalize_profile_list_value(values, limit=limit * 2):
        text = item.rstrip("/").rsplit("/", 1)[-1] if item.startswith(("http://", "https://")) else item
        if not looks_like_profile_arxiv_id(text):
            continue
        if text not in normalized:
            normalized.append(text)
        if len(normalized) >= limit:
            break
    return normalized
