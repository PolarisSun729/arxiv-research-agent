"""SQLite 存储层各 Store 共享的常量、异常和小工具函数。

这里只放“无状态、与具体表无关”的内容；任何依赖某张表结构的逻辑都应留在对应 Store 中，
避免 shared 模块重新膨胀成万能工具箱。
"""

import logging
from datetime import datetime, timezone
from typing import Any, Sized

from utils.config import get_legacy_default_user_id

logger = logging.getLogger("services.storage.sqlite")

# 未登录/旧数据场景下使用的兜底用户；由配置层统一给出，避免各 Store 自行硬编码。
DEFAULT_USER_ID = get_legacy_default_user_id()

# 画像构建链路的版本号会写入 job 和快照，用于判断旧缓存是否仍可复用。
PROFILE_BUILD_VERSION = "research_profile_v2"
PROFILE_EXTRACTOR_VERSION = "llm_paper_evidence_v1"
PROFILE_NORMALIZER_VERSION = "profile_normalizer_v1"


class PaperQATurnPersistenceError(RuntimeError):
    """QA 单轮对话写入失败时抛出的强语义异常，避免关键写路径静默丢失。"""


def normalize_user_id(user_id: Any) -> str:
    """把空值、纯空白的 user_id 统一回落到 DEFAULT_USER_ID。

    多个 Store 都需要按 user_id 做归属隔离；集中归一化可以保证
    “None / '' / '  '” 在所有表里都落到同一个兜底用户，而不是各自写出不同的空值。
    """
    return str(user_id or DEFAULT_USER_ID).strip() or DEFAULT_USER_ID


def utc_now_iso() -> str:
    """返回带时区的 UTC ISO 时间字符串，供应用层自行写入的时间字段统一格式。"""
    return datetime.now(timezone.utc).isoformat()


def sql_placeholders(values: Sized) -> str:
    """为 SQL `IN (...)` 子句生成与参数个数一致的 `?` 占位符。

    只拼接占位符、不拼接值本身，真实参数仍通过 execute 绑定，避免 SQL 注入。
    调用方需要自行保证 values 非空，空的 `IN ()` 在 SQLite 中是语法错误。
    """
    return ", ".join("?" for _ in range(len(values)))


# 论文笔记允许的类型；不在集合内的类型在写入时统一回落为 custom。
PAPER_NOTE_TYPES = {
    "summary",
    "method",
    "experiment",
    "result",
    "limitation",
    "idea",
    "todo",
    "custom",
}
