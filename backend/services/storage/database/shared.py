import logging

from utils.config import get_default_user_id

logger = logging.getLogger("services.storage.database_service")

DEFAULT_USER_ID = get_default_user_id()

PROFILE_BUILD_VERSION = "research_profile_v2"
PROFILE_EXTRACTOR_VERSION = "llm_paper_evidence_v1"
PROFILE_NORMALIZER_VERSION = "profile_normalizer_v1"


class PaperQATurnPersistenceError(RuntimeError):
    """QA 单轮对话写入失败时抛出的强语义异常，避免关键写路径静默丢失。"""


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
