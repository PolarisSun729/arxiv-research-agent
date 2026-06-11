import importlib.util
import sys
import types
from pathlib import Path


def _reload_real_config_module() -> None:
    backend_dir = Path(__file__).resolve().parents[2]
    utils_dir = backend_dir / "utils"
    utils_package = sys.modules.get("utils")
    if utils_package is None or not getattr(utils_package, "__path__", None):
        utils_package = types.ModuleType("utils")
        utils_package.__path__ = [str(utils_dir)]
        sys.modules["utils"] = utils_package

    spec = importlib.util.spec_from_file_location("utils.config", utils_dir / "config.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["utils.config"] = module
    spec.loader.exec_module(module)


embedding_module = types.ModuleType("services.embedding.embedding_service")
embedding_module.EmbeddingService = object
sys.modules["services.embedding.embedding_service"] = embedding_module

vector_store_module = types.ModuleType("services.storage.vector_store_service")
vector_store_module.VectorStoreService = object
sys.modules["services.storage.vector_store_service"] = vector_store_module

# 前序 Agent 单测会改写 utils.config 的运行时函数；这里恢复真实配置模块，
# 既保证 OAI 单测拿到完整配置，也避免半截桩影响后续 arXiv 查询测试收集。
_reload_real_config_module()

# 前序集成测试会为轻量导入写入同名空桩模块；本单测验证真实 OAI 查询逻辑，
# 因此导入前必须清理残留桩，避免拿到没有查询方法的占位类。
sys.modules.pop("services.arxiv.arxiv_oai_service", None)

from services.arxiv.arxiv_oai_service import ArxivOaiDatabaseService


def _make_service_without_init() -> ArxivOaiDatabaseService:
    return ArxivOaiDatabaseService.__new__(ArxivOaiDatabaseService)


def test_short_english_query_matches_complete_token_only() -> None:
    service = _make_service_without_init()

    relevant_paper = {
        "title": "Retrieval-Augmented Generation for Scientific QA",
        "abstract": "This paper studies RAG systems for multi-hop question answering.",
        "authors": "Alice Example",
        "categories": "cs.CL",
        "primary_category": "cs.CL",
    }
    substring_only_paper = {
        "title": "Operation-Guided Progressive Human-to-AI Text Transformation Benchmark",
        "abstract": "The benchmark studies progressive editing at multiple granularities.",
        "authors": "Marius Dragoi, Alexandra Dragomir",
        "categories": "cs.CL",
        "primary_category": "cs.CL",
    }

    # RAG 是短英文主题词，必须按完整 token 匹配，不能因为作者名或普通单词包含 rag 子串而误召回。
    assert service._matches_query(relevant_paper, "all:RAG") is True
    assert service._matches_query(substring_only_paper, "all:RAG") is False


def test_relevance_score_prioritizes_topic_fields_over_author_substrings() -> None:
    service = _make_service_without_init()

    topic_paper = {
        "title": "IA-RAG for Dynamic Knowledge Retrieval",
        "abstract": "The method improves retrieval-augmented generation over temporal knowledge.",
        "authors": "Alice Example",
        "categories": "cs.CL",
        "primary_category": "cs.CL",
    }
    incidental_paper = {
        "title": "Continual Learning with Spectral Updates",
        "abstract": "The method studies parameter-efficient adaptation.",
        "authors": "Marius Dragoi, Alexandra Dragomir",
        "categories": "cs.LG",
        "primary_category": "cs.LG",
    }

    assert service._score_relevance_query(topic_paper, "all:RAG") > service._score_relevance_query(incidental_paper, "all:RAG")
