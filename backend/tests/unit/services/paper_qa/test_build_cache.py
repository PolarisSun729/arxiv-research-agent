from __future__ import annotations

from pathlib import Path

from services.paper_qa.build_cache import PaperQABuildCache


def _cache_config(tmp_path: Path) -> dict[str, object]:
    return {
        "enabled": True,
        "root_dir": str(tmp_path),
        "llm_cache_name": "llm",
        "embedding_cache_name": "embedding",
        "llm_size_limit": 1024 * 1024,
        "embedding_size_limit": 1024 * 1024,
    }


def test_paper_qa_build_cache_persists_llm_results(tmp_path: Path) -> None:
    cache = PaperQABuildCache(_cache_config(tmp_path))
    key_kwargs = {
        "kind": "retrieval_summary",
        "model_name": "qwen",
        "prompt_version": "v1",
        "chunk_text": "The method trains a retriever.",
        "metadata": {"chunk_type": "text", "section_title": "Method", "ignored_runtime_field": "x"},
    }

    cache.set_llm_result("summary text", **key_kwargs)
    reopened = PaperQABuildCache(_cache_config(tmp_path))

    assert reopened.get_llm_result(**key_kwargs) == "summary text"


def test_paper_qa_build_cache_keys_change_when_prompt_or_input_changes(tmp_path: Path) -> None:
    cache = PaperQABuildCache(_cache_config(tmp_path))
    base_kwargs = {
        "kind": "retrieval_questions",
        "model_name": "qwen",
        "prompt_version": "v1",
        "chunk_text": "Retriever training text.",
        "metadata": {"chunk_type": "text", "section_title": "Training"},
    }
    cache.set_llm_result(["What is trained?"], **base_kwargs)

    changed_prompt = {**base_kwargs, "prompt_version": "v2"}
    changed_text = {**base_kwargs, "chunk_text": "Different chunk text."}

    assert cache.get_llm_result(**base_kwargs) == ["What is trained?"]
    assert cache.get_llm_result(**changed_prompt) is None
    assert cache.get_llm_result(**changed_text) is None


def test_paper_qa_build_cache_persists_embedding_vectors(tmp_path: Path) -> None:
    cache = PaperQABuildCache(_cache_config(tmp_path))
    key_kwargs = {
        "provider": "dashscope",
        "model_name": "text-embedding",
        "dimension": 3,
        "embedding_input": {"mode": "text", "text": "Index text for embedding."},
    }

    cache.set_embedding([1, 2, 3], **key_kwargs)
    reopened = PaperQABuildCache(_cache_config(tmp_path))

    assert reopened.get_embedding(**key_kwargs) == [1.0, 2.0, 3.0]
