import ast
from pathlib import Path

import dependencies


REMOVED_SEARCH_METHODS = {
    "search_papers",
    "search_advanced",
    "search_by_author",
    "search_by_title",
    "search_by_category",
    "search_by_abstract",
}


def _backend_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _class_method_names(relative_path: str, class_name: str) -> set[str]:
    source_path = _backend_dir() / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            # 结构测试只关心源码暴露的方法名，不 import 重型服务，避免触发向量库等运行时依赖。
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"{class_name} not found in {relative_path}")


def test_local_arxiv_service_compatibility_file_is_removed() -> None:
    assert not (_backend_dir() / "services" / "arxiv" / "local_arxiv_service.py").exists()


def test_search_backends_expose_only_the_shared_search_contract() -> None:
    backend_sources = {
        "services/arxiv/arxiv_search_service.py": "ArxivSearchService",
        "services/arxiv/arxiv_oai_service.py": "ArxivOaiDatabaseService",
    }
    for relative_path, class_name in backend_sources.items():
        methods = _class_method_names(relative_path, class_name)
        # 新搜索边界只允许 search + capability 元数据接口；旧便捷方法会重新制造分叉路径。
        assert "search" in methods
        assert "get_available_fields" in methods
        assert "get_subject_categories" in methods
        for method_name in REMOVED_SEARCH_METHODS:
            assert method_name not in methods


def test_dependencies_expose_search_backend_factory_only() -> None:
    service_getter_names = {name for name, _getter in dependencies.iter_service_getters()}

    assert hasattr(dependencies, "get_arxiv_search_backend")
    assert not hasattr(dependencies, "get_arxiv_service")
    assert not hasattr(dependencies, "get_local_arxiv_service")
    assert "arxiv_search_backend" in service_getter_names
    assert "arxiv_service" not in service_getter_names
    assert "local_arxiv_service" not in service_getter_names
