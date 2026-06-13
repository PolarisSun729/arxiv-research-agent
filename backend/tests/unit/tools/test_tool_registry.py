import unittest
from unittest import mock

from pydantic import BaseModel

import tools.tool_registry as tool_registry


class _DemoInput(BaseModel):
    count: int


class ToolRegistryUnitTests(unittest.TestCase):
    def test_validate_arguments_returns_normalized_payload(self) -> None:
        payload = tool_registry._validate_arguments(_DemoInput, {"count": 3})

        self.assertEqual(payload, {"count": 3})

    def test_validate_arguments_raises_for_invalid_payload(self) -> None:
        with self.assertRaises(Exception):
            tool_registry._validate_arguments(_DemoInput, {"count": "bad"})

    def test_invoke_tool_returns_not_found_error(self) -> None:
        result = tool_registry.invoke_tool("missing_tool")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "tool_not_found")

    def test_invoke_tool_wraps_successful_execution(self) -> None:
        spec = tool_registry.ToolSpec(
            name="demo",
            description="demo tool",
            input_schema=_DemoInput,
            func=lambda count: {"ok": True, "tool_name": "demo", "summary": f"count={count}", "data": {"count": count}, "trace": {}, "error": None},
        )
        with mock.patch.dict(tool_registry.TOOL_REGISTRY, {"demo": spec}, clear=True):
            result = tool_registry.invoke_tool("demo", count=5)

        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["count"], 5)
        self.assertEqual(result["tool_name"], "demo")
        self.assertEqual(result["trace"]["requested_tool_name"], "demo")
        self.assertEqual(result["trace"]["canonical_tool_name"], "demo")

    def test_invoke_tool_normalizes_alias_result_name(self) -> None:
        spec = tool_registry.ToolSpec(
            name="demo_alias",
            description="demo alias tool",
            input_schema=_DemoInput,
            func=lambda count: {
                "ok": True,
                "tool_name": "demo",
                "summary": f"count={count}",
                "data": {"count": count},
                "trace": {"source": "unit-test"},
                "error": None,
            },
            result_tool_name="demo",
        )
        with mock.patch.dict(tool_registry.TOOL_REGISTRY, {"demo_alias": spec}, clear=True):
            result = tool_registry.invoke_tool("demo_alias", count=5)

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool_name"], "demo_alias")
        self.assertEqual(result["trace"]["requested_tool_name"], "demo_alias")
        self.assertEqual(result["trace"]["canonical_tool_name"], "demo")
        self.assertTrue(result["trace"]["tool_alias"])

    def test_invoke_tool_wraps_validation_failure(self) -> None:
        spec = tool_registry.ToolSpec(
            name="demo",
            description="demo tool",
            input_schema=_DemoInput,
            func=lambda count: {"ok": True, "tool_name": "demo", "summary": "unused", "data": {"count": count}, "trace": {}, "error": None},
        )
        with mock.patch.dict(tool_registry.TOOL_REGISTRY, {"demo": spec}, clear=True):
            result = tool_registry.invoke_tool("demo", count="bad")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "tool_argument_validation_failed")

    def test_invoke_tool_wraps_execution_failure(self) -> None:
        def _raise(count: int):
            raise RuntimeError(f"boom:{count}")

        spec = tool_registry.ToolSpec(
            name="demo",
            description="demo tool",
            input_schema=_DemoInput,
            func=_raise,
        )
        with mock.patch.dict(tool_registry.TOOL_REGISTRY, {"demo": spec}, clear=True):
            result = tool_registry.invoke_tool("demo", count=2)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "tool_execution_failed")
        self.assertIn("boom:2", result["error"]["message"])


    def test_registry_has_no_duplicate_user_alias_specs(self) -> None:
        # canonical 名仍在注册表里；冗余的 *_user_* ToolSpec 副本已被 alias map 取代。
        self.assertIn("record_paper_preference", tool_registry.TOOL_REGISTRY)
        self.assertIn("remove_paper_preference", tool_registry.TOOL_REGISTRY)
        self.assertNotIn("record_user_paper_preference", tool_registry.TOOL_REGISTRY)
        self.assertNotIn("remove_user_paper_preference", tool_registry.TOOL_REGISTRY)

    def test_resolve_tool_name_maps_alias_to_canonical(self) -> None:
        self.assertEqual(tool_registry.resolve_tool_name("record_user_paper_preference"), "record_paper_preference")
        self.assertEqual(tool_registry.resolve_tool_name("remove_user_paper_preference"), "remove_paper_preference")
        # 非别名原样返回。
        self.assertEqual(tool_registry.resolve_tool_name("search_arxiv_raw"), "search_arxiv_raw")

    def test_invoke_tool_resolves_alias_to_canonical_spec_and_records_trace(self) -> None:
        captured = {}

        def _canonical_func(**kwargs):
            captured.update(kwargs)
            return {"ok": True, "tool_name": "record_paper_preference", "summary": "done", "data": {"count": kwargs.get("count")}, "trace": {}, "error": None}

        spec = tool_registry.ToolSpec(
            name="record_paper_preference",
            description="canonical demo",
            input_schema=_DemoInput,
            func=_canonical_func,
        )
        aliases = {"record_user_paper_preference": "record_paper_preference"}
        with mock.patch.dict(tool_registry.TOOL_REGISTRY, {"record_paper_preference": spec}, clear=True), \
             mock.patch.dict(tool_registry.TOOL_ALIASES, aliases, clear=True):
            result = tool_registry.invoke_tool("record_user_paper_preference", count=7)

        self.assertTrue(result["ok"])
        self.assertEqual(captured["count"], 7)
        # 调用者看到的 tool_name 保留请求别名，trace 记录请求名与 canonical 名以便审计。
        self.assertEqual(result["tool_name"], "record_user_paper_preference")
        self.assertEqual(result["trace"]["requested_tool_name"], "record_user_paper_preference")
        self.assertEqual(result["trace"]["canonical_tool_name"], "record_paper_preference")

    def test_get_tool_aliases_returns_alias_map_copy(self) -> None:
        aliases = tool_registry.get_tool_aliases()
        self.assertEqual(aliases.get("record_user_paper_preference"), "record_paper_preference")
        aliases["__mutation__"] = "x"
        self.assertNotIn("__mutation__", tool_registry.get_tool_aliases())


if __name__ == "__main__":
    unittest.main()
