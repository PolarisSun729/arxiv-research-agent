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


if __name__ == "__main__":
    unittest.main()
