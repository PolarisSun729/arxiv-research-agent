from __future__ import annotations

import unittest
from unittest import mock

from pydantic import BaseModel

from tests.helpers.agent_runtime import load_agent_test_modules


_MODULES = load_agent_test_modules()
tool_registry_module = _MODULES["tool_registry_module"]
ToolSpec = tool_registry_module.ToolSpec
invoke_tool = tool_registry_module.invoke_tool


class _EchoInput(BaseModel):
    topic: str


class AgentToolRegistryTests(unittest.TestCase):
    def test_tool_not_found_returns_structured_error(self) -> None:
        result = invoke_tool("missing_tool", topic="rag")

        self.assertFalse(result["ok"])
        self.assertEqual(result["tool_name"], "missing_tool")
        self.assertEqual(result["error"]["code"], "tool_not_found")
        self.assertIn("Unknown tool", result["error"]["message"])

    def test_tool_argument_validation_failure_is_wrapped(self) -> None:
        registry = {
            "echo_tool": ToolSpec(
                name="echo_tool",
                description="Echo topic",
                input_schema=_EchoInput,
                func=lambda **kwargs: {"ok": True, "tool_name": "echo_tool", "data": kwargs},
            )
        }

        with mock.patch.dict(tool_registry_module.TOOL_REGISTRY, registry, clear=True):
            result = invoke_tool("echo_tool")

        self.assertFalse(result["ok"])
        self.assertEqual(result["summary"], "Tool argument validation failed")
        self.assertEqual(result["trace"]["tool_name"], "echo_tool")
        self.assertFalse(result["trace"]["validated"])
        self.assertEqual(result["error"]["code"], "tool_argument_validation_failed")
        self.assertIsInstance(result["error"]["detail"], list)

    def test_tool_execution_exception_is_wrapped(self) -> None:
        def boom(**_kwargs):
            raise RuntimeError("boom")

        registry = {
            "boom_tool": ToolSpec(
                name="boom_tool",
                description="Raise runtime error",
                input_schema=_EchoInput,
                func=boom,
            )
        }

        with mock.patch.dict(tool_registry_module.TOOL_REGISTRY, registry, clear=True):
            result = invoke_tool("boom_tool", topic="rag")

        self.assertFalse(result["ok"])
        self.assertEqual(result["summary"], "Tool execution failed")
        self.assertEqual(result["trace"]["tool_name"], "boom_tool")
        self.assertTrue(result["trace"]["validated"])
        self.assertEqual(result["error"]["code"], "tool_execution_failed")
        self.assertEqual(result["error"]["message"], "boom")


if __name__ == "__main__":
    unittest.main()
