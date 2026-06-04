import unittest

from tools.tool_result import make_tool_error, make_tool_result, make_tool_trace


class ToolResultUnitTests(unittest.TestCase):
    def test_make_tool_result_populates_defaults(self) -> None:
        payload = make_tool_result(ok=True, tool_name="demo", summary="done")

        self.assertEqual(payload["tool_name"], "demo")
        self.assertEqual(payload["trace"], {})
        self.assertIsNone(payload["error"])

    def test_make_tool_trace_includes_optional_fields(self) -> None:
        trace = make_tool_trace("demo", inputs={"q": "rag"}, source="mock", notes={"n": 1})

        self.assertEqual(trace["tool_name"], "demo")
        self.assertIn("timestamp", trace)
        self.assertEqual(trace["inputs"], {"q": "rag"})
        self.assertEqual(trace["source"], "mock")
        self.assertEqual(trace["notes"], {"n": 1})

    def test_make_tool_error_supports_optional_detail(self) -> None:
        error = make_tool_error("bad_request", "invalid", detail={"field": "query"})

        self.assertEqual(error["code"], "bad_request")
        self.assertEqual(error["detail"]["field"], "query")


if __name__ == "__main__":
    unittest.main()
