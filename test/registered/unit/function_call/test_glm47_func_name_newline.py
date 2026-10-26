import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


FETCH_SCHEMA = {
    "type": "object",
    "properties": {
        "method": {"type": "string"},
        "params": {"type": "string"},
        "url": {"type": "string"},
    },
    "required": ["method", "params", "url"],
}

FETCH_ARGS = [
    ("method", "GET"),
    ("params", "page=2"),
    ("url", "https://example.com"),
]

EXPECTED_FETCH = {
    "method": "GET",
    "params": "page=2",
    "url": "https://example.com",
}


def _tools(name, schema):
    return [Tool(type="function", function=Function(name=name, parameters=schema))]


def _fetch_text(newline_after_name, newline_after_key):
    name_sep = "\n" if newline_after_name else ""
    key_sep = "\n" if newline_after_key else ""
    parts = [f"<tool_call>api_fetch{name_sep}"]
    for key, value in FETCH_ARGS:
        parts.append(
            f"<arg_key>{key}</arg_key>{key_sep}<arg_value>{value}</arg_value>{key_sep}"
        )
    parts.append("</tool_call>")
    return "".join(parts)


def _parse(detector, text, tools, chunk_size):
    if chunk_size is None:
        return detector.detect_and_parse(text, tools).calls
    calls = []
    for offset in range(0, len(text), chunk_size):
        calls.extend(
            detector.parse_streaming_increment(
                text[offset : offset + chunk_size], tools
            ).calls
        )
    return calls


class TestGlm47FuncNameNewline(unittest.TestCase):
    """glm47 non-streaming must strip a newline after the function name."""

    def test_non_streaming_newline_after_name(self):
        text = _fetch_text(newline_after_name=True, newline_after_key=False)
        calls = (
            Glm47MoeDetector()
            .detect_and_parse(text, _tools("api_fetch", FETCH_SCHEMA))
            .calls
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "api_fetch")

    def test_non_streaming_newlines_around_arguments(self):
        text = _fetch_text(newline_after_name=True, newline_after_key=True)
        calls = (
            Glm47MoeDetector()
            .detect_and_parse(text, _tools("api_fetch", FETCH_SCHEMA))
            .calls
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "api_fetch")
        self.assertEqual(json.loads(calls[0].parameters), EXPECTED_FETCH)

    def test_streaming_still_strips_name(self):
        text = _fetch_text(newline_after_name=True, newline_after_key=False)
        for chunk_size in (1, 4, 9):
            with self.subTest(chunk_size=chunk_size):
                calls = _parse(
                    Glm47MoeDetector(),
                    text,
                    _tools("api_fetch", FETCH_SCHEMA),
                    chunk_size,
                )
                self.assertEqual(
                    json.loads("".join(c.parameters for c in calls)), EXPECTED_FETCH
                )


if __name__ == "__main__":
    unittest.main()
