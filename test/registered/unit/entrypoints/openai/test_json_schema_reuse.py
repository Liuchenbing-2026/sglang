"""Preserve caller-owned schemas during chat request normalization."""

import copy
import unittest

from pydantic import BaseModel, ValidationError

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestJsonSchemaReuse(CustomTestCase):
    def build(self, response_format):
        return ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            response_format=response_format,
        )

    def test_repeated_schema_keeps_strict_default(self):
        for strict in (True, False):
            with self.subTest(strict=strict):
                strict_default = strict

                class Note(BaseModel):
                    title: str
                    strict: bool = strict_default

                schema = Note.model_json_schema()
                before = copy.deepcopy(schema)
                first = self.build({"type": "json_schema", "schema": schema})
                second = self.build({"type": "json_schema", "schema": schema})
                self.assertIs(first.response_format.json_schema.strict, strict)
                self.assertIs(second.response_format.json_schema.strict, strict)
                self.assertEqual(schema, before)

    def test_repeated_response_format_does_not_change_input(self):
        response_format = {
            "type": "json_schema",
            "schema": {
                "title": "Answer",
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        }
        before = copy.deepcopy(response_format)
        first = self.build(response_format)
        self.assertEqual(response_format, before)
        second = self.build(response_format)
        self.assertEqual(first.response_format, second.response_format)
        self.assertEqual(response_format, before)

    def test_canonical_format_takes_precedence_without_popping_legacy_schema(self):
        response_format = {
            "type": "json_schema",
            "schema": {"type": "string"},
            "json_schema": {
                "name": "Canonical",
                "schema": {"type": "integer"},
                "strict": False,
            },
        }
        before = copy.deepcopy(response_format)
        result = self.build(response_format)
        self.assertEqual(result.response_format.json_schema.name, "Canonical")
        self.assertEqual(
            result.response_format.json_schema.schema_, {"type": "integer"}
        )
        self.assertIs(result.response_format.json_schema.strict, False)
        self.assertEqual(response_format, before)

    def test_schema_without_properties_keeps_its_shape(self):
        schema = {"title": "Answer", "type": "array", "items": {"type": "string"}}
        before = copy.deepcopy(schema)
        result = self.build({"type": "json_schema", "schema": schema})
        self.assertEqual(result.response_format.json_schema.schema_, before)
        self.assertEqual(schema, before)
        self.assertIsNone(result.response_format.json_schema.strict)

    def test_strict_extraction_preserves_other_properties_and_nested_definitions(self):
        schema = {
            "title": "Answer",
            "properties": {
                "strict": {"type": "boolean", "default": True},
                "text": {"$ref": "#/$defs/Text"},
            },
            "$defs": {"Text": {"type": "string", "maxLength": 12}},
        }
        before = copy.deepcopy(schema)
        result = self.build({"type": "json_schema", "schema": schema})
        expected = copy.deepcopy(before)
        expected["properties"].pop("strict")
        self.assertEqual(result.response_format.json_schema.schema_, expected)
        self.assertIs(result.response_format.json_schema.strict, True)
        self.assertEqual(schema, before)

    def test_empty_legacy_schema_is_unchanged(self):
        response_format = {"type": "json_schema", "schema": {}}
        before = copy.deepcopy(response_format)
        self.assertIsNone(self.build(response_format).response_format.json_schema)
        self.assertEqual(response_format, before)

    def test_non_schema_formats_are_unchanged(self):
        for response_format in (None, {"type": "json_object"}, {"type": "text"}):
            with self.subTest(response_format=response_format):
                before = copy.deepcopy(response_format)
                result = self.build(response_format)
                if response_format is not None:
                    self.assertEqual(
                        result.response_format.type, response_format["type"]
                    )
                else:
                    self.assertIsNone(result.response_format)
                self.assertEqual(response_format, before)

    def test_failed_request_validation_does_not_change_input(self):
        response_format = {
            "type": "json_schema",
            "schema": {
                "properties": {"strict": {"type": "boolean", "default": True}},
            },
        }
        before = copy.deepcopy(response_format)
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(model="test-model", response_format=response_format)
        self.assertEqual(response_format, before)


if __name__ == "__main__":
    unittest.main()
