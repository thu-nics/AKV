"""Token-native GPT-OSS regressions shared by Chat streaming/non-streaming."""

import unittest

from openai_harmony import HarmonyEncodingName, load_harmony_encoding
from sglang.srt.parser.harmony_chat_parser import HarmonyChatParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

TOOLS = [
    {"type": "function", "function": {"name": name}}
    for name in ("search", "get_document")
]


class TestHarmonyChatParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def tokens(self, text):
        return self.encoding.encode(text, allowed_special="all")

    def collect(self, ids, size, incremental=True, finish="stop"):
        parser = HarmonyChatParser(TOOLS)
        pieces = []
        for i in range(0, len(ids), size):
            end = min(i + size, len(ids))
            pieces.append(
                parser.feed_output(
                    ids[i:end] if incremental else ids[:end],
                    incremental=incremental,
                    completion_tokens=end,
                )
            )
        pieces.append(parser.finish(finish))
        self.assertFalse(parser.finish().tool_calls)
        return parser._join(pieces)

    def test_recipient_before_channel_at_every_token_split(self):
        for name, args in [
            ("search", '{"query":"中文 🐈"}'),
            ("get_document", '{"docid":"98474"}'),
        ]:
            for channel in ["analysis", "commentary"]:
                for header in [
                    f"to=functions.{name}<|channel|>{channel}",
                    f"<|channel|>{channel} to=functions.{name}",
                ]:
                    ids = self.tokens(header + "<|message|>" + args + "<|call|>")
                    for size in range(1, len(ids) + 1):
                        for incremental in [True, False]:
                            with self.subTest(
                                name=name,
                                header=header,
                                size=size,
                                incremental=incremental,
                            ):
                                result = self.collect(ids, size, incremental)
                                self.assertEqual(result.reasoning_content, "")
                                self.assertEqual(result.content, "")
                                self.assertEqual(len(result.tool_calls), 1)
                                self.assertEqual(
                                    result.tool_calls[0]["function"],
                                    {"name": name, "arguments": args},
                                )

    def test_reasoning_then_tool_then_final(self):
        text = (
            "<|channel|>analysis<|message|>Think 中文<|end|>"
            '<|start|>assistant to=functions.search<|channel|>analysis<|message|>{"query":"x"}<|call|>'
            '<|start|>assistant to=functions.get_document<|channel|>commentary<|message|>{"docid":"1"}<|call|>'
            "<|start|>assistant<|channel|>final<|message|>Answer<|return|>"
        )
        for size in [1, 7, 1000]:
            result = self.collect(self.tokens(text), size)
            self.assertEqual(result.reasoning_content, "Think 中文")
            self.assertEqual(result.content, "Answer")
            self.assertEqual(
                [c["function"]["name"] for c in result.tool_calls],
                ["search", "get_document"],
            )
            self.assertEqual([c["index"] for c in result.tool_calls], [0, 1])

    def test_partial_tool_is_not_completed_on_length_or_abort(self):
        ids = self.tokens(
            'to=functions.search<|channel|>analysis<|message|>{"query":"unfinished'
        )
        for reason in ["length", "abort"]:
            result = self.collect(ids, 1, finish=reason)
            self.assertEqual(result.tool_calls, [])
            self.assertEqual(result.reasoning_content, "")

    def test_raw_invalid_arguments_preserved(self):
        result = self.collect(
            self.tokens(
                "to=functions.search<|channel|>analysis<|message|>{bad}<|call|>"
            ),
            1,
        )
        self.assertEqual(result.tool_calls[0]["function"]["arguments"], "{bad}")

    def test_choices_have_independent_state_and_cumulative_duplicates(self):
        ids = self.tokens(
            'to=functions.get_document<|channel|>analysis<|message|>{"docid":"2"}<|call|>'
        )
        a, b = HarmonyChatParser(TOOLS), HarmonyChatParser(TOOLS)
        a.feed_output(ids[:4])
        self.assertEqual(len(b.feed_output(ids).tool_calls), 1)
        self.assertEqual(len(a.feed_output(ids).tool_calls), 1)
        self.assertFalse(a.feed_output(ids).tool_calls)
        self.assertFalse(
            a.feed_output(
                ids[-1:], incremental=True, completion_tokens=len(ids)
            ).tool_calls
        )

    def test_unknown_tool_default_is_not_executed(self):
        result = self.collect(
            self.tokens("to=functions.unknown<|channel|>analysis<|message|>{}<|call|>"),
            1,
        )
        self.assertFalse(result.tool_calls)
        self.assertFalse(result.reasoning_content)

    def test_legacy_header_fallback(self):
        result = self.collect(
            self.tokens(
                '<|channel|>commentary<|constrain|>json to=functions.search<|message|>{"query":"x"}<|call|>'
            ),
            1,
        )
        self.assertEqual(result.tool_calls[0]["function"]["name"], "search")
        self.assertFalse(result.reasoning_content)


if __name__ == "__main__":
    unittest.main()
