import json
import unittest
from unittest.mock import patch, MagicMock
import sector_scout_3

# Reasons are >= 50 chars so validate_llm_response does not apply its
# weak-reasoning (x0.7) penalty, and avoid the "insufficient"/"not enough"
# hedging keywords that force a 0.5 score.
GOOD_REASON = "Strong revenue growth and broad analyst upgrades support continued upside."
BAD_REASON = "Margins are compressing and guidance was cut sharply for the next two quarters."


class TestParser(unittest.TestCase):
    def _mock_llm(self, content, finish_reason="stop"):
        """Build a mock matching LM Studio's /v1/chat/completions response shape."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": content},
                "finish_reason": finish_reason,
            }]
        }
        return mock_response

    @patch('sector_scout_3.requests.post')
    def test_clean_json(self, mock_post):
        # Clean JSON straight from the model
        mock_post.return_value = self._mock_llm(
            f'{{"score": 0.9, "reason": "{GOOD_REASON}"}}'
        )
        score, reason = sector_scout_3.ask_llama("AAPL", "trend_targets", "headline text", "tier1_news")
        self.assertEqual(score, 0.9)
        self.assertEqual(reason, GOOD_REASON)

    @patch('sector_scout_3.requests.post')
    def test_chatty_json(self, mock_post):
        # Markdown-fenced JSON with preamble/postamble (regex fallback path)
        chatty = (
            "Here is the analysis you requested:\n"
            "```json\n"
            f'{{"score": 0.4, "reason": "{BAD_REASON}"}}\n'
            "```\n"
            "Hope this helps!"
        )
        mock_post.return_value = self._mock_llm(chatty)
        score, reason = sector_scout_3.ask_llama("AAPL", "trend_targets", "headline text", "tier1_news")
        self.assertEqual(score, 0.4)
        self.assertEqual(reason, BAD_REASON)

    @patch('sector_scout_3.requests.post')
    def test_broken_json(self, mock_post):
        # No JSON object anywhere -> a failed call (None), not a 0.0 verdict
        mock_post.return_value = self._mock_llm("I cannot do that.")
        score, reason = sector_scout_3.ask_llama("AAPL", "trend_targets", "headline text", "tier1_news")
        self.assertIsNone(score)
        self.assertEqual(reason, "JSON Parse Failed")

    @patch('sector_scout_3.requests.post')
    def test_lm_studio_error_is_a_failed_call(self, mock_post):
        # The 09-22 shape: HTTP 400 with no model loaded. The body has no
        # 'choices', which used to log KeyError 'choices' and score 0.0.
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "No models loaded. Please load a model."}
        mock_post.return_value = mock_response
        score, reason = sector_scout_3.ask_llama("AAPL", "trend_targets", "headline text", "tier1_news")
        self.assertIsNone(score)
        self.assertIn("HTTP 400", reason)
        self.assertIn("No models loaded", reason)

    @patch('sector_scout_3.requests.post')
    def test_unreachable_lm_studio_is_a_failed_call(self, mock_post):
        mock_post.side_effect = ConnectionError("connection refused")
        score, reason = sector_scout_3.ask_llama("AAPL", "trend_targets", "headline text", "tier1_news")
        self.assertIsNone(score)
        self.assertIn("connection refused", reason)

    @patch('sector_scout_3.requests.post')
    def test_reply_without_a_score_is_a_failed_call(self, mock_post):
        # Missing, non-numeric and non-finite scores used to default to (or
        # crash into) 0.0. None of them is the model's verdict.
        for body in (f'{{"reason": "{GOOD_REASON}"}}',
                     f'{{"score": "high", "reason": "{GOOD_REASON}"}}',
                     f'{{"score": NaN, "reason": "{GOOD_REASON}"}}',
                     '[0.9]'):
            with self.subTest(body=body):
                mock_post.return_value = self._mock_llm(body)
                score, _ = sector_scout_3.ask_llama("AAPL", "trend_targets", "headline text", "tier1_news")
                self.assertIsNone(score)

    @patch('sector_scout_3.requests.post')
    def test_bearish_verdict_is_still_zero(self, mock_post):
        # A real 0.0 from the model is a verdict and must stay one.
        mock_post.return_value = self._mock_llm(
            f'{{"score": 0.0, "reason": "{BAD_REASON}"}}'
        )
        score, reason = sector_scout_3.ask_llama("AAPL", "trend_targets", "headline text", "tier1_news")
        self.assertEqual(score, 0.0)
        self.assertEqual(reason, BAD_REASON)

    @patch('sector_scout_3.requests.post')
    def test_shadow_advisor_sends_json_schema(self, mock_post):
        mock_post.return_value = self._mock_llm(json.dumps({
            "decision": "approve",
            "confidence": 0.75,
            "reasoning": "Durable trend with liquid trading conditions.",
            "risk_flags": [],
        }))

        vote = sector_scout_3.ask_shadow_advisor(
            "NVDA", "trend_targets", 0.80, 0.70, {}, "")

        self.assertFalse(vote["advisor_failed"])
        self.assertEqual(mock_post.call_count, 1)
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(vote["diagnostics"]["model"], sector_scout_3.MODEL_NAME)
        self.assertEqual(vote["diagnostics"]["attempt_count"], 1)
        self.assertFalse(vote["diagnostics"]["retry_used"])

    @patch('sector_scout_3.requests.post')
    def test_shadow_advisor_repairs_one_parse_failure(self, mock_post):
        mock_post.side_effect = [
            self._mock_llm("Analysis: {not valid JSON", finish_reason="length"),
            self._mock_llm(json.dumps({
                "decision": "watch",
                "confidence": 0.58,
                "reasoning": "Evidence is mixed and needs another observation.",
                "risk_flags": ["mixed_signal"],
            })),
        ]

        vote = sector_scout_3.ask_shadow_advisor(
            "AMD", "wheel_targets", 0.65, 0.61, {}, "")

        self.assertEqual(mock_post.call_count, 2)
        self.assertFalse(vote["advisor_failed"])
        self.assertEqual(vote["decision"], "watch")
        diagnostics = vote["diagnostics"]
        self.assertEqual(diagnostics["attempt_count"], 2)
        self.assertTrue(diagnostics["retry_used"])
        self.assertTrue(diagnostics["recovered_after_retry"])
        self.assertEqual(diagnostics["attempts"][0]["finish_reason"], "length")
        self.assertIn("parse_error", diagnostics["attempts"][0])
        retry_payload = mock_post.call_args_list[1].kwargs["json"]
        self.assertIn("Repair the malformed", retry_payload["messages"][0]["content"])
        self.assertEqual(retry_payload["temperature"], 0.0)

    @patch('sector_scout_3.requests.post')
    def test_shadow_advisor_stops_after_one_failed_repair(self, mock_post):
        mock_post.side_effect = [
            self._mock_llm("first malformed response"),
            self._mock_llm("second malformed response"),
        ]

        vote = sector_scout_3.ask_shadow_advisor(
            "AMD", "wheel_targets", 0.65, 0.61, {}, "")

        self.assertEqual(mock_post.call_count, 2)
        self.assertTrue(vote["advisor_failed"])
        self.assertEqual(vote["reasoning"], "specialist_json_parse_failed")
        self.assertEqual(vote["diagnostics"]["attempt_count"], 2)
        self.assertIn("raw_response_excerpt", vote["diagnostics"]["attempts"][1])

    @patch('sector_scout_3.requests.post')
    def test_shadow_advisor_skips_repair_for_empty_response(self, mock_post):
        # Nothing to repair — a schema-forced retry would fabricate a vote
        # from zero evidence and record it as a recovery.
        mock_post.return_value = self._mock_llm("")

        vote = sector_scout_3.ask_shadow_advisor(
            "AMD", "wheel_targets", 0.65, 0.61, {}, "")

        self.assertEqual(mock_post.call_count, 1)
        self.assertTrue(vote["advisor_failed"])
        self.assertEqual(vote["reasoning"], "specialist_json_parse_failed")
        self.assertEqual(vote["diagnostics"]["attempt_count"], 1)
        self.assertFalse(vote["diagnostics"]["retry_used"])

    @patch('sector_scout_3.requests.post')
    def test_shadow_advisor_trusts_parsed_sentinel_reasoning(self, mock_post):
        # A cleanly parsed vote never retries, even if its reasoning text
        # happens to echo the parse-failure sentinel.
        mock_post.return_value = self._mock_llm(json.dumps({
            "decision": "watch",
            "confidence": 0.52,
            "reasoning": "specialist_json_parse_failed",
            "risk_flags": [],
        }))

        vote = sector_scout_3.ask_shadow_advisor(
            "AMD", "wheel_targets", 0.65, 0.61, {}, "")

        self.assertEqual(mock_post.call_count, 1)
        self.assertFalse(vote["advisor_failed"])
        self.assertEqual(vote["decision"], "watch")

    @patch('sector_scout_3.requests.post')
    def test_shadow_advisor_records_request_failure(self, mock_post):
        mock_post.side_effect = RuntimeError("LM Studio unavailable")

        vote = sector_scout_3.ask_shadow_advisor(
            "AMD", "wheel_targets", 0.65, 0.61, {}, "")

        self.assertTrue(vote["advisor_failed"])
        diagnostics = vote["diagnostics"]
        self.assertEqual(diagnostics["attempt_count"], 1)
        self.assertEqual(diagnostics["attempts"][0]["finish_reason"], "request_failed")
        self.assertIn("LM Studio unavailable", diagnostics["api_error"])


if __name__ == '__main__':
    unittest.main()
