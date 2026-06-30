import unittest
from unittest.mock import patch, MagicMock
import sector_scout_3

# Reasons are >= 50 chars so validate_llm_response does not apply its
# weak-reasoning (x0.7) penalty, and avoid the "insufficient"/"not enough"
# hedging keywords that force a 0.5 score.
GOOD_REASON = "Strong revenue growth and broad analyst upgrades support continued upside."
BAD_REASON = "Margins are compressing and guidance was cut sharply for the next two quarters."


class TestParser(unittest.TestCase):
    def _mock_llm(self, content):
        """Build a mock matching LM Studio's /v1/chat/completions response shape."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        return mock_response

    @patch('requests.post')
    def test_clean_json(self, mock_post):
        # Clean JSON straight from the model
        mock_post.return_value = self._mock_llm(
            f'{{"score": 0.9, "reason": "{GOOD_REASON}"}}'
        )
        score, reason = sector_scout_3.ask_llama("AAPL", "trend_targets", "headline text", "tier1_news")
        self.assertEqual(score, 0.9)
        self.assertEqual(reason, GOOD_REASON)

    @patch('requests.post')
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

    @patch('requests.post')
    def test_broken_json(self, mock_post):
        # No JSON object anywhere -> JSON Parse Failed
        mock_post.return_value = self._mock_llm("I cannot do that.")
        score, reason = sector_scout_3.ask_llama("AAPL", "trend_targets", "headline text", "tier1_news")
        self.assertEqual(score, 0.0)
        self.assertEqual(reason, "JSON Parse Failed")


if __name__ == '__main__':
    unittest.main()
