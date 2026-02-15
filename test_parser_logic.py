
import unittest
from unittest.mock import patch, MagicMock
import json
import sector_scout_3

class TestParser(unittest.TestCase):
    @patch('requests.post')
    def test_clean_json(self, mock_post):
        # Setup: Clean JSON response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'response': '{"score": 0.9, "reason": "Good"}'
        }
        mock_post.return_value = mock_response

        score, reason = sector_scout_3.ask_llama("AAPL", "trend", ["news"])
        self.assertEqual(score, 0.9)
        self.assertEqual(reason, "Good")

    @patch('requests.post')
    def test_chatty_json(self, mock_post):
        # Setup: Chatty JSON response (markdown, extra text)
        chatty_text = """
        Here is the analysis you requested:
        ```json
        {
            "score": 0.4,
            "reason": "Bad"
        }
        ```
        Hope this helps!
        """
        mock_response = MagicMock()
        mock_response.json.return_value = {'response': chatty_text}
        mock_post.return_value = mock_response

        score, reason = sector_scout_3.ask_llama("AAPL", "trend", ["news"])
        self.assertEqual(score, 0.4)
        self.assertEqual(reason, "Bad")

    @patch('requests.post')
    def test_broken_json(self, mock_post):
        # Setup: Totally broken
        mock_response = MagicMock()
        mock_response.json.return_value = {'response': 'I cannot do that.'}
        mock_post.return_value = mock_response

        score, reason = sector_scout_3.ask_llama("AAPL", "trend", ["news"])
        self.assertEqual(score, 0.0)
        self.assertEqual(reason, "JSON Parse Failed")

if __name__ == '__main__':
    unittest.main()
