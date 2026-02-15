
import unittest
from unittest.mock import patch, MagicMock
import subprocess
import sector_scout_3

class TestSCP(unittest.TestCase):
    @patch('subprocess.run')
    def test_scp_success_first_try(self, mock_run):
        # Setup: Success on first try
        mock_run.return_value = MagicMock(returncode=0)
        
        result = sector_scout_3.beam_to_beelink(retries=3)
        
        self.assertTrue(result)
        self.assertEqual(mock_run.call_count, 1)

    @patch('subprocess.run')
    @patch('time.sleep')
    def test_scp_retry_success(self, mock_sleep, mock_run):
        # Setup: Fail twice, succeed third time
        # Side effect: [Fail, Fail, Success]
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, 'cmd'),
            subprocess.TimeoutExpired('cmd', 10),
            MagicMock(returncode=0)
        ]
        
        result = sector_scout_3.beam_to_beelink(retries=3)
        
        self.assertTrue(result)
        self.assertEqual(mock_run.call_count, 3)

    @patch('subprocess.run')
    @patch('time.sleep')
    def test_scp_all_fail(self, mock_sleep, mock_run):
        # Setup: Fail all times
        mock_run.side_effect = subprocess.CalledProcessError(1, 'cmd')
        
        result = sector_scout_3.beam_to_beelink(retries=2)
        
        self.assertFalse(result)
        self.assertEqual(mock_run.call_count, 2)

if __name__ == '__main__':
    unittest.main()
