import json
import tempfile
import unittest

import shadow_advisors


class ShadowAdvisorRoutingTest(unittest.TestCase):
    def test_asset_specialist_routing(self):
        self.assertEqual(
            shadow_advisors.advisor_for("wheel_targets", "AMD"),
            "options_specialist",
        )
        self.assertEqual(
            shadow_advisors.advisor_for("trend_targets", "NVDA"),
            "equity_specialist",
        )
        self.assertEqual(
            shadow_advisors.advisor_for("crypto_targets", "BTC/USD"),
            "crypto_specialist",
        )

    def test_decision_thresholds(self):
        self.assertEqual(shadow_advisors.decision_from_confidence(0.70), "approve")
        self.assertEqual(shadow_advisors.decision_from_confidence(0.50), "watch")
        self.assertEqual(shadow_advisors.decision_from_confidence(0.30), "reject")


class ShadowAdvisorVoteTest(unittest.TestCase):
    def test_parse_vote_adds_shadow_metadata_and_flags(self):
        raw = json.dumps({
            "decision": "approve",
            "confidence": 0.72,
            "reasoning": "Stable option premium candidate with liquid markets.",
            "risk_flags": ["wide_spread_watch"],
        })
        vote = shadow_advisors.parse_vote(
            raw, "AMD", "wheel_targets", tech_norm=0.80, scout_confidence=0.70)
        self.assertEqual(vote["advisor"], "options_specialist")
        self.assertEqual(vote["decision"], "approve")
        self.assertTrue(vote["shadow_only"])
        self.assertFalse(vote["advisor_failed"])
        self.assertIn("wide_spread_watch", vote["risk_flags"])

    def test_parse_failure_falls_back_to_scout_confidence(self):
        vote = shadow_advisors.parse_vote(
            "no json here", "NVDA", "trend_targets",
            tech_norm=0.30, scout_confidence=0.50)
        self.assertEqual(vote["decision"], "watch")
        self.assertTrue(vote["advisor_failed"])
        self.assertIn("weak_technical_base", vote["risk_flags"])

    def test_snapshot_summarizes_by_advisor(self):
        votes = [
            shadow_advisors.fallback_vote("AMD", "wheel_targets", 0.8, 0.7, "ok"),
            shadow_advisors.fallback_vote("NVDA", "trend_targets", 0.8, 0.5, "ok"),
            shadow_advisors.fallback_vote("BTC/USD", "crypto_targets", 0.8, 0.3, "ok"),
        ]
        snap = shadow_advisors.build_snapshot(votes, updated="2026-07-09T00:00:00Z")
        self.assertEqual(snap["summary"]["total_votes"], 3)
        self.assertEqual(snap["summary"]["by_advisor"]["options_specialist"]["approve"], 1)
        self.assertEqual(snap["summary"]["by_advisor"]["equity_specialist"]["watch"], 1)
        self.assertEqual(snap["summary"]["by_advisor"]["crypto_specialist"]["reject"], 1)

    def test_snapshot_and_history_are_written(self):
        snap = shadow_advisors.build_snapshot([], updated="2026-07-09T00:00:00Z")
        with tempfile.NamedTemporaryFile() as snapshot_file, tempfile.NamedTemporaryFile() as history_file:
            shadow_advisors.write_snapshot(snapshot_file.name, snap)
            shadow_advisors.append_history(history_file.name, snap)
            snapshot_file.seek(0)
            history_file.seek(0)
            self.assertIn(b'"shadow_only": true', snapshot_file.read())
            self.assertIn(b'"shadow_only": true', history_file.read())


if __name__ == "__main__":
    unittest.main()
