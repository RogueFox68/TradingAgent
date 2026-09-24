import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import sector_scout_3


def normalize(tech_score, category):
    # Default scaling (0-100) -> 0-1
    tech_norm = min(max(tech_score / 100.0, 0.0), 1.0)
    
    if category in ["trend_targets", "short_targets"]:
        # ADX-based scores. Target > 40.0
        tech_norm = min(max(tech_score / 40.0, 0.0), 1.0)
    elif category == "wheel_targets":
        # Wheel scores are -5 to 10 (RSI 40-55). 10 is perfect.
        # Score 8 -> 0.8
        tech_norm = min(max(tech_score / 10.0, 0.0), 1.0)
        
    elif category == "survivor_targets":
        # Survivor scores are 10-50 (50-RSI). 30 (RSI 20) is perfect.
        # Score 30 -> 1.0
        tech_norm = min(max(tech_score / 30.0, 0.0), 1.0)
    else:
        # Fallback for unmapped categories
        tech_norm = min(max(tech_score / 100.0, 0.0), 1.0)
        
    return tech_norm

def test_weighted_scoring():
    print("\n--- Testing Weighted Scoring (Mock) ---")
    print(f"{'Scenario':<20} | {'Tech':<6} | {'T1':<6} | {'T2':<6} | {'Soc':<6} | {'Final':<6} | {'Expected':<8}")
    print("-" * 75)

    scenarios = [
        # 1. All Data Present (0.3, 0.3, 0.2, 0.1, 0.1)
        {"tech": 0.5, "t1": 0.5, "t2": 0.5, "t3": 0.5, "soc": 0.5, "exp": 0.50},
        
        # 2. Tech Only (No reallocation, neutral missing)
        {"tech": 0.8, "t1": None, "t2": None, "t3": None, "soc": None, "exp": 0.59},
        
        # 3. Tech + Elite News (Rest missing)
        {"tech": 0.5, "t1": 0.8, "t2": None, "t3": None, "soc": None, "exp": 0.59},
        
        # 4. Full Bullish 
        {"tech": 1.0, "t1": 1.0, "t2": 1.0, "t3": 1.0, "soc": 1.0, "exp": 1.00},
    ]

    for s in scenarios:
        tech = s["tech"]
        t1 = s["t1"]
        t2 = s["t2"]
        t3 = s["t3"]
        soc = s["soc"]
        
        scores = []
        weights = []
        
        scores.append(tech)
        weights.append(0.30)
        
        if t1 is not None:
            scores.append(t1)
            weights.append(0.30)
        else:
            scores.append(0.50)
            weights.append(0.30)
            
        if t2 is not None:
            scores.append(t2)
            weights.append(0.20)
        else:
            scores.append(0.50)
            weights.append(0.20)
            
        if t3 is not None:
            scores.append(t3)
            weights.append(0.10)
        else:
            scores.append(0.50)
            weights.append(0.10)
            
        if soc is not None:
            scores.append(soc)
            weights.append(0.10)
        else:
            scores.append(0.50)
            weights.append(0.10)
            
        final = 0.0
        for i in range(len(scores)):
            final += scores[i] * weights[i]
            
        assert abs(final - s["exp"]) < 0.01, f"Weighted scoring failed: {final} != {s['exp']}"

def test_normalize():
    tests = [
        ("trend_targets", 20, 0.50),   # 20/40
        ("trend_targets", 40, 1.00),   # 40/40
        ("short_targets", 10, 0.25),   # 10/40
        ("wheel_targets", 5, 0.50),    # 5/10
        ("wheel_targets", 12, 1.00),   # Cap at 1.0
        ("survivor_targets", 15, 0.50),# 15/30
        ("survivor_targets", 30, 1.00),# 30/30
        ("unknown_category", 50, 0.50) # Fallback 50/100
    ]

    for cat, score, expected in tests:
        result = normalize(score, cat)
        assert abs(result - expected) < 0.05, f"{cat} failed: {result} != {expected}"

# --- run_scout with LM Studio failing -----------------------------------------

FAIL = object()  # an LLM call that LM Studio answers with HTTP 400
LABEL_TO_TIER = {"T1": "tier1", "T2": "tier2", "T3": "tier3"}
# >= 50 chars, so validate_llm_response applies no weak-reasoning penalty.
REASON = "Steady fundamentals with no headline risk in the recent coverage window."
WEBHOOK = "https://discord.invalid/webhook"


class RunScoutLLMFailureTest(unittest.TestCase):
    """Drives the real run_scout -> ask_llama -> publish path, with only
    requests.post, the data sources and the SCP step stubbed out.

    A candidate is (ticker, raw tech score, {source label: LLM outcome}). A
    label that is absent has no coverage; FAIL makes LM Studio answer that
    call with HTTP 400 and no 'choices', the 09-22 outage shape.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.output = os.path.join(self.tmp, "active_targets.json")
        self.webhooks = []
        # Prompts the fake could not route. Recorded, not asserted, inside the
        # fake: ask_llama catches every Exception, so an assertion there would
        # just look like one more failed LLM call.
        self.unrouted = []

    def _data(self, ticker, label):
        return f"- [Wire] {ticker} {label} coverage"

    def _post(self, outcomes):
        def post(url, json=None, timeout=None, **kwargs):
            response = MagicMock()
            if url != sector_scout_3.LM_STUDIO_URL:
                self.webhooks.append(json["content"])
                response.status_code = 204
                return response
            prompt = json["messages"][0]["content"]
            hits = [v for text, v in outcomes.items() if text in prompt]
            if len(hits) != 1:
                self.unrouted.append(prompt)
                raise RuntimeError("test fake could not route this prompt")
            if hits[0] is FAIL:
                response.status_code = 400
                response.json.return_value = {"error": "No models loaded. Please load a model."}
            else:
                response.status_code = 200
                response.json.return_value = {"choices": [{"message": {
                    "content": f'{{"score": {hits[0]}, "reason": "{REASON}"}}'}}]}
            return response
        return post

    def _run(self, candidates):
        outcomes, news, social = {}, {}, {}
        for ticker, _, sources in candidates:
            news[ticker] = {"tier1": [], "tier2": [], "tier3": []}
            social[ticker] = None
            for label, outcome in sources.items():
                text = self._data(ticker, label)
                outcomes[text] = outcome
                if label == "Soc":
                    social[ticker] = text
                else:
                    news[ticker][LABEL_TO_TIER[label]].append(text)
        dragnet = {"trend_targets": [{"symbol": t, "tech_score": tech}
                                     for t, tech, _ in candidates]}
        self.beam = MagicMock(return_value=True)
        out = io.StringIO()
        with patch.object(sector_scout_3, "get_candidates", return_value=dragnet), \
             patch.object(sector_scout_3, "get_tiered_news", side_effect=news.__getitem__), \
             patch.object(sector_scout_3, "get_reddit_sentiment", side_effect=social.__getitem__), \
             patch.object(sector_scout_3.requests, "post", side_effect=self._post(outcomes)), \
             patch.object(sector_scout_3, "beam_to_beelink", self.beam), \
             patch.object(sector_scout_3, "ENABLE_SHADOW_ADVISORS", False), \
             patch.object(sector_scout_3, "WEBHOOK_OVERSEER", WEBHOOK), \
             patch.object(sector_scout_3, "OUTPUT_FILE", self.output), \
             contextlib.redirect_stdout(out):
            try:
                sector_scout_3.run_scout()
                code = None
            except SystemExit as e:
                code = e.code
        self.assertEqual(self.unrouted, [])
        return code, out.getvalue()

    def _published(self):
        with open(self.output) as f:
            return json.load(f)

    def test_failed_call_counts_as_missing_not_bearish(self):
        # AAA: tech 1.0, T1 0.7, T2 fails. As missing (0.5):
        # .30*1.0 + .30*.7 + .20*.5 + .10*.5 + .10*.5 = 0.71 -> approved.
        # Scored 0.0, as before, it was 0.61 -> rejected.
        code, out = self._run([
            ("AAA", 40.0, {"T1": 0.7, "T2": FAIL}),
            ("BBB", 40.0, {"T1": 0.9, "T2": 0.9}),
            ("CCC", 20.0, {"T1": 0.2, "T2": 0.2}),
        ])
        self.assertIsNone(code, out)
        published = self._published()
        self.assertEqual(published["status"], "success")
        self.assertAlmostEqual(published["trend_targets"]["AAA"]["confidence"], 0.71)
        self.assertIn("BBB", published["trend_targets"])
        self.assertNotIn("CCC", published["trend_targets"])
        self.beam.assert_called_once()
        self.assertIn("T2: N/A | T3: N/A | Soc: N/A] (LLM failed: T2)", out)
        self.assertIn("LLM Calls: 6 (1 failed)", out)

    def test_llm_outage_aborts_and_keeps_previous_targets(self):
        # The 09-22 runs: every call fails. This used to publish an empty
        # "success" file that the fleet reads as a deliberate stand-by.
        with open(self.output, "w") as f:
            f.write('{"previous": "targets"}')
        code, out = self._run([
            ("AAA", 40.0, {"T1": FAIL, "Soc": FAIL}),
            ("BBB", 40.0, {"T1": FAIL, "T2": FAIL}),
            ("CCC", 20.0, {"T3": FAIL}),
        ])
        self.assertEqual(code, 1, out)
        self.assertEqual(self._published(), {"previous": "targets"})
        self.beam.assert_not_called()
        self.assertEqual(len(self.webhooks), 1, self.webhooks)
        self.assertIn("SCOUT PUBLISH ABORTED", self.webhooks[0])
        self.assertIn("5/5 LLM calls failed", self.webhooks[0])
        self.assertIn("No models loaded", self.webhooks[0])
        # The abort is the only alert. "0 TARGETS, bots will STAND BY" would
        # be false: the bots keep the previous file.
        self.assertNotIn("0 TARGETS", "".join(self.webhooks))

    def test_majority_failure_aborts_even_with_approvals(self):
        # The 09-17 12:00 shape: most calls failed but a few candidates still
        # cleared the threshold on the calls that worked. 6 of 10 failed.
        code, out = self._run([
            ("AAA", 40.0, {"T1": 0.9, "T2": 0.9, "T3": 0.9, "Soc": 0.9}),
            ("BBB", 40.0, {"T1": FAIL, "T2": FAIL, "T3": FAIL}),
            ("CCC", 40.0, {"T1": FAIL, "T2": FAIL, "T3": FAIL}),
        ])
        self.assertEqual(code, 1, out)
        self.assertIn("✅ AAA", out)
        self.assertFalse(os.path.exists(self.output))
        self.beam.assert_not_called()
        self.assertIn("6/10 LLM calls failed", self.webhooks[0])

    def test_healthy_llm_rejecting_everything_still_publishes(self):
        # 0 approved from a WORKING model is a real signal. The bots standing
        # by is the right response, so this must still publish.
        code, out = self._run([
            ("AAA", 40.0, {"T1": 0.1, "T2": 0.1}),
            ("BBB", 20.0, {"T1": 0.1, "Soc": 0.1}),
        ])
        self.assertIsNone(code, out)
        published = self._published()
        self.assertEqual(published["status"], "success")
        self.assertEqual(published["trend_targets"], {})
        self.beam.assert_called_once()
        self.assertIn("0 TARGETS", "".join(self.webhooks))

    def test_no_candidates_still_aborts(self):
        code, out = self._run([])
        self.assertEqual(code, 1, out)
        self.assertFalse(os.path.exists(self.output))
        self.beam.assert_not_called()
        self.assertIn("0 candidates", self.webhooks[0])


class PublishAbortReasonTest(unittest.TestCase):
    def test_threshold_is_more_than_half_of_calls(self):
        self.assertIsNone(sector_scout_3.publish_abort_reason(40, 10, 5))
        self.assertIsNotNone(sector_scout_3.publish_abort_reason(40, 10, 6))

    def test_a_run_with_no_llm_calls_is_not_an_llm_outage(self):
        # Nothing was asked, so nothing failed. The scout's no-coverage
        # scoring already holds every such candidate below the threshold.
        self.assertIsNone(sector_scout_3.publish_abort_reason(40, 0, 0))

    def test_nothing_to_analyse_aborts(self):
        self.assertIn("0 candidates", sector_scout_3.publish_abort_reason(0, 0, 0))


if __name__ == "__main__":
    test_weighted_scoring()
    test_normalize()
    unittest.main()
