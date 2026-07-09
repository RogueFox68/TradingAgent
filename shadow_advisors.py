"""Shadow specialist advisors for target-quality measurement.

The sector scout remains authoritative. These helpers only shape and persist
parallel specialist votes so future reviews can compare target outcomes against
asset-specific opinions.
"""
import datetime
import json
import re


VERSION = "1.0"
DECISION_APPROVE = "approve"
DECISION_WATCH = "watch"
DECISION_REJECT = "reject"


def advisor_for(category, symbol):
    if "/" in (symbol or "") or category in {"crypto_targets", "moon_targets"}:
        return "crypto_specialist"
    if category == "wheel_targets":
        return "options_specialist"
    return "equity_specialist"


def decision_from_confidence(confidence):
    confidence = max(0.0, min(1.0, float(confidence)))
    if confidence >= 0.66:
        return DECISION_APPROVE
    if confidence <= 0.45:
        return DECISION_REJECT
    return DECISION_WATCH


def risk_flags(category, symbol, confidence, tech_norm, reasoning):
    flags = []
    if confidence < 0.55:
        flags.append("low_specialist_confidence")
    if tech_norm < 0.40:
        flags.append("weak_technical_base")
    if category == "wheel_targets" and confidence < 0.66:
        flags.append("premium_candidate_needs_review")
    if "/" in (symbol or ""):
        flags.append("crypto_shadow_only")
    if "insufficient" in (reasoning or "").lower() or "not enough" in (reasoning or "").lower():
        flags.append("thin_information")
    return flags


def build_prompt(symbol, category, tech_norm, scout_confidence, source_context):
    advisor = advisor_for(category, symbol)
    if advisor == "options_specialist":
        role = "options income risk analyst"
        focus = "assignment risk, volatility expansion, liquidity, and boring stability"
    elif advisor == "crypto_specialist":
        role = "crypto market structure analyst"
        focus = "trend quality, volatility regime, exchange liquidity, and downside flush risk"
    else:
        role = "equity long/short analyst"
        focus = "trend quality, mean-reversion safety, catalyst durability, and downside risk"

    return (
        f"You are a shadow {role}. Your vote is logged for analysis only and must not "
        f"authorize trades.\n"
        f"Asset: {symbol}\n"
        f"Strategy bucket: {category}\n"
        f"Technical score: {tech_norm:.2f}\n"
        f"Current scout confidence: {scout_confidence:.2f}\n"
        f"Focus: {focus}\n\n"
        f"Evidence:\n{source_context}\n\n"
        "Return compact JSON only: "
        "{\"decision\":\"approve|watch|reject\",\"confidence\":0.0,"
        "\"reasoning\":\"...\",\"risk_flags\":[\"...\"]}"
    )


def _extract_json(raw_text):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text or "", re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def parse_vote(raw_text, symbol, category, tech_norm, scout_confidence):
    advisor = advisor_for(category, symbol)
    try:
        payload = _extract_json(raw_text)
    except Exception:
        return fallback_vote(symbol, category, tech_norm, scout_confidence,
                             "specialist_json_parse_failed", advisor_failed=True)

    confidence = max(0.0, min(1.0, float(payload.get("confidence", scout_confidence))))
    decision = payload.get("decision") or decision_from_confidence(confidence)
    if decision not in {DECISION_APPROVE, DECISION_WATCH, DECISION_REJECT}:
        decision = decision_from_confidence(confidence)
    reasoning = payload.get("reasoning") or "Specialist returned no reasoning."
    flags = payload.get("risk_flags") or []
    if not isinstance(flags, list):
        flags = [str(flags)]
    flags = list(dict.fromkeys(flags + risk_flags(category, symbol, confidence, tech_norm, reasoning)))

    return {
        "symbol": symbol,
        "strategy_bucket": category,
        "advisor": advisor,
        "decision": decision,
        "confidence": round(confidence, 3),
        "reasoning": reasoning,
        "risk_flags": flags,
        "scout_confidence": round(float(scout_confidence), 3),
        "tech_score_norm": round(float(tech_norm), 3),
        "shadow_only": True,
        "advisor_failed": False,
    }


def fallback_vote(symbol, category, tech_norm, scout_confidence, reason,
                  advisor_failed=False):
    confidence = max(0.0, min(1.0, float(scout_confidence)))
    return {
        "symbol": symbol,
        "strategy_bucket": category,
        "advisor": advisor_for(category, symbol),
        "decision": decision_from_confidence(confidence),
        "confidence": round(confidence, 3),
        "reasoning": reason,
        "risk_flags": risk_flags(category, symbol, confidence, tech_norm, reason),
        "scout_confidence": round(float(scout_confidence), 3),
        "tech_score_norm": round(float(tech_norm), 3),
        "shadow_only": True,
        "advisor_failed": advisor_failed,
    }


def build_snapshot(votes, updated=None):
    updated = updated or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    by_advisor = {}
    for vote in votes:
        by_advisor.setdefault(vote["advisor"], {"approve": 0, "watch": 0, "reject": 0})
        by_advisor[vote["advisor"]][vote["decision"]] += 1
    return {
        "version": VERSION,
        "status": "success",
        "updated": updated,
        "shadow_only": True,
        "summary": {
            "total_votes": len(votes),
            "by_advisor": by_advisor,
        },
        "votes": votes,
    }


def write_snapshot(path, snapshot):
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=4)


def append_history(path, snapshot):
    with open(path, "a") as f:
        f.write(json.dumps(snapshot) + "\n")
