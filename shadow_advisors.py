"""Shadow specialist advisors for target-quality measurement.

The sector scout remains authoritative. These helpers only shape and persist
parallel specialist votes so future reviews can compare target outcomes against
asset-specific opinions.
"""
import datetime
import json


VERSION = "1.1"
DECISION_APPROVE = "approve"
DECISION_WATCH = "watch"
DECISION_REJECT = "reject"
RESPONSE_EXCERPT_LIMIT = 320


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


def build_prompt(symbol, category, tech_norm, source_context):
    """Specialist prompt. Deliberately excludes the scout's confidence and
    per-source score breakdown: the specialists are benchmarked against the
    scout, so showing them its answer would anchor their votes toward it and
    make the comparison measure agreement instead of independent signal."""
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
        f"Focus: {focus}\n\n"
        f"Evidence:\n{source_context}\n\n"
        "Judge independently from the evidence above. Return compact JSON only: "
        "{\"decision\":\"approve|watch|reject\",\"confidence\":0.0,"
        "\"reasoning\":\"...\",\"risk_flags\":[\"...\"]}"
    )


def structured_response_format():
    """LM Studio/OpenAI-compatible schema for grammar-constrained votes."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "shadow_specialist_vote",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": [DECISION_APPROVE, DECISION_WATCH, DECISION_REJECT],
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "reasoning": {"type": "string"},
                    "risk_flags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["decision", "confidence", "reasoning", "risk_flags"],
            },
        },
    }


def response_excerpt(raw_text, limit=RESPONSE_EXCERPT_LIMIT):
    """Return a single-line, bounded excerpt safe to persist in diagnostics."""
    printable = "".join(
        char if char.isprintable() else " " for char in str(raw_text or "")
    )
    compact = " ".join(printable.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def build_repair_prompt(raw_text):
    return (
        "Repair the malformed specialist response below. Preserve its intended vote and "
        "return only one JSON object with decision, confidence, reasoning, and risk_flags.\n\n"
        f"Malformed response:\n{response_excerpt(raw_text, limit=4000)}"
    )


def _extract_json(raw_text):
    if not isinstance(raw_text, str):
        raise TypeError("specialist response must be text")
    text = raw_text.strip().lstrip("\ufeff")
    if not text:
        raise ValueError("specialist response was empty")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as initial_error:
        decoder = json.JSONDecoder()
        first_object = None
        expected_keys = {"decision", "confidence", "reasoning", "risk_flags"}
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if not isinstance(candidate, dict):
                continue
            if first_object is None:
                first_object = candidate
            if expected_keys.intersection(candidate):
                return candidate
        if first_object is not None:
            return first_object
        raise ValueError(
            f"no valid JSON object found: {initial_error.msg} at char {initial_error.pos}"
        ) from initial_error

    if not isinstance(payload, dict):
        raise ValueError("specialist response must be a JSON object")
    return payload


def parse_vote(raw_text, symbol, category, tech_norm, scout_confidence):
    advisor = advisor_for(category, symbol)
    try:
        payload = _extract_json(raw_text)
    except Exception as error:
        return fallback_vote(symbol, category, tech_norm, scout_confidence,
                             "specialist_json_parse_failed", advisor_failed=True,
                             diagnostics={
                                 "parse_error": f"{type(error).__name__}: {error}",
                                 "raw_response_excerpt": response_excerpt(raw_text),
                             })

    raw_confidence = payload.get("confidence")
    missing_confidence = raw_confidence is None
    if missing_confidence:
        # No specialist number to use; fall back to the scout's, but flag it so
        # analysis can exclude these votes instead of mistaking them for
        # independent agreement.
        confidence = max(0.0, min(1.0, float(scout_confidence)))
    else:
        try:
            confidence = max(0.0, min(1.0, float(raw_confidence)))
        except (TypeError, ValueError):
            return fallback_vote(symbol, category, tech_norm, scout_confidence,
                                 "specialist_confidence_invalid", advisor_failed=True)
    decision = payload.get("decision") or decision_from_confidence(confidence)
    if decision not in {DECISION_APPROVE, DECISION_WATCH, DECISION_REJECT}:
        decision = decision_from_confidence(confidence)
    reasoning = payload.get("reasoning") or "Specialist returned no reasoning."
    flags = payload.get("risk_flags") or []
    if not isinstance(flags, list):
        flags = [str(flags)]
    if missing_confidence:
        flags = flags + ["specialist_confidence_missing"]
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
                  advisor_failed=False, diagnostics=None):
    confidence = max(0.0, min(1.0, float(scout_confidence)))
    vote = {
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
    if diagnostics:
        vote["diagnostics"] = diagnostics
    return vote


def build_snapshot(votes, updated=None):
    updated = updated or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    by_advisor = {}
    failures = 0
    for vote in votes:
        by_advisor.setdefault(vote["advisor"], {"approve": 0, "watch": 0, "reject": 0})
        by_advisor[vote["advisor"]][vote["decision"]] += 1
        if vote.get("advisor_failed"):
            failures += 1
    return {
        "version": VERSION,
        "status": "success",
        "updated": updated,
        "shadow_only": True,
        "summary": {
            "total_votes": len(votes),
            # A misconfigured SHADOW_ADVISOR_MODELS id fails every call and
            # leaves a plausible-looking file of fallback votes; this count
            # makes that unmissable.
            "advisor_failures": failures,
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
