# --- IMPORTS ---
try:
    import requests
except ImportError:
    class _MissingRequests:
        def post(self, *args, **kwargs):
            raise RuntimeError("requests is not installed")

        def get(self, *args, **kwargs):
            raise RuntimeError("requests is not installed")
    requests = _MissingRequests()
import json
import os
import time
import datetime
try:
    import yfinance as yf
except ImportError:
    yf = None
import subprocess
import sys
try:
    import config
except ImportError:
    class _MissingConfig:
        WEBHOOK_OVERSEER = ""
    config = _MissingConfig()

import shadow_advisors

# Force UTF-8 Output for Windows Console
import builtins
def safe_print(*args, **kwargs):
    try:
        builtins.print(*args, **kwargs)
    except UnicodeEncodeError:
        try:
            encoding = sys.stdout.encoding or 'ascii'
            new_args = []
            for arg in args:
                if isinstance(arg, str):
                    new_args.append(arg.encode(encoding, errors='replace').decode(encoding))
                else:
                    new_args.append(arg)
            builtins.print(*new_args, **kwargs)
        except:
            pass 

print = safe_print

# --- CONFIGURATION ---
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "google/gemma-4-26b-a4b"
INPUT_FILE = "dragnet_candidates.json"
OUTPUT_FILE = "active_targets.json"
SHADOW_ADVISOR_FILE = getattr(config, 'SHADOW_ADVISOR_FILE', "shadow_advisor_votes.json")
SHADOW_ADVISOR_HISTORY_FILE = getattr(config, 'SHADOW_ADVISOR_HISTORY_FILE', "shadow_advisor_votes.jsonl")
ENABLE_SHADOW_ADVISORS = getattr(config, 'ENABLE_SHADOW_ADVISORS', True)
SHADOW_ADVISOR_MODELS = getattr(config, 'SHADOW_ADVISOR_MODELS', {})
# SCP target: overridable in config.py (gitignored) so infrastructure
# coordinates don't live in tracked source.
BEELINK_IP = getattr(config, 'BEELINK_IP', "192.168.5.87")
BEELINK_USER = getattr(config, 'BEELINK_USER', "trader")
BEELINK_PATH = getattr(config, 'BEELINK_PATH', "~/bots/repo/active_targets.json")
WEBHOOK_OVERSEER = getattr(config, 'WEBHOOK_OVERSEER', '')

# --- REDDIT CONFIG ---
REDDIT_SUBS = ["wallstreetbets", "stocks", "investing", "options", "thetagang"]
last_reddit_call = 0 

# --- CORE BACKUP (Unchanged) ---
CORE_WATCHLIST = {
    "wheel_targets": ["F", "PLTR", "SOFI", "AMD"],
    "trend_targets": ["NVDA", "TSLA", "COIN", "MSTR"],
    "survivor_targets": ["TQQQ", "SOXL", "UPRO"], 
    "short_targets": ["CVNA", "RIVN", "LCID"] 
}

def get_candidates():
    if os.path.exists(INPUT_FILE):
        try:
            file_time = os.path.getmtime(INPUT_FILE)
            if (time.time() - file_time) < 86400:
                with open(INPUT_FILE, 'r') as f:
                    data = json.load(f)
                    if "survivor_targets" not in data: data["survivor_targets"] = []
                    if "short_targets" not in data: data["short_targets"] = []
                    print(f"✅ Loaded Candidates: {len(data.get('trend_targets',[]))} Bull, {len(data.get('survivor_targets',[]))} Dip")
                    return data
            else:
                age_h = (time.time() - file_time) / 3600
                print(f"[!] {INPUT_FILE} is {age_h:.1f}h old (>24h) — "
                      f"the scanner has not produced a fresh dragnet. "
                      f"Falling back to CORE_WATCHLIST.")
        except Exception as e:
            # Was `except: pass`. A corrupt candidate file silently downgrading
            # the whole scan to the static watchlist is exactly the kind of
            # quiet degradation that hides an upstream outage for days.
            print(f"[!] Could not read {INPUT_FILE} ({type(e).__name__}: {e}) — "
                  f"falling back to CORE_WATCHLIST.")
    else:
        print(f"[!] {INPUT_FILE} missing — falling back to CORE_WATCHLIST.")
    return CORE_WATCHLIST

# --- NEWS SOURCE TIERS ---
TIER_1_ELITE = [
    "Bloomberg", "Reuters", "WSJ", "CNBC", "Financial Times"
]

TIER_2_MAINSTREAM = [
    "MarketWatch", "Barron's", "Investor's Business Daily", 
    "The Motley Fool", "Yahoo Finance", "Forbes", "Fortune"
]

TIER_3_SPECIALTY = [
    "Seeking Alpha", "TheStreet", "Benzinga", 
    "Business Insider", "TipRanks"
]

TIER_4_INDUSTRY = [
    "TechCrunch", "The Verge", "Ars Technica", 
    "BioSpace", "OilPrice.com", "Mining.com", "FiercePharma"
]

# Combined set for fast lookup
ALL_TRUSTED_SOURCES = set(TIER_1_ELITE + TIER_2_MAINSTREAM + TIER_3_SPECIALTY + TIER_4_INDUSTRY)

def get_reddit_sentiment(ticker):
    """
    Scrapes recent Reddit posts using Reddit's public JSON API.
    Returns a summary string or None.
    NO AUTH REQUIRED (Rate Limited).
    """
    global last_reddit_call
    
    mentions = []
    
    # We must treat the ticker carefully. $TICKER is safer.
    query = f"${ticker}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # Scan Top Subreddits (limit to 3 most relevant to save time/requests)
        for sub in ["wallstreetbets", "stocks", "investing"]:
            
            # Rate Limit (1.5s per request to be safe)
            elapsed = time.time() - last_reddit_call
            if elapsed < 1.5:
                time.sleep(1.5 - elapsed)
            last_reddit_call = time.time()
            
            url = f"https://www.reddit.com/r/{sub}/search.json"
            params = {
                "q": query,
                "restrict_sr": 1,
                "sort": "new",
                "limit": 10
            }
            
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                
                if resp.status_code == 429:
                    print(f"   [!] Reddit Rate Limit. Skipping {sub}...")
                    continue
                if resp.status_code != 200:
                    continue
                    
                data = resp.json()
                children = data.get("data", {}).get("children", [])
                
                for post in children:
                    p_data = post.get("data", {})
                    title = p_data.get("title", "")
                    score = p_data.get("score", 0)
                    url_link = p_data.get("permalink", "")
                    
                    # Score Filter (Noise Reduction)
                    if score < 5: continue
                    
                    mentions.append({
                        "title": title,
                        "score": score,
                        "sub": sub,
                        "url": f"https://reddit.com{url_link}"
                    })
            except Exception:
                continue # Skip sub on error
            
        if not mentions: return None
        
        # Sort by Score
        mentions.sort(key=lambda x: x['score'], reverse=True)
        
        # Take Top 3 Uniques
        seen_titles = set()
        summary_lines = []
        count = 0
        
        for m in mentions:
            if m['title'] in seen_titles: continue
            seen_titles.add(m['title'])
            
            summary_lines.append(f"- [r/{m['sub']}] {m['title']} ({m['score']} pts)")
            count += 1
            if count >= 3: break
            
        return "\n".join(summary_lines)

    except Exception as e:
        print(f"   [!] Reddit Error ({ticker}): {e}")
        return None

def get_tiered_news(ticker):
    """
    Fetches news from Yahoo and organizes into Tiers.
    Returns dict: {'tier1': [], 'tier2': [], 'tier3': []}
    """
    try:
        if yf is None:
            print(f"   [!] yfinance not installed; skipping news for {ticker}.")
            return {"tier1": [], "tier2": [], "tier3": []}
        stock = yf.Ticker(ticker)
        news = stock.news
        
        tiered_news = {
            "tier1": [],
            "tier2": [],
            "tier3": [] # Using Tier 3 bucket for Specialty + Industry + Unknowns
        }
        
        now = time.time()
        
        if not news:
            print(f"   [!] No news found for {ticker} from yfinance.")
            return tiered_news

        for n in news:
            if not n: continue
            
            # Handle new vs old structure
            info = n.get('content', n)
            if not info: continue
            
            # 1. Get Time
            pub_time = info.get('providerPublishTime', 0)
            if not pub_time and 'pubDate' in info:
                try:
                    # Parse ISO: 2026-02-18T14:43:06Z
                    dt = datetime.datetime.strptime(info['pubDate'], "%Y-%m-%dT%H:%M:%SZ")
                    # Convert to epoch
                    pub_time = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
                except Exception as e:
                    pass

            # Recency Check (7 days)
            age_hours = (now - pub_time) / 3600
            
            if (now - pub_time) > (168 * 3600): continue
            
            # 2. Get Publisher
            publisher = info.get('publisher', '')
            if not publisher:
                # Safe get for nested provider
                provider = info.get('provider') or {}
                publisher = provider.get('displayName', 'Unknown')
                
            title = info.get('title', '')
            link = info.get('link', '')
            if not link:
                 # Safe get for nested clickThroughUrl
                 ctu = info.get('clickThroughUrl') or {}
                 link = ctu.get('url', '')
                 
            item = f"- [{publisher}] {title}"
            
            # Bucketing
            if any(src in publisher for src in TIER_1_ELITE):
                tiered_news['tier1'].append(item)
            elif any(src in publisher for src in TIER_2_MAINSTREAM):
                tiered_news['tier2'].append(item)
            else:
                # Everyone else goes to Tier 3 (Specialty/Industry/Other)
                tiered_news['tier3'].append(item)
                
        return tiered_news
    except Exception as e:
        print(f"   [!] Error getting news for {ticker}: {e}")
        return {"tier1": [], "tier2": [], "tier3": []}

def validate_llm_response(score, reason, ticker):
    if not (0.0 <= score <= 1.0):
        print(f"   [!] {ticker}: Invalid score {score}, clamping to 0.0-1.0")
        score = max(0.0, min(1.0, score))
    
    if len(reason) < 50:
        print(f"   [!] {ticker}: Weak reasoning ({len(reason)} chars)")
        score = score * 0.7 
    
    if "insufficient" in reason.lower() or "not enough" in reason.lower():
        print(f"   [!] {ticker}: LLM confused, using 0.5")
        return 0.5, reason
    
    return score, reason

def ask_llama(ticker, strategy, content_text, source_type="news"):
    """
    source_type: 'tier1_news', 'tier2_news', 'social'
    """
    if not content_text: return 0.5, "Insufficient Data"

    if strategy == "short_targets":
        role = "short seller"
        goal = "identifying weakness, bad earnings, or regulatory trouble"
        scoring = "High score (1.0) means CRASH LIKELY. Low score (0.0) means STRONG/SAFE."
    elif strategy == "survivor_targets":
        role = "value investor"
        goal = "identifying if a recent price drop is an overreaction"
        scoring = "High score (1.0) means SAFE TO BUY. Low score (0.0) means FALLING KNIFE."
    elif strategy in ["wheel_targets"]:
        role = "options income trader"
        goal = "identifying STABILITY and LACK of volatility"
        scoring = "High score (1.0) means BORING/STABLE. Low score (0.0) means VOLATILE/RISKY."
    else: 
        role = "growth investor"
        goal = "identifying breakouts, strong earnings, and momentum"
        scoring = "High score (1.0) means RALLY LIKELY. Low score (0.0) means WEAKNESS."

    # Adjust perspective based on Source Type
    if source_type == "social":
        context = "Reddit/Social Media Sentiment"
        instruction = "Analyze the retail sentiment. Look for hype, panic, or irrational exuberance."
    else:
        context = "Financial News"
        instruction = "Analyze the fundamental and headline risks."

    system_prompt = (
        f"You are a hedge fund {role}. Analyze {ticker} based on this {context}.\n"
        f"Goal: {goal}\n\n"
        f"DATA:\n{content_text}\n\n"
        "Instructions:\n"
        f"1. {instruction}\n"
        f"2. {scoring}\n"
        "3. Return JSON: {\"score\": 0.85, \"reason\": \"Analysis...\"}"
    )

    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": system_prompt}],
            "max_tokens": 2048,
            "temperature": 0.1
        }
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=300)
        response_json = response.json()
        
        raw_text = response_json['choices'][0]['message']['content']
        try:
            analysis = json.loads(raw_text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                analysis = json.loads(match.group(0))
            else:
                return 0.0, "JSON Parse Failed"

        return validate_llm_response(analysis.get('score', 0.0), analysis.get('reason', 'N/A'), ticker)
    except Exception as e:
        print(f"   [!] AI Error on {ticker}: {e}")
        return 0.0, "AI Failed"

def _shadow_source_context(news_map, reddit_text):
    # Raw evidence only — the specialist never sees the scout's scores, so its
    # vote stays independent of the signal it's benchmarked against.
    lines = []
    for tier in ["tier1", "tier2", "tier3"]:
        items = (news_map or {}).get(tier, [])
        if items:
            lines.append(f"{tier.upper()}:\n" + "\n".join(items[:3]))
    if reddit_text:
        lines.append("SOCIAL:\n" + reddit_text)
    return "\n\n".join(lines) if lines else "No news or social coverage found."

def _request_shadow_completion(model, messages, max_tokens, temperature):
    payload = {
        "model": model,
        "messages": messages,
        "response_format": shadow_advisors.structured_response_format(),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = requests.post(LM_STUDIO_URL, json=payload, timeout=300)
    response.raise_for_status()
    response_json = response.json()
    choice = response_json["choices"][0]
    raw_text = choice.get("message", {}).get("content") or ""
    return raw_text, choice.get("finish_reason")

def _shadow_attempt_diagnostic(number, finish_reason, vote):
    attempt = {
        "attempt": number,
        "finish_reason": finish_reason or "unknown",
    }
    parse_diagnostics = vote.get("diagnostics") or {}
    for key in ["parse_error", "raw_response_excerpt"]:
        if parse_diagnostics.get(key):
            attempt[key] = parse_diagnostics[key]
    return attempt

def _attach_shadow_diagnostics(vote, model, attempts, api_error=None):
    diagnostics = {
        "model": model,
        "attempt_count": len(attempts),
        "retry_used": len(attempts) > 1,
        "recovered_after_retry": len(attempts) > 1 and not vote.get("advisor_failed"),
        "attempts": attempts,
    }
    if api_error is not None:
        diagnostics["api_error"] = shadow_advisors.response_excerpt(
            f"{type(api_error).__name__}: {api_error}", limit=240)
    vote["diagnostics"] = diagnostics
    return vote

def ask_shadow_advisor(ticker, category, tech_norm, final_confidence,
                       news_map, reddit_text):
    """Ask the asset-specialist shadow advisor. Never affects target approval."""
    context = _shadow_source_context(news_map, reddit_text)
    advisor_name = shadow_advisors.advisor_for(category, ticker)
    model = SHADOW_ADVISOR_MODELS.get(advisor_name, MODEL_NAME) if isinstance(SHADOW_ADVISOR_MODELS, dict) else MODEL_NAME
    prompt = shadow_advisors.build_prompt(ticker, category, tech_norm, context)
    attempts = []
    try:
        raw_text, finish_reason = _request_shadow_completion(
            model,
            [{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.05,
        )
        vote = shadow_advisors.parse_vote(
            raw_text, ticker, category, tech_norm, final_confidence)
        attempts.append(_shadow_attempt_diagnostic(1, finish_reason, vote))

        parse_failed = (vote.get("advisor_failed")
                        and vote.get("reasoning") == "specialist_json_parse_failed")
        # an empty response has no intended vote to repair; a schema-forced
        # retry would just fabricate one
        if parse_failed and shadow_advisors.response_excerpt(raw_text):
            repair_prompt = shadow_advisors.build_repair_prompt(raw_text)
            raw_text, finish_reason = _request_shadow_completion(
                model,
                [{"role": "user", "content": repair_prompt}],
                max_tokens=512,
                temperature=0.0,
            )
            vote = shadow_advisors.parse_vote(
                raw_text, ticker, category, tech_norm, final_confidence)
            attempts.append(_shadow_attempt_diagnostic(2, finish_reason, vote))

        return _attach_shadow_diagnostics(vote, model, attempts)
    except Exception as e:
        print(f"   [!] Shadow advisor failed on {ticker}: {e}")
        attempts.append({
            "attempt": len(attempts) + 1,
            "finish_reason": "request_failed",
            "api_error": shadow_advisors.response_excerpt(
                f"{type(e).__name__}: {e}", limit=240),
        })
        vote = shadow_advisors.fallback_vote(
            ticker, category, tech_norm, final_confidence,
            f"shadow_advisor_failed: {e}", advisor_failed=True)
        return _attach_shadow_diagnostics(vote, model, attempts, api_error=e)

def beam_to_beelink(retries=3):
    print(f"\n4. Beaming {OUTPUT_FILE} to Beelink...")
    
    for attempt in range(retries):
        try:
            cmd = f"scp {OUTPUT_FILE} {BEELINK_USER}@{BEELINK_IP}:{BEELINK_PATH}"
            subprocess.run(cmd, shell=True, check=True, timeout=30,  stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            print(f"   ✅ Transfer Complete (Attempt {attempt+1}).")
            return True
        except subprocess.TimeoutExpired:
             print(f"   ⚠️ SCP Timeout (Attempt {attempt+1})...")
        except Exception as e:
             print(f"   ⚠️ SCP Failed (Attempt {attempt+1}): {e}")
        time.sleep(5) 

    print(f"   🚨 ALL SCP ATTEMPTS FAILED. Using Fallback.")
    
    try:
        if WEBHOOK_OVERSEER:
            requests.post(WEBHOOK_OVERSEER, json={
                "content": "🚨 **SCP TRANSFER FAILED**\n"
                           "Targets not updated on Beelink.\n"
                           "Check Corsair→Beelink network connection.",
                "username": "Sector Scout"
            })
    except: pass
    return False

def run_scout():
    print("--- 🔬 SECTOR SCOUT 4.1 (Segregated Targets) ---")
    candidates = get_candidates()
    final_targets = {
        "version": "1.1",
        "status": "success",
        "updated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wheel_targets": {},
        "trend_targets": {}, 
        "survivor_targets": {},
        "short_targets": {}
    }
    shadow_votes = []

    print("\n2. Deep Diving Candidates...")

    for category, tickers in candidates.items():
        if category not in final_targets or category in ["updated", "version", "status"]: continue
        if not tickers: continue
        
        print(f"   👉 Analyzing {category}...")
        for item in tickers:
            if isinstance(item, dict):
                ticker = item.get('symbol')
                tech_score = item.get('tech_score', 50.0)
            else:
                ticker = item
                tech_score = 50.0 

            if "/" in ticker:
                if ENABLE_SHADOW_ADVISORS:
                    shadow_votes.append(shadow_advisors.fallback_vote(
                        ticker, category, 0.50, 0.50,
                        "crypto candidate observed; scout does not emit crypto targets yet"))
                continue
            
            # 1. Normalize Technical Score (Strategy-Aware)
            # Default scaling (0-100)
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
                tech_norm = min(max(tech_score / 100.0, 0.0), 1.0)

            # 2. Gather Intelligence
            news_map = get_tiered_news(ticker)
            reddit_text = get_reddit_sentiment(ticker)
            
            # 3. Multi-Factor Scoring
            scores = []
            weights = []
            reasons = []

            # --- A. Technicals (30%) ---
            scores.append(tech_norm)
            weights.append(0.30)
            reasons.append(f"Tech: {tech_norm:.2f}")

            # --- B. Elite News (30%) ---
            if news_map['tier1']:
                txt = "\n".join(news_map['tier1'][:3])
                s, r = ask_llama(ticker, category, txt, "tier1_news")
                scores.append(s)
                weights.append(0.30)
                reasons.append(f"T1: {s:.2f}")
            else:
                scores.append(0.50)
                weights.append(0.30)
                reasons.append("T1: N/A")

            # --- C. Mainstream News (20%) ---
            if news_map['tier2']:
                txt = "\n".join(news_map['tier2'][:3])
                s, r = ask_llama(ticker, category, txt, "tier2_news")
                scores.append(s)
                weights.append(0.20)
                reasons.append(f"T2: {s:.2f}")
            else:
                scores.append(0.50)
                weights.append(0.20)
                reasons.append("T2: N/A")

            # --- D. Specialty/Industry News (10%) ---
            if news_map['tier3']:
                txt = "\n".join(news_map['tier3'][:3])
                s, r = ask_llama(ticker, category, txt, "tier3_news")
                scores.append(s)
                weights.append(0.10)
                reasons.append(f"T3: {s:.2f}")
            else:
                scores.append(0.50)
                weights.append(0.10)
                reasons.append("T3: N/A")

            # --- E. Social/Reddit (10%) ---
            social_score = None
            if reddit_text:
                s, r = ask_llama(ticker, category, reddit_text, "social")
                social_score = s
                scores.append(s)
                weights.append(0.10)
                reasons.append(f"Soc: {s:.2f}")
            else:
                scores.append(0.50)
                weights.append(0.10)
                reasons.append("Soc: N/A")

            # 4. Calculate Weighted Final Score
            final_confidence = 0.0
            total_weight = sum(weights)
            
            if total_weight > 0:
                for i in range(len(scores)):
                    final_confidence += scores[i] * weights[i]
            
            is_approved = False
            # Threshold Check
            if final_confidence >= 0.66: is_approved = True
            
            emoji = "✅" if is_approved else "❌"
            breakdown = " | ".join(reasons)
            print(f"      {emoji} {ticker:<4} | Conf: {final_confidence:>4.2f} [{breakdown}]")

            if ENABLE_SHADOW_ADVISORS:
                shadow_votes.append(ask_shadow_advisor(
                    ticker, category, tech_norm, final_confidence,
                    news_map, reddit_text))
            
            if is_approved:
                # Synthesize a master reason from available data
                master_reason = f"Tech Score: {tech_score} -> {tech_norm:.2f}. "
                if social_score is not None: master_reason += f"Social: {social_score:.2f}. "
                
                final_targets[category][ticker] = {
                    "confidence": round(final_confidence, 2),
                    "reason": master_reason
                }


    total_analyzed = sum(len(v) for k, v in candidates.items() if isinstance(v, list))
    total_approved = sum(len(v) for k, v in final_targets.items() if isinstance(v, dict) and k.endswith('_targets'))
    approval_rate = total_approved / total_analyzed if total_analyzed > 0 else 0

    all_confidences = []
    for category, data in final_targets.items():
        if not isinstance(data, dict) or not category.endswith('_targets'): continue
        for ticker, item in data.items():
            all_confidences.append(item.get('confidence', 0))

    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

    print(f"\n📊 SCOUT SUMMARY:")
    print(f"   Analyzed: {total_analyzed}")
    print(f"   Approved: {total_approved} ({approval_rate*100:.0f}%)")
    print(f"   Avg Confidence: {avg_confidence:.2f}")
    if ENABLE_SHADOW_ADVISORS:
        shadow_snapshot = shadow_advisors.build_snapshot(
            shadow_votes, updated=final_targets["updated"])
        shadow_advisors.write_snapshot(SHADOW_ADVISOR_FILE, shadow_snapshot)
        shadow_advisors.append_history(SHADOW_ADVISOR_HISTORY_FILE, shadow_snapshot)
        shadow_failures = shadow_snapshot["summary"]["advisor_failures"]
        fail_note = f" ({shadow_failures} failed)" if shadow_failures else ""
        print(f"   Shadow Votes: {len(shadow_votes)}{fail_note} -> {SHADOW_ADVISOR_FILE}")
        if shadow_failures == len(shadow_votes) and shadow_votes:
            print("   [!] Every shadow vote failed — check SHADOW_ADVISOR_MODELS ids "
                  "against the models loaded in LM Studio.")

    if WEBHOOK_OVERSEER:
        try:
            if total_approved == 0:
                 requests.post(WEBHOOK_OVERSEER, json={
                    "content": (
                        f"⚠️ **SCOUT ALERT: 0 TARGETS**\n"
                        f"Analysis complete but nothing approved.\n"
                        f"Bots will STAND BY (No fallback).\n"
                        f"Avg Confidence: {avg_confidence:.2f}"
                    ),
                    "username": "Sector Scout"
                })
            else:
                requests.post(WEBHOOK_OVERSEER, json={
                    "content": (
                        f"📊 **SCOUT COMPLETE**\n"
                        f"Analyzed: {total_analyzed}\n"
                        f"Approved: {total_approved} ({approval_rate*100:.0f}%)\n"
                        f"Avg Confidence: {avg_confidence:.2f}"
                    ),
                    "username": "Sector Scout"
                })
        except Exception as e:
            print(f"[!] Scout summary webhook failed: {e}")

    print("\n3. Saving Results...")

    # Publish guard. `total_approved == 0` is a LEGITIMATE outcome — the model
    # rejecting everything in a bad tape is a real signal, and the bots standing
    # by is the correct response to it. `total_analyzed == 0` is not: it means
    # there was nothing to analyse, i.e. the scanner produced nothing and even
    # CORE_WATCHLIST came back empty. Writing that would SCP an empty file over
    # the Beelink's good targets, and the fleet's 24h staleness check cannot
    # catch it because the file it receives is fresh — just empty.
    if total_analyzed == 0:
        msg = (f"Scout had 0 candidates to analyse. Refusing to write and beam "
               f"{OUTPUT_FILE} — the Beelink keeps its previous targets "
               f"(which will age into a STALE TARGETS alert if this persists).")
        print(f"[!] PUBLISH ABORTED: {msg}")
        if WEBHOOK_OVERSEER:
            try:
                requests.post(WEBHOOK_OVERSEER, json={
                    "content": f"🚨 **SCOUT PUBLISH ABORTED**\n{msg}",
                    "username": "Sector Scout"}, timeout=10)
            except Exception as e:
                print(f"[!] Abort alert failed: {e}")
        sys.exit(1)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_targets, f, indent=4)
        
    beam_to_beelink()

if __name__ == "__main__":
    run_scout()
