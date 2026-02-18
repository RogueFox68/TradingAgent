# --- IMPORTS ---
import requests
import json
import os
import time
import datetime
import yfinance as yf
import subprocess
import sys
import config
from psaw import PushshiftAPI

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
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"
INPUT_FILE = "dragnet_candidates.json"
OUTPUT_FILE = "active_targets.json"
BEELINK_IP = "192.168.5.87"
BEELINK_USER = "trader"
BEELINK_PATH = "~/bots/repo/active_targets.json"
WEBHOOK_OVERSEER = getattr(config, 'WEBHOOK_OVERSEER') 

# --- REDDIT API ---
reddit_api = PushshiftAPI()
last_reddit_call = 0 

# --- CORE BACKUP (Unchanged) ---
CORE_WATCHLIST = {
    "condor_targets": ["SPY", "IWM", "QQQ"],
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
        except: pass
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
    Scrapes recent Reddit posts using Pushshift.
    Returns a summary string or None.
    """
    global last_reddit_call
    
    # Rate Limit (1s)
    elapsed = time.time() - last_reddit_call
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    last_reddit_call = time.time()

    try:
        # 7-day lookback
        after = int((datetime.datetime.now() - datetime.timedelta(days=7)).timestamp())
        
        # Strict Query: $TICKER to avoid generic word matches (e.g. $PUMP vs PUMP)
        query = f"${ticker}"
        
        # Search relevant subreddits
        submissions = reddit_api.search_submissions(
            q=query,
            subreddit="wallstreetbets,stocks,investing,options,thetagang,pennystocks",
            after=after,
            limit=50
        )
        
        mentions = []
        for post in submissions:
            # Filter low quality / spam
            if post.score < 5: continue 
            # Note: Pushshift might not always have up-to-date scores/karma, but we try.
            
            mentions.append({
                "title": post.title,
                "score": post.score,
                "sub": post.subreddit,
                "url": post.full_link
            })
            
        if not mentions: return None
        
        # Sort by engagement (Score)
        mentions.sort(key=lambda x: x['score'], reverse=True)
        
        # Take Top 3
        summary_lines = []
        for m in mentions[:3]:
            summary_lines.append(f"- [r/{m['sub']}] {m['title']} ({m['score']} pts)")
            
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
        stock = yf.Ticker(ticker)
        news = stock.news
        
        tiered_news = {
            "tier1": [],
            "tier2": [],
            "tier3": [] # Using Tier 3 bucket for Specialty + Industry + Unknowns
        }
        
        now = time.time()
        
        for n in news:
            # Recency Check (7 days)
            pub_time = n.get('providerPublishTime', 0)
            if (now - pub_time) > (168 * 3600): continue
            
            publisher = n.get('publisher', '')
            title = n.get('title', '')
            link = n.get('link', '')
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
    except:
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
    elif strategy in ["condor_targets", "wheel_targets"]:
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
            "model": MODEL_NAME, "prompt": system_prompt, "stream": False, "format": "json" 
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response_json = response.json()
        
        raw_text = response_json['response']
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
                           "Check 5080 network connection.",
                "username": "Sector Scout"
            })
    except: pass
    return False

def run_scout():
    print("--- 🔬 SECTOR SCOUT 4.1 (Segregated Targets) ---")
    candidates = get_candidates()
    final_targets = {
        "condor_targets": [], "wheel_targets": [],
        "trend_targets": [], "survivor_targets": [],
        "short_targets": [], "updated": str(datetime.datetime.now())
    }

    print("\n2. Deep Diving Candidates...")

    for category, tickers in candidates.items():
        if category == "updated" or not tickers: continue
        
        print(f"   👉 Analyzing {category}...")
        for item in tickers:
            if isinstance(item, dict):
                ticker = item.get('symbol')
                tech_score = item.get('tech_score', 50.0)
            else:
                ticker = item
                tech_score = 50.0 

            if "/" in ticker: continue
            
            # 1. Normalize Technical Score (Strategy-Aware)
            # Default scaling (0-100)
            tech_norm = min(max(tech_score / 100.0, 0.0), 1.0)
            
            if category == "condor_targets":
                # Condor scores are 0-20 (Low ADX). 20 is perfect.
                # Score 15 -> 1.0
                tech_norm = min(max(tech_score / 15.0, 0.0), 1.0)
                
            elif category == "wheel_targets":
                # Wheel scores are -5 to 10 (RSI 40-55). 10 is perfect.
                # Score 8 -> 0.8
                tech_norm = min(max(tech_score / 10.0, 0.0), 1.0)
                
            elif category == "survivor_targets":
                # Survivor scores are 10-50 (50-RSI). 30 (RSI 20) is perfect.
                # Score 30 -> 1.0
                tech_norm = min(max(tech_score / 30.0, 0.0), 1.0)

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
                # Reallocate weight to Tech
                weights[0] += 0.30 
                reasons.append("T1: N/A")

            # --- C. Mainstream News (20%) ---
            if news_map['tier2']:
                txt = "\n".join(news_map['tier2'][:3])
                s, r = ask_llama(ticker, category, txt, "tier2_news")
                scores.append(s)
                weights.append(0.20)
                reasons.append(f"T2: {s:.2f}")
            else:
                # Reallocate to Tech (or T1/T3 if we had complex logic, but simplifying)
                weights[0] += 0.20
                reasons.append("T2: N/A")

            # --- D. Specialty/Industry News (10%) ---
            if news_map['tier3']:
                txt = "\n".join(news_map['tier3'][:3])
                s, r = ask_llama(ticker, category, txt, "tier3_news")
                scores.append(s)
                weights.append(0.10)
                reasons.append(f"T3: {s:.2f}")
            else:
                weights[0] += 0.10
                reasons.append("T3: N/A")

            # --- E. Social/Reddit (10%) ---
            if reddit_text:
                s, r = ask_llama(ticker, category, reddit_text, "social")
                scores.append(s)
                weights.append(0.10)
                reasons.append(f"Soc: {s:.2f}")
            else:
                weights[0] += 0.10
                reasons.append("Soc: N/A")

            # 4. Calculate Weighted Final Score
            final_confidence = 0.0
            total_weight = sum(weights)
            
            if total_weight > 0:
                for i in range(len(scores)):
                    final_confidence += scores[i] * weights[i]
                
                # Normalize if re-allocation messed up sums (shouldn't, but safety)
                if abs(total_weight - 1.0) > 0.01:
                    final_confidence = final_confidence / total_weight
            
            is_approved = False
            # Threshold Check
            if final_confidence > 0.50: is_approved = True
            
            emoji = "✅" if is_approved else "❌"
            breakdown = " | ".join(reasons)
            print(f"      {emoji} {ticker:<4} | Conf: {final_confidence:>4.2f} [{breakdown}]")
            
            if is_approved:
                # Synthesize a master reason from available data
                master_reason = f"Tech Score: {tech_score} -> {tech_norm:.2f}. "
                if reddit_text: master_reason += f"Social: {s:.2f}. "
                
                final_targets[category].append({
                    "symbol": ticker,
                    "confidence": round(final_confidence, 2),
                    "reason": master_reason
                })


    total_analyzed = sum(len(v) for k, v in candidates.items() if k != "updated")
    total_approved = sum(len(v) for k, v in final_targets.items() if k != "updated")
    approval_rate = total_approved / total_analyzed if total_analyzed > 0 else 0

    all_confidences = []
    for category in final_targets:
        if category == "updated": continue
        for item in final_targets[category]:
            if isinstance(item, dict):
                all_confidences.append(item.get('confidence', 0))

    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

    print(f"\n📊 SCOUT SUMMARY:")
    print(f"   Analyzed: {total_analyzed}")
    print(f"   Approved: {total_approved} ({approval_rate*100:.0f}%)")
    print(f"   Avg Confidence: {avg_confidence:.2f}")

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
        except: pass

    print("\n3. Saving Results...")
    
    # [PHASE 2.5] Add Success Flag
    final_targets["status"] = "success"
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_targets, f, indent=4)
        
    beam_to_beelink()

if __name__ == "__main__":
    run_scout()