import requests
import json
import os
import time
import datetime
import yfinance as yf
import subprocess
import sys
import config

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

TRUSTED_SOURCES = ["Bloomberg", "Reuters", "WSJ", "CNBC", "Financial Times"]

def get_news_summary(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        recent_news = []
        now = time.time()
        
        for n in news:
            # Check recency
            pub_time = n.get('providerPublishTime', 0)
            age_hours = (now - pub_time) / 3600
            
            # [Revised] 7 Days (was 48h)
            if age_hours > 168:
                continue
            
            # Boost trusted sources
            publisher = n.get('publisher', '')
            if any(src in publisher for src in TRUSTED_SOURCES):
                recent_news.insert(0, n)
            else:
                recent_news.append(n)
        
        headlines = []
        for n in recent_news[:3]:
            headlines.append(f"- {n.get('title', '')}")
        
        return "\n".join(headlines) if headlines else "No recent news."
    except:
        return "No recent news found."

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

def ask_llama(ticker, strategy, headlines):
    if not headlines: return 0.0, "No Data"

    if strategy == "short_targets":
        role = "short seller"
        goal = "identifying weakness, bad earnings, or regulatory trouble"
        scoring = "High score (1.0) means the stock is likely to CRASH. Low score means it is strong/safe."
    
    elif strategy == "survivor_targets":
        role = "value investor"
        goal = "identifying if a recent price drop is an overreaction"
        scoring = "High score (1.0) means the DIP IS SAFE TO BUY. Low score means 'catching a falling knife'."
        
    elif strategy in ["condor_targets", "wheel_targets"]:
        role = "options income trader"
        goal = "identifying STABILITY and LACK of volatility"
        scoring = "High score (1.0) means the stock is BORING/FLAT. Low score means it is volatile/risky."
        
    else: 
        role = "growth investor"
        goal = "identifying breakouts, strong earnings, and momentum"
        scoring = "High score (1.0) means the stock is likely to RALLY."

    system_prompt = (
        f"You are a hedge fund {role}. Analyze {ticker} for the purpose of {goal}.\n"
        f"News:\n{headlines}\n\n"
        "Instructions:\n"
        f"1. {scoring}\n"
        "2. Return JSON: {\"score\": 0.85, \"reason\": \"Analysis...\"}"
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

            # 2. Get AI Opinion
            headlines = get_news_summary(ticker)
            
            # [Revised] Intelligent Fallback
            if "No recent news" in headlines:
                 # If no news, trust the technicals completely (especially for Condors)
                 llm_score = tech_norm
                 reason = "No recent news - relying on technical analysis"
            else:
                 llm_score, reason = ask_llama(ticker, category, headlines)
            
            # 3. Weighted Final Confidence
            # tech_norm is now properly scaled for the strategy
            final_confidence = (tech_norm * 0.4) + (llm_score * 0.6)
            
            is_approved = False
            # [Revised] Lower Threshold (was 0.60, then 0.50)
            if final_confidence > 0.50: is_approved = True
            
            emoji = "✅" if is_approved else "❌"
            print(f"      {emoji} {ticker:<4} | Conf: {final_confidence:>4.2f} (Tech: {tech_score:>4.1f} -> {tech_norm:.2f}, AI: {llm_score:.2f})")
            
            if is_approved:
                final_targets[category].append({
                    "symbol": ticker,
                    "confidence": round(final_confidence, 2),
                    "reason": reason
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