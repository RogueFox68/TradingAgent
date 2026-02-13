import requests
import json
import os
import time
import datetime
import yfinance as yf
import subprocess

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1"
INPUT_FILE = "dragnet_candidates.json"
OUTPUT_FILE = "active_targets.json"
BEELINK_IP = "192.168.5.87"
BEELINK_USER = "trader"
BEELINK_PATH = "~/bots/repo/active_targets.json"

# --- CORE BACKUP ---
CORE_WATCHLIST = {
    "condor_targets": ["SPY", "IWM", "QQQ"],
    "wheel_targets": ["F", "PLTR", "SOFI", "AMD"],
    "trend_targets": ["NVDA", "TSLA", "COIN", "MSTR"],
    "survivor_targets": ["TQQQ", "SOXL", "UPRO"], # [NEW]
    "short_targets": ["CVNA", "RIVN", "LCID"] 
}

def get_candidates():
    if os.path.exists(INPUT_FILE):
        try:
            file_time = os.path.getmtime(INPUT_FILE)
            if (time.time() - file_time) < 86400:
                with open(INPUT_FILE, 'r') as f:
                    data = json.load(f)
                    # Ensure keys exist
                    if "survivor_targets" not in data: data["survivor_targets"] = []
                    if "short_targets" not in data: data["short_targets"] = []
                    
                    print(f"✅ Loaded Candidates: {len(data.get('trend_targets',[]))} Bull, {len(data.get('survivor_targets',[]))} Dip")
                    return data
        except: pass
    return CORE_WATCHLIST

def get_news_summary(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        headlines = []
        for n in news[:3]:
            headlines.append(f"- {n.get('title', '')}")
        return "\n".join(headlines)
    except: return "No recent news found."

def ask_llama(ticker, strategy, headlines):
    """
    The Brain: Context-Aware Analysis.
    """
    if not headlines: return 0.0, "No Data"

    # --- DYNAMIC PROMPTING ---
    if strategy == "short_targets":
        role = "short seller"
        goal = "identifying weakness, bad earnings, or regulatory trouble"
        scoring = "High score (1.0) means the stock is likely to CRASH. Low score means it is strong/safe."
        
    elif strategy == "survivor_targets":
        role = "value investor"
        goal = "identifying if a recent price drop is an overreaction or buying opportunity"
        scoring = "High score (1.0) means the stock is fundamentally strong and the DIP IS SAFE TO BUY."
        
    else: # trend, wheel, condor
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
        analysis = json.loads(response_json['response'])
        return analysis.get('score', 0.0), analysis.get('reason', 'N/A')
    except Exception as e:
        print(f"   [!] AI Error on {ticker}: {e}")
        return 0.0, "AI Failed"

def beam_to_beelink():
    print(f"\n4. Beaming {OUTPUT_FILE} to Beelink...")
    try:
        cmd = f"scp {OUTPUT_FILE} {BEELINK_USER}@{BEELINK_IP}:{BEELINK_PATH}"
        subprocess.run(cmd, shell=True, check=True)
        print("   ✅ Transfer Complete.")
    except Exception as e:
        print(f"   [!] SCP Failed: {e}")

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
        for ticker in tickers:
            if "/" in ticker: continue
            
            headlines = get_news_summary(ticker)
            score, reason = ask_llama(ticker, category, headlines)
            
            is_approved = False
            
            # Logic Flip for Short Targets handled in Prompt Definition
            # All scores > 0.6 mean "Good for this Strategy"
            if score > 0.6: is_approved = True
            
            emoji = "✅" if is_approved else "❌"
            print(f"      {emoji} {ticker:<4} | Score: {score:>4.2f} | {reason[:60]}...")
            
            if is_approved:
                final_targets[category].append(ticker)

    print("\n3. Saving Results...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_targets, f, indent=4)
        
    beam_to_beelink()

if __name__ == "__main__":
    run_scout()