from sector_scout_3 import get_tiered_news
import json

def verify():
    ticker = "AAPL"
    print(f"Testing get_tiered_news('{ticker}')...")
    
    # This calls the actual function in sector_scout_3.py
    results = get_tiered_news(ticker)
    
    print("\n--- RESULTS ---")
    print(json.dumps(results, indent=2))
    
    total = len(results['tier1']) + len(results['tier2']) + len(results['tier3'])
    print(f"\nTotal Items: {total}")
    
    if total > 0:
        print("✅ SUCCESS: News fetched and bucketed.")
    else:
        print("❌ FAILURE: No news found (Check logic or yfinance source).")

if __name__ == "__main__":
    verify()
