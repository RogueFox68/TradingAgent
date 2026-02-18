import yfinance as yf
import json
import time

def test_news(ticker):
    print(f"--- Fecthing News for {ticker} ---")
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        print(f"Raw News Count: {len(news)}")
        
        if news:
            print("First Item Raw:")
            print(json.dumps(news[0], indent=2))
            
            print("\nAll Publishers:")
            import datetime
            for n in news:
                # Handle new structure
                info = n.get('content', n)
                
                title = info.get('title', 'No Title')
                
                # Provider/Publisher
                provider = info.get('provider', {})
                publisher = provider.get('displayName', 'Unknown')
                
                # Date Parsing
                pub_date_str = info.get('pubDate', '')
                try:
                    # ISO Format: 2026-02-18T14:43:06Z
                    dt = datetime.datetime.strptime(pub_date_str, "%Y-%m-%dT%H:%M:%SZ")
                    # Make timezone aware if needed, or just comparable
                    age_hours = (datetime.datetime.utcnow() - dt).total_seconds() / 3600
                except:
                    age_hours = 9999
                
                print(f" - [{age_hours:.1f}h ago] {publisher} : {title}")
        else:
            print("No news found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_news("AAPL")
    test_news("AMD")
