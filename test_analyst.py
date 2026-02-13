import ollama
from GoogleNews import GoogleNews
import yfinance as yf

# --- CONFIGURATION ---
SYMBOL = "TSLA"
MODEL = "llama3.1"

def get_market_data(ticker):
    print(f"1. Fetching Price for {ticker}...")
    stock = yf.Ticker(ticker)
    price = stock.history(period='1d')['Close'].iloc[-1]
    return price

def get_headlines(ticker):
    print(f"2. Reading the News for {ticker}...")
    googlenews = GoogleNews(period='1d')
    googlenews.search(ticker)
    results = googlenews.result()
    
    # Just grab the titles to save context window space
    headlines = [x['title'] for x in results[:10]] # Top 10 headlines
    return headlines

def ask_the_oracle(ticker, price, headlines):
    print(f"3. Waking up the 5080 ({MODEL})...")
    
    # The "System Prompt" is where we fix the 'lemonade' problem.
    # We tell it EXACTLY who it is and how to behave.
    system_prompt = (
        "You are a ruthless Hedge Fund Analyst. "
        "Your job is to analyze news headlines and determine sentiment. "
        "You do not explain. You do not chatter. "
        "You output ONLY a JSON object with a score (-1.0 to 1.0) and a brief reason."
    )
    
    user_prompt = (
        f"Ticker: {ticker}\n"
        f"Current Price: ${price:.2f}\n"
        f"Headlines: {headlines}\n\n"
        "Analyze. Return JSON format: {\"score\": float, \"reason\": \"string\"}"
    )

    response = ollama.chat(model=MODEL, messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ])
    
    return response['message']['content']

# --- MAIN LOOP ---
if __name__ == "__main__":
    try:
        price = get_market_data(SYMBOL)
        news = get_headlines(SYMBOL)
        
        if not news:
            print("No news found. The internet might be broken.")
        else:
            print(f"   -> Found {len(news)} headlines.")
            analysis = ask_the_oracle(SYMBOL, price, news)
            
            print("\n--- 🧠 ANALYST REPORT ---")
            print(analysis)
            print("-------------------------")
            
    except Exception as e:
        print(f"CRASH: {e}")