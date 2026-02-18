from sector_scout_3 import get_reddit_sentiment
import time

print("Testing Reddit Fetch for AAPL...")
result = get_reddit_sentiment("AAPL")
print("\n--- RESULT ---")
print(result)
print("----------------")
