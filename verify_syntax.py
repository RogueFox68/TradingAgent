
import sys
import os

# Add paths
sys.path.append("c:/Trading System/Trading_Lab/trading-bot-fleet")
sys.path.append("c:/Trading System/TradingAgent")

def test_import(module_name, file_path):
    print(f"Testing {module_name}...", end=" ")
    try:
        # We just check if file exists and is valid python syntax primarily, 
        # actual import might fail if dependencies are missing in this env (like pandas_ta)
        # But we want to catch SyntaxError which replace_file_content might introduce
        with open(file_path, 'r', encoding='utf-8') as f:
            compile(f.read(), file_path, 'exec')
        print("[OK] Syntax Valid")
    except Exception as e:
        print(f"[FAIL] Syntax Error: {e}")

files = [
    ("logger", "c:/Trading System/Trading_Lab/trading-bot-fleet/logger.py"),
    ("utils", "c:/Trading System/Trading_Lab/trading-bot-fleet/utils.py"),
    ("wheel_bot", "c:/Trading System/Trading_Lab/trading-bot-fleet/wheel_bot.py"),
    ("trend_bot", "c:/Trading System/Trading_Lab/trading-bot-fleet/trend_bot.py"),
    ("survivor_bot", "c:/Trading System/Trading_Lab/trading-bot-fleet/survivor_bot.py"),
    ("condor_bot", "c:/Trading System/Trading_Lab/trading-bot-fleet/condor_bot.py"),
    ("sector_scout_3", "c:/Trading System/TradingAgent/sector_scout_3.py"),
    ("commander", "c:/Trading System/Trading_Lab/trading-bot-fleet/commander.py"),
]

for name, path in files:
    test_import(name, path)
