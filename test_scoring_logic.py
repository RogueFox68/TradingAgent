def normalize(tech_score, category):
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
        
    return tech_norm

tests = [
    ("trend_targets", 40, 0.40),
    ("trend_targets", 80, 0.80),
    ("condor_targets", 5, 0.33),   # 5/15
    ("condor_targets", 15, 1.00),  # 15/15
    ("wheel_targets", 5, 0.50),    # 5/10
    ("wheel_targets", 12, 1.00),   # Cap at 1.0
    ("survivor_targets", 15, 0.50),# 15/30
    ("survivor_targets", 30, 1.00) # 30/30
]

print(f"{'Category':<20} | {'Score':<5} | {'Norm':<5} | {'Expected':<5} | Status")
print("-" * 60)

for cat, score, expected in tests:
    result = normalize(score, cat)
    status = "OK" if abs(result - expected) < 0.05 else "FAIL"
    print(f"{cat:<20} | {score:<5} | {result:<5.2f} | {expected:<5.2f} | {status}")
