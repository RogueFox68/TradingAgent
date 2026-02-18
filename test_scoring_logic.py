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

def test_weighted_scoring():
    print("\n--- Testing Weighted Scoring (Mock) ---")
    print(f"{'Scenario':<20} | {'Tech':<6} | {'T1':<6} | {'T2':<6} | {'Soc':<6} | {'Final':<6} | {'Expected':<8}")
    print("-" * 75)

    scenarios = [
        # 1. All Data Present (0.3, 0.3, 0.2, 0.1, 0.1)
        {"tech": 0.5, "t1": 0.5, "t2": 0.5, "t3": 0.5, "soc": 0.5, "exp": 0.50},
        
        # 2. Tech Only (Weights reallocated to Tech)
        # Tech: 0.3 + 0.3(T1) + 0.2(T2) + 0.1(T3) + 0.1(Soc) = 1.0 * Tech
        {"tech": 0.8, "t1": None, "t2": None, "t3": None, "soc": None, "exp": 0.80},
        
        # 3. Tech + Elite News (Rest missing)
        # Tech: 0.3 + 0.2(T2) + 0.1(T3) + 0.1(Soc) = 0.7 weight
        # T1: 0.3 weight
        # Final = (0.5 * 0.7) + (0.8 * 0.3) = 0.35 + 0.24 = 0.59
        {"tech": 0.5, "t1": 0.8, "t2": None, "t3": None, "soc": None, "exp": 0.59},
        
        # 4. Full Bullish 
        # (1.0*0.3) + (1.0*0.3) + (1.0*0.2) + (1.0*0.1) + (1.0*0.1) = 1.0
        {"tech": 1.0, "t1": 1.0, "t2": 1.0, "t3": 1.0, "soc": 1.0, "exp": 1.00},
    ]

    for s in scenarios:
        tech = s["tech"]
        t1 = s["t1"]
        t2 = s["t2"]
        t3 = s["t3"]
        soc = s["soc"]
        
        # Simulate logic from sector_scout_3.py
        scores = []
        weights = []
        
        # Tech (Base 30%)
        scores.append(tech)
        weights.append(0.30)
        
        # T1 (30%)
        if t1 is not None:
            scores.append(t1)
            weights.append(0.30)
        else:
            weights[0] += 0.30
            
        # T2 (20%)
        if t2 is not None:
            scores.append(t2)
            weights.append(0.20)
        else:
            weights[0] += 0.20
            
        # T3 (10%)
        if t3 is not None:
            scores.append(t3)
            weights.append(0.10)
        else:
            weights[0] += 0.10
            
        # Soc (10%)
        if soc is not None:
            scores.append(soc)
            weights.append(0.10)
        else:
            weights[0] += 0.10
            
        final = 0.0
        for i in range(len(scores)):
            final += scores[i] * weights[i]
            
        status = "OK" if abs(final - s["exp"]) < 0.01 else "FAIL"
        
        # Format for print
        t1_s = f"{t1:.1f}" if t1 is not None else "N/A"
        t2_s = f"{t2:.1f}" if t2 is not None else "N/A"
        soc_s = f"{soc:.1f}" if soc is not None else "N/A"
        
        print(f"Scenario             | {tech:<6.1f} | {t1_s:<6} | {t2_s:<6} | {soc_s:<6} | {final:<6.2f} | {s['exp']:<8.2f} {status}")

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

if __name__ == "__main__":
    test_weighted_scoring()
