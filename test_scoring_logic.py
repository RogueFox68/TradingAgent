def normalize(tech_score, category):
    # Default scaling (0-100) -> 0-1
    tech_norm = min(max(tech_score / 100.0, 0.0), 1.0)
    
    if category in ["trend_targets", "short_targets"]:
        # ADX-based scores. Target > 40.0
        tech_norm = min(max(tech_score / 40.0, 0.0), 1.0)
    elif category == "wheel_targets":
        # Wheel scores are -5 to 10 (RSI 40-55). 10 is perfect.
        # Score 8 -> 0.8
        tech_norm = min(max(tech_score / 10.0, 0.0), 1.0)
        
    elif category == "survivor_targets":
        # Survivor scores are 10-50 (50-RSI). 30 (RSI 20) is perfect.
        # Score 30 -> 1.0
        tech_norm = min(max(tech_score / 30.0, 0.0), 1.0)
    else:
        # Fallback for unmapped categories
        tech_norm = min(max(tech_score / 100.0, 0.0), 1.0)
        
    return tech_norm

def test_weighted_scoring():
    print("\n--- Testing Weighted Scoring (Mock) ---")
    print(f"{'Scenario':<20} | {'Tech':<6} | {'T1':<6} | {'T2':<6} | {'Soc':<6} | {'Final':<6} | {'Expected':<8}")
    print("-" * 75)

    scenarios = [
        # 1. All Data Present (0.3, 0.3, 0.2, 0.1, 0.1)
        {"tech": 0.5, "t1": 0.5, "t2": 0.5, "t3": 0.5, "soc": 0.5, "exp": 0.50},
        
        # 2. Tech Only (No reallocation, neutral missing)
        {"tech": 0.8, "t1": None, "t2": None, "t3": None, "soc": None, "exp": 0.59},
        
        # 3. Tech + Elite News (Rest missing)
        {"tech": 0.5, "t1": 0.8, "t2": None, "t3": None, "soc": None, "exp": 0.59},
        
        # 4. Full Bullish 
        {"tech": 1.0, "t1": 1.0, "t2": 1.0, "t3": 1.0, "soc": 1.0, "exp": 1.00},
    ]

    for s in scenarios:
        tech = s["tech"]
        t1 = s["t1"]
        t2 = s["t2"]
        t3 = s["t3"]
        soc = s["soc"]
        
        scores = []
        weights = []
        
        scores.append(tech)
        weights.append(0.30)
        
        if t1 is not None:
            scores.append(t1)
            weights.append(0.30)
        else:
            scores.append(0.50)
            weights.append(0.30)
            
        if t2 is not None:
            scores.append(t2)
            weights.append(0.20)
        else:
            scores.append(0.50)
            weights.append(0.20)
            
        if t3 is not None:
            scores.append(t3)
            weights.append(0.10)
        else:
            scores.append(0.50)
            weights.append(0.10)
            
        if soc is not None:
            scores.append(soc)
            weights.append(0.10)
        else:
            scores.append(0.50)
            weights.append(0.10)
            
        final = 0.0
        for i in range(len(scores)):
            final += scores[i] * weights[i]
            
        assert abs(final - s["exp"]) < 0.01, f"Weighted scoring failed: {final} != {s['exp']}"

def test_normalize():
    tests = [
        ("trend_targets", 20, 0.50),   # 20/40
        ("trend_targets", 40, 1.00),   # 40/40
        ("short_targets", 10, 0.25),   # 10/40
        ("wheel_targets", 5, 0.50),    # 5/10
        ("wheel_targets", 12, 1.00),   # Cap at 1.0
        ("survivor_targets", 15, 0.50),# 15/30
        ("survivor_targets", 30, 1.00),# 30/30
        ("unknown_category", 50, 0.50) # Fallback 50/100
    ]

    for cat, score, expected in tests:
        result = normalize(score, cat)
        assert abs(result - expected) < 0.05, f"{cat} failed: {result} != {expected}"

if __name__ == "__main__":
    test_weighted_scoring()
    test_normalize()
    print("All tests passed!")
