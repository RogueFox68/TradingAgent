
import sys
import traceback

print("Diagnostic start")
try:
    import sector_scout_3
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
    traceback.print_exc()
