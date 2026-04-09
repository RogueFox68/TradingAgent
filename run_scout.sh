#!/bin/bash
# run_scout.sh

echo "==========================================================" >> scout_log.txt
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 STARTING DAILY TRADING SEQUENCE (Linux)" >> scout_log.txt

export PYTHONUTF8=1

# Phase 1: Market Scanner
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 1: Launching Market Scanner..." >> scout_log.txt
python3 -u market_scanner.py >> scout_log.txt 2>&1

if [ $? -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ CRITICAL: Scanner Failed! Scout will use Core Backup." >> scout_log.txt
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Scanner Complete. Dragnet file ready." >> scout_log.txt
fi

# Phase 2: Sector Scout
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 2: Launching Sector Scout..." >> scout_log.txt
python3 -u sector_scout_3.py >> scout_log.txt 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🏁 MISSION COMPLETE." >> scout_log.txt
echo "==========================================================" >> scout_log.txt
