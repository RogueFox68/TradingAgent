@echo off
:: Go to the script's directory
cd /d "%~dp0"

:: Create a visual separator in the logs
echo ========================================================== >> scout_log.txt
echo [%DATE% %TIME%] 🚀 STARTING DAILY TRADING SEQUENCE (Fixed) >> scout_log.txt

:: Enable UTF-8 for Emojis
set PYTHONUTF8=1

:: ---------------------------------------------------------
:: STEP 1: MARKET SCANNER (The Wide Net)
:: ---------------------------------------------------------
echo [%DATE% %TIME%] Phase 1: Launching Market Scanner... >> scout_log.txt
python -u market_scanner.py >> scout_log.txt 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] ❌ CRITICAL: Scanner Failed! Scout will use Core Backup. >> scout_log.txt
) else (
    echo [%DATE% %TIME%] ✅ Scanner Complete. Dragnet file ready. >> scout_log.txt
)

:: ---------------------------------------------------------
:: STEP 2: SECTOR SCOUT (The Brain)
:: ---------------------------------------------------------
echo [%DATE% %TIME%] Phase 2: Launching Sector Scout... >> scout_log.txt
python -u sector_scout_3.py >> scout_log.txt 2>&1

:: Final Log
echo [%DATE% %TIME%] 🏁 MISSION COMPLETE. >> scout_log.txt
echo ========================================================== >> scout_log.txt