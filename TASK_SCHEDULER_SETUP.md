# Task Scheduler Setup (Corsair)

Three scheduled tasks run `run_scout.bat` daily:

| Task                              | Time (CST) | Purpose             |
|-----------------------------------|------------|---------------------|
| TradingAgent_DailyScout_Open      | 8:30 AM    | Market-open scan    |
| TradingAgent_DailyScout_Lunch     | 12:00 PM   | Mid-session refresh |
| TradingAgent_DailyScout_Close     | 3:00 PM    | Pre-close scan      |

> **Note:** `market_scanner.py` only proceeds between 08:30 and 15:00 CST (its
> `is_mission_time()` gate). A task scheduled before 08:30 makes the scan exit
> immediately, so the first run must be at/after 08:30.

## Standardized Configuration

All three tasks share the same security and action settings:

- **Run as user:** RemoteAdmin
- **Logon type:** Password (runs whether user is logged in or not)
- **Run level:** Highest privileges
- **Action — Execute:** `C:\TradingAgent\run_scout.bat`
- **Action — Working directory:** `C:\TradingAgent`

## Recreating from scratch

1. Open `taskschd.msc`
2. Create Basic Task → set time → action: Start a program
3. Program/script: `C:\TradingAgent\run_scout.bat`
4. Start in: `C:\TradingAgent`
5. After creating, Properties → General → "Run whether user is logged on or not" + "Run with highest privileges"