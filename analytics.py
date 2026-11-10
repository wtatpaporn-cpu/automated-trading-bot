import os
import pandas as pd

# ==========================================
# 1. LOAD TRADE LOG
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
log_filename = os.path.join(script_dir, "trade_log.csv")

if not os.path.exists(log_filename):
    print("❌ No trade_log.csv file found yet! Let your bot run at least once to generate data.")
    exit()

df = pd.read_csv(log_filename)

print("==========================================")
print("       TRADING BOT PERFORMANCE REPORT     ")
print("==========================================")

# ==========================================
# 2. OVERALL ACTIVITY METRICS
# ==========================================
total_rows = len(df)
total_buys = len(df[df['Decision'].str.contains('BUY', na=False)])
total_rejections = len(df[df['Decision'] == 'REJECTED'])
total_holds = len(df[df['Decision'] == 'HOLD'])
total_errors = len(df[df['Decision'] == 'ERROR'])

print(f"📊 Total Watchlist Evaluations Logged: {total_rows}")
print(f"✅ Total Bracket Buy Orders Placed:   {total_buys}")
print(f"🚫 Total Stocks Filtered Out:         {total_rejections}")
print(f"⏸️ Total AI 'HOLD' Verdicts:          {total_holds}")
print(f"⚠️ Total System/API Errors Logged:    {total_errors}")
print("------------------------------------------")

# ==========================================
# 3. BREAKDOWN BY REJECTION REASON
# ==========================================
if total_rejections > 0:
    print("📉 Rejection Breakdown:")
    rejection_counts = df[df['Decision'] == 'REJECTED']['Notes'].value_counts()
    for reason, count in rejection_counts.items():
        print(f"   - {reason}: {count} time(s)")
    print("------------------------------------------")

# ==========================================
# 4. RECENT BUY ORDERS AUDIT
# ==========================================
buy_df = df[df['Decision'].str.contains('BUY', na=False)]
if not buy_df.empty:
    print("🚀 Recent Executed Trades:")
    for idx, row in buy_df.tail(5).iterrows():
        print(f"   [{row['Timestamp']}] {row['Ticker']} -> {row['Notes']}")
else:
    print("🚀 No buy orders executed yet.")

print("==========================================")