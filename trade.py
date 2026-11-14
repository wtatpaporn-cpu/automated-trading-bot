from datetime import datetime
import os
import csv
import time
import json
from pydantic import BaseModel
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from google import genai
import pandas as pd
import yfinance as yf

# ==========================================
# 1. CONFIGURATION
# ==========================================
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

MAX_STOCKS_PER_INDUSTRY = 3
TARGET_ALLOCATION = 500.0
TAKE_PROFIT_PCT = 0.25
STOP_LOSS_PCT = 0.15

BUY_MAX_FORWARD_PE = 90.0
BUY_MIN_ROE = 0.08
BUY_MIN_REV_GROWTH = 0.12
BUY_MAX_DIST_HIGH = 0.50
SELL_MIN_REV_GROWTH = 0.05

ai_client = genai.Client(api_key=GEMINI_API_KEY)
alpaca_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# ==========================================
# 2. SCHEMA & LOGGING
# ==========================================
class StockEvaluation(BaseModel):
    pass_stock: bool
    reason: str

script_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILENAME = os.path.join(script_dir, "trade_log.csv")

def log_trade(timestamp, ticker, pe, roe, decision, notes=""):
    exists = os.path.isfile(LOG_FILENAME)
    with open(LOG_FILENAME, mode="a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["Timestamp", "Ticker", "Forward P/E", "ROE (%)", "Decision", "Notes"])
        w.writerow([timestamp, ticker, pe, roe, decision, notes])

# ==========================================
# 3. LOAD WATCHLIST
# ==========================================
watchlist = []
try:
    with open(os.path.join(script_dir, "watchlist.csv"), "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    h_idx = next(i for i, l in enumerate(lines) if "Symbol" in l)
    df = pd.read_csv(os.path.join(script_dir, "watchlist.csv"), skiprows=h_idx, on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    watchlist = [str(t).strip() for t in df["Symbol"].dropna() if str(t) != "Symbol" and len(str(t)) <= 5]
    print(f"Watchlist: {len(watchlist)} tickers loaded.")
except Exception as e:
    watchlist = ["AAPL", "MSFT", "NVDA", "AVGO", "PLTR"]

# ==========================================
# 4. SCREENING & AI EVALUATION
# ==========================================
qualified_stocks = []
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for raw in watchlist:
    ticker = raw.replace(".", "-")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        pe, roe, rev, pm, high = info.get("forwardPE"), info.get("returnOnEquity"), info.get("revenueGrowth", 0), info.get("profitMargins", 0), info.get("fiftyTwoWeekHigh")

        if not pe or not roe or not high:
            log_trade(current_time, raw, "N/A", "N/A", "REJECTED", "Missing Data")
            continue

        if not ((0 < pe <= BUY_MAX_FORWARD_PE) and (roe >= BUY_MIN_ROE) and (rev >= BUY_MIN_REV_GROWTH) and (pm > -0.10)):
            log_trade(current_time, raw, f"{pe:.2f}", f"{roe*100:.1f}%", "REJECTED", "Failed Pre-Screen")
            continue

        df_hist = stock.history(period="60d")
        if len(df_hist) < 20:
            continue

        price = df_hist["Close"].iloc[-1]
        sma50 = df_hist["Close"].rolling(50).mean().iloc[-1] if len(df_hist) >= 50 else df_hist["Close"].mean()
        if not (price > sma50 and (high - price) / high <= BUY_MAX_DIST_HIGH):
            continue

        prompt = f"Evaluate {info.get('shortName', ticker)} ({ticker}) | P/E: {pe}, ROE: {roe*100:.1f}%, RevGrowth: {rev*100:.1f}%. Is this a high-conviction innovation leader?"
        resp = ai_client.models.generate_content(model="gemini-3.5-flash-lite", contents=prompt, config={"response_mime_type": "application/json", "response_schema": StockEvaluation})
        res = json.loads(resp.text)

        if res["pass_stock"]:
            qualified_stocks.append(ticker)
            print(f"  [PASS] {ticker} | {res['reason'][:60]}...")
            log_trade(current_time, raw, f"{pe:.2f}", f"{roe*100:.1f}%", "AI_PASS", res["reason"])
        else:
            log_trade(current_time, raw, f"{pe:.2f}", f"{roe*100:.1f}%", "AI_REJECT", res["reason"])

        time.sleep(0.5)
    except Exception as e:
        pass

print(f"Screening complete: {len(qualified_stocks)} approved.")

# ==========================================
# 5. PORTFOLIO AUDIT & REBALANCING
# ==========================================
account = alpaca_client.get_account()
cash = float(account.cash)
positions = alpaca_client.get_all_positions()

industry_counts = {}
for p in positions:
    ind = "Unknown"
    try:
        ind = yf.Ticker(p.symbol.replace(".", "-")).info.get("industry", "Unknown")
    except:
        pass
    industry_counts[ind] = industry_counts.get(ind, 0) + 1

for p in positions:
    t = p.symbol.replace(".", "-")
    try:
        obj = yf.Ticker(t)
        ind = obj.info.get("industry", "Unknown")
        rev = obj.info.get("revenueGrowth", 0) or 0
        df_p = obj.history(period="60d")
        uptrend = df_p["Close"].iloc[-1] > df_p["Close"].rolling(50).mean().iloc[-1] if len(df_p) >= 50 else True

        if industry_counts.get(ind, 0) > MAX_STOCKS_PER_INDUSTRY or not (rev >= SELL_MIN_REV_GROWTH and uptrend):
            print(f"  [SELL] {t} (Audit/Cap fail)")
            alpaca_client.close_position(t)
            log_trade(current_time, t, "N/A", "N/A", "SELL", "Rebalance / Audit fail")
            industry_counts[ind] = max(0, industry_counts.get(ind, 1) - 1)
    except Exception as e:
        print(f"  [ERROR] Selling {t}: {e}")

time.sleep(2.0)
cash = float(alpaca_client.get_account().cash)

# ==========================================
# 6. EXECUTE ORDERS (WHOLE SHARES + BRACKETS)
# ==========================================
for t in qualified_stocks:
    if cash < TARGET_ALLOCATION:
        print("Stopping: Insufficient cash.")
        break
    try:
        obj = yf.Ticker(t)
        ind = obj.info.get("industry", "Unknown")
        if industry_counts.get(ind, 0) >= MAX_STOCKS_PER_INDUSTRY:
            continue

        price = obj.history(period="1d")["Close"].iloc[-1]
        
        # Round to the closest whole share amount (minimum 1 share)
        qty = max(1, int(round(TARGET_ALLOCATION / price)))

        tp, sl = round(price * (1 + TAKE_PROFIT_PCT), 2), round(price * (1 - STOP_LOSS_PCT), 2)
        order = MarketOrderRequest(
            symbol=t, 
            qty=qty, 
            side=OrderSide.BUY, 
            time_in_force=TimeInForce.DAY, 
            order_class=OrderClass.BRACKET, 
            take_profit=TakeProfitRequest(limit_price=tp), 
            stop_loss=StopLossRequest(stop_price=sl)
        )
        
        alpaca_client.submit_order(order_data=order)
        print(f"  [BUY] {t} ({qty} whole shares) [TP: {tp} | SL: {sl}]")
        
        industry_counts[ind] = industry_counts.get(ind, 0) + 1
        cash = float(alpaca_client.get_account().cash)
        log_trade(current_time, t, "N/A", "N/A", "BUY", f"Bought {qty} whole shares. TP: {tp}, SL: {sl}")
    except Exception as e:
        print(f"  [ERROR] {t}: {e}")

print("Execution finished.")
