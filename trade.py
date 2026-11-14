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
# 1. CONFIGURATION & CREDENTIALS
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

print("Initializing clients...")
ai_client = genai.Client(api_key=GEMINI_API_KEY)
alpaca_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)


# ==========================================
# 2. STRUCTURED SCHEMA FOR GEMINI
# ==========================================
class StockEvaluation(BaseModel):
    pass_stock: bool
    reason: str


# ==========================================
# 3. LOGGING HELPER FUNCTION
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILENAME = os.path.join(script_dir, "trade_log.csv")


def log_trade_decision(timestamp, ticker, pe, pb, roe, decision, notes=""):
    file_exists = os.path.isfile(LOG_FILENAME)
    with open(LOG_FILENAME, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Ticker", "Forward P/E", "P/B", "ROE (%)", "Decision", "Notes"
            ])
        writer.writerow([timestamp, ticker, pe, pb, roe, decision, notes])


# ==========================================
# 4. LOAD WATCHLIST FROM CSV
# ==========================================
csv_filename = os.path.join(script_dir, "watchlist.csv")

watchlist = []
try:
    with open(csv_filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    for idx, line in enumerate(lines):
        if "Symbol" in line:
            header_idx = idx
            break

    if header_idx is not None:
        df_watchlist = pd.read_csv(csv_filename, skiprows=header_idx, on_bad_lines="skip")
        df_watchlist.columns = df_watchlist.columns.str.strip()
        watchlist = df_watchlist["Symbol"].dropna().tolist()
        watchlist = [str(t).strip() for t in watchlist if str(t) != "Symbol" and len(str(t)) <= 5]
        print(f"Loaded {len(watchlist)} tickers from CSV.")
    else:
        raise ValueError("Could not locate 'Symbol' row in the file.")

except Exception as e:
    print(f"Error reading CSV file: {e}")
    watchlist = ["AAPL", "MSFT", "NVDA", "AVGO", "PLTR"]


# ==========================================
# 5. GROWTH SCREENING & AI EVALUATION
# ==========================================
print(f"\nScreening {len(watchlist)} stocks...")
qualified_stocks = []
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for raw_ticker in watchlist:
    ticker = raw_ticker.replace(".", "-")

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        forward_pe = info.get("forwardPE")
        pb_ratio = info.get("priceToBook")
        roe = info.get("returnOnEquity")
        revenue_growth = info.get("revenueGrowth", 0)
        profit_margins = info.get("profitMargins", 0)
        fifty_two_week_high = info.get("fiftyTwoWeekHigh")
        industry = info.get("industry", "Unknown")
        sector = info.get("sector", "Unknown")
        company_name = info.get("shortName", ticker)

        if not forward_pe or not roe or not fifty_two_week_high:
            log_trade_decision(current_time, raw_ticker, "N/A", "N/A", "N/A", "REJECTED", "Missing Data")
            continue

        # Pre-screen filters
        is_growth_and_quality = (
            (0 < forward_pe <= BUY_MAX_FORWARD_PE)
            and (roe >= BUY_MIN_ROE)
            and (revenue_growth >= BUY_MIN_REV_GROWTH)
            and (profit_margins > -0.10)
        )

        if not is_growth_and_quality:
            log_trade_decision(current_time, raw_ticker, f"{forward_pe:.2f}", "N/A", f"{roe*100:.1f}%", "REJECTED", "Failed Pre-Screen")
            continue

        df = stock.history(period="60d")
        if len(df) < 20:
            continue

        current_price = df["Close"].iloc[-1]
        sma_50 = df["Close"].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else df["Close"].mean()
        distance_from_high = (fifty_two_week_high - current_price) / fifty_two_week_high

        if not (current_price > sma_50 and distance_from_high <= BUY_MAX_DIST_HIGH):
            continue

        # Gemini AI Evaluation with Rate-Limit Handling
        prompt = f"""
        Evaluate this stock for an aggressive growth portfolio in 1 concise sentence:
        - Company: {company_name} ({ticker}) | Sector: {sector} / {industry}
        - Forward P/E: {forward_pe}, ROE: {roe * 100:.1f}%, Rev Growth: {revenue_growth * 100:.1f}%
        Is this a high-conviction innovation leader with strong secular tailwinds, and is its valuation justifiable?
        """

        response = None
        for attempt in range(3):
            try:
                response = ai_client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": StockEvaluation,
                    },
                )
                break
            except Exception as api_err:
                if "429" in str(api_err) or "RESOURCE_EXHAUSTED" in str(api_err):
                    print(f"  ⚠️ Rate limit hit. Pausing 25s (Attempt {attempt+1}/3)...")
                    time.sleep(25)
                else:
                    raise api_err

        if not response:
            raise Exception("API request failed after retries.")

        result_data = json.loads(response.text)
        is_passed = result_data["pass_stock"]
        reason = result_data["reason"]

        if is_passed:
            qualified_stocks.append(ticker)
            print(f"  ✅ [PASS]   {ticker:<5} | {reason[:75]}...")
            log_trade_decision(current_time, raw_ticker, f"{forward_pe:.2f}", "N/A", f"{roe*100:.1f}%", "AI_PASS", reason)
        else:
            print(f"  ❌ [REJECT] {ticker:<5} | {reason[:75]}...")
            log_trade_decision(current_time, raw_ticker, f"{forward_pe:.2f}", "N/A", f"{roe*100:.1f}%", "AI_REJECT", reason)

        time.sleep(4.1)

    except Exception as e:
        print(f"  ⚠️ [ERROR]  {raw_ticker}: {e}")

print(f"\nScreening complete! {len(qualified_stocks)} approved.")


# ==========================================
# 6. FETCH PORTFOLIO & CASH
# ==========================================
current_industry_counts = {}
available_cash = 0.0

try:
    account = alpaca_client.get_account()
    available_cash = float(account.cash)

    for p in alpaca_client.get_all_positions():
        pos_ticker = p.symbol.replace(".", "-")
        pos_ind = yf.Ticker(pos_ticker).info.get("industry", "Unknown")
        current_industry_counts[pos_ind] = current_industry_counts.get(pos_ind, 0) + 1

    print(f"💰 Cash: ${available_cash:,.2f} | Industries: {len(current_industry_counts)} tracked")
except Exception as e:
    print(f"⚠️ Account fetch error: {e}")
    exit()


# ==========================================
# 7. PORTFOLIO REBALANCING & AUDIT
# ==========================================
print("\nAuditing portfolio...")

industry_positions = {}
for p in alpaca_client.get_all_positions():
    pos_ticker = p.symbol.replace(".", "-")
    try:
        pos_ind = yf.Ticker(pos_ticker).info.get("industry", "Unknown")
    except:
        pos_ind = "Unknown"
    industry_positions.setdefault(pos_ind, []).append((p, pos_ticker))

# Enforce Industry Caps
for ind, positions in industry_positions.items():
    if len(positions) > MAX_STOCKS_PER_INDUSTRY:
        excess = len(positions) - MAX_STOCKS_PER_INDUSTRY
