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
# 1. CONFIGURATION & CREDENTIALS (SECURE)
# ==========================================
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Aggressive Growth Strategy Limits (Optimized)
MAX_STOCKS_PER_INDUSTRY = 3  # Capped by specific industry (e.g. Semiconductors)
TARGET_ALLOCATION = 500.0  # Dollar amount per stock
TAKE_PROFIT_PCT = 0.25  # +25% Take Profit for aggressive runners
STOP_LOSS_PCT = 0.15  # -15% Stop Loss for high-beta volatility room

# Relaxed Buy Thresholds (Balanced to prevent false rejections)
BUY_MAX_FORWARD_PE = 90.0  # Raised ceiling to accommodate high-beta tech multiples
BUY_MIN_ROE = 0.08  # 8% Return on Equity (relaxed from 12%)
BUY_MIN_REV_GROWTH = 0.12  # 12%+ top-line revenue growth (relaxed from 15%)
BUY_MAX_DIST_HIGH = (
    0.50  # Allows up to 50% pullbacks to catch major trend bounces
)

# Sell / Hold Audit Thresholds
SELL_MIN_REV_GROWTH = 0.05  # Only force-sell if revenue growth drops below 5%

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
  """Appends a decision row to trade_log.csv right next to the script."""
  file_exists = os.path.isfile(LOG_FILENAME)
  with open(LOG_FILENAME, mode="a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
      writer.writerow([
          "Timestamp",
          "Ticker",
          "Forward P/E",
          "P/B",
          "ROE (%)",
          "Decision",
          "Notes",
      ])
    writer.writerow([timestamp, ticker, pe, pb, roe, decision, notes])


# ==========================================
# 4. LOAD WATCHLIST FROM CSV AUTOMATICALLY
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
    df_watchlist = pd.read_csv(
        csv_filename, skiprows=header_idx, on_bad_lines="skip"
    )
    df_watchlist.columns = df_watchlist.columns.str.strip()
    watchlist = df_watchlist["Symbol"].dropna().tolist()
    watchlist = [
        str(t).strip() for t in watchlist if str(t) != "Symbol" and len(str(t)) <= 5
    ]
    print(f"Successfully loaded {len(watchlist)} tickers from CSV.")
  else:
    raise ValueError("Could not locate 'Symbol' row in the file.")

except Exception as e:
  print(f"Error reading CSV file: {e}")
  watchlist = ["AAPL", "MSFT", "NVDA", "AVGO", "PLTR"]


# ==========================================
# 5. REFINED GROWTH SCREENING & GEMINI AI
# ==========================================
print(
    f"\nRunning Relaxed Growth & Structured AI Screen on {len(watchlist)}"
    " stocks..."
)
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

    # Handle missing data gracefully with fallback logic if needed
    if not forward_pe or not roe or not fifty_two_week_high:
      log_trade_decision(
          current_time,
          raw_ticker,
          "N/A",
          "N/A",
          "N/A",
          "REJECTED",
          "Missing Fundamental Data",
      )
      continue

    # 1. Relaxed Math Pre-Screen
    is_growth_and_quality = (
        (0 < forward_pe <= BUY_MAX_FORWARD_PE)
        and (roe >= BUY_MIN_ROE)
        and (revenue_growth >= BUY_MIN_REV_GROWTH)
        and (profit_margins > -0.10)
    )  # Allows slight negative margins for heavy R&D tech disruptors

    if not is_growth_and_quality:
      log_trade_decision(
          current_time,
          raw_ticker,
          f"{forward_pe:.2f}",
          f"{pb_ratio:.2f}" if pb_ratio else "N/A",
          f"{roe*100:.1f}%",
          "REJECTED",
          "Failed Relaxed Growth Pre-Screen",
      )
      continue

    # 2. Technical Momentum & Expanded Pullback Check (Up to 50% pullback)
    df = stock.history(period="60d")
    if len(df) < 20:
      log_trade_decision(
          current_time,
          raw_ticker,
          f"{forward_pe:.2f}",
          f"{pb_ratio:.2f}" if pb_ratio else "N/A",
          f"{roe*100:.1f}%",
          "REJECTED",
          "Insufficient History",
      )
      continue

    current_price = df["Close"].iloc[-1]
    sma_50 = (
        df["Close"].rolling(window=50).mean().iloc[-1]
        if len(df) >= 50
        else df["Close"].mean()
    )
    distance_from_high = (
        fifty_two_week_high - current_price
    ) / fifty_two_week_high

    if not (current_price > sma_50 and distance_from_high <= BUY_MAX_DIST_HIGH):
      log_trade_decision(
          current_time,
          raw_ticker,
          f"{forward_pe:.2f}",
          f"{pb_ratio:.2f}" if pb_ratio else "N/A",
          f"{roe*100:.1f}%",
          "REJECTED",
          "Failed Momentum or Overextended",
      )
      continue

    # 3. Gemini Aggressive Growth Evaluation (Structured JSON Mode)
    prompt = f"""
        You are an elite aggressive growth equity analyst. Evaluate this stock for a high-beta growth portfolio:
        - Company: {company_name} ({ticker})
        - Sector / Industry: {sector} / {industry}
        - Current Price: ${current_price:.2f}
        - Forward P/E Ratio: {forward_pe}
        - ROE: {roe * 100:.1f}%
        - Revenue Growth Rate: {revenue_growth * 100:.1f}%
        - Profit Margin: {profit_margins * 100:.1f}%

        Is this a high-conviction innovation leader with strong secular tailwinds, and is its valuation justifiable to buy today?
        """

    response = ai_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": StockEvaluation,
        },
    )

    result_data = json.loads(response.text)
    is_passed = result_data["pass_stock"]
    reason = result_data["reason"]

    if is_passed:
      qualified_stocks.append(ticker)
      print(f"  [AI GROWTH PASS] {ticker} ({company_name}) — {reason}")
      log_trade_decision(
          current_time,
          raw_ticker,
          f"{forward_pe:.2f}",
          f"{pb_ratio:.2f}" if pb_ratio else "N/A",
          f"{roe*100:.1f}%",
          "AI_GROWTH_PASS",
          reason,
      )
    else:
      print(f"  [AI REJECTED] {ticker} ({company_name}) — {reason}")
      log_trade_decision(
          current_time,
          raw_ticker,
          f"{forward_pe:.2f}",
          f"{pb_ratio:.2f}" if pb_ratio else "N/A",
          f"{roe*100:.1f}%",
          "AI_REJECT",
          reason,
      )

    time.sleep(0.5)

  except Exception as e:
    print(f"  [ERROR] Processing {raw_ticker}: {e}")

print(
    f"\nScreening complete! {len(qualified_stocks)} stock(s) approved:"
    f" {qualified_stocks}"
)


# ==========================================
# 6. FETCH EXISTING PORTFOLIO & CASH BALANCE
# ==========================================
current_industry_counts = {}
available_cash = 0.0

try:
  account = alpaca_client.get_account()
  available_cash = float(account.cash)

  open_positions = alpaca_client.get_all_positions()
  for p in open_positions:
    pos_ticker = p.symbol.replace(".", "-")
    pos_info = yf.Ticker(pos_ticker).info
    pos_ind = pos_info.get("industry", "Unknown")
    current_industry_counts[pos_ind] = (
        current_industry_counts.get(pos_ind, 0) + 1
    )

  print(f"\n💰 Current Available Cash: ${available_cash:,.2f}")
  print(f"📊 Current Portfolio Industry Counts: {current_industry_counts}")
except Exception as e:
  print(f"⚠️ Could not fetch Alpaca account details: {e}")
  exit()


# ==========================================
# 7. PORTFOLIO REBALANCING & INDUSTRY PRUNING
# ==========================================
print("\n🔄 Auditing portfolio for industry caps and filter compliance...")

industry_positions = {}
open_positions = alpaca_client.get_all_positions()
for p in open_positions:
  pos_ticker = p.symbol.replace(".", "-")
  try:
    pos_info = yf.Ticker(pos_ticker).info
    pos_ind = pos_info.get("industry", "Unknown")
  except Exception:
    pos_ind = "Unknown"

  if pos_ind not in industry_positions:
    industry_positions[pos_ind] = []
  industry_positions[pos_ind].append((p, pos_ticker))

# Enforce Industry Caps
for ind, positions in industry_positions.items():
  if len(positions) > MAX_STOCKS_PER_INDUSTRY:
    excess_count = len(positions) - MAX_STOCKS_PER_INDUSTRY
    print(
        f"  ⚠️ Industry '{ind}' has {len(positions)} positions (Cap:"
        f" {MAX_STOCKS_PER_INDUSTRY}). Pruning {excess_count} excess..."
    )

    for p, ticker in positions[-excess_count:]:
      print(
          f"    📉 [INDUSTRY PRUNE] Selling {ticker} to bring '{ind}' back down"
          " to cap."
      )
      try:
        open_orders = alpaca_client.get_orders(
            filter=GetOrdersRequest(
                status=QueryOrderStatus.OPEN, symbols=[ticker]
            )
        )
        for order in open_orders:
          alpaca_client.cancel_order_by_id(order.id)

        time.sleep(1.5)

        sell_order_data = MarketOrderRequest(
            symbol=ticker,
            qty=p.qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        alpaca_client.submit_order(order_data=sell_order_data)
        log_trade_decision(
            current_time,
            ticker,
            "N/A",
            "N/A",
            "N/A",
            "INDUSTRY_PRUNE_SELL",
            f"Exceeded industry cap for {ind}",
        )
      except Exception as e:
        print(f"    ❌ Failed to sell {ticker}: {e}")

# Relaxed Hold Audit for Existing Positions
print("\n🔍 Auditing remaining positions against relaxed growth filters...")
remaining_positions = alpaca_client.get_all_positions()
for p in remaining_positions:
  pos_ticker = p.symbol.replace(".", "-")
  try:
    stock_obj = yf.Ticker(pos_ticker)
    info = stock_obj.info

    rev_growth = info.get("revenueGrowth", 0) or 0

    df_pos = stock_obj.history(period="60d")
    is_in_uptrend = True
    if len(df_pos) >= 50:
      df_pos["SMA_50"] = df_pos["Close"].rolling(window=50).mean()
      is_in_uptrend = df_pos["Close"].iloc[-1] > df_pos["SMA_50"].iloc[-1]

    is_still_valid = (rev_growth >= SELL_MIN_REV_GROWTH) and is_in_uptrend

    if not is_still_valid:
      print(
          f"  📉 [FILTER SELL] {pos_ticker} failed growth/trend audit. Selling"
          " position."
      )

      open_orders = alpaca_client.get_orders(
          filter=GetOrdersRequest(
              status=QueryOrderStatus.OPEN, symbols=[pos_ticker]
          )
      )
      for order in open_orders:
        alpaca_client.cancel_order_by_id(order.id)

      time.sleep(1.5)

      sell_order_data = MarketOrderRequest(
          symbol=p.symbol,
          qty=p.qty,
          side=OrderSide.SELL,
          time_in_force=TimeInForce.DAY,
      )
      alpaca_client.submit_order(order_data=sell_order_data)
      log_trade_decision(
          current_time,
          pos_ticker,
          "N/A",
          "N/A",
          "N/A",
          "FILTER_SELL",
          "Failed ongoing growth/trend audit",
      )

  except Exception as e:
    print(f"  ⚠️ Error auditing {pos_ticker}: {e}")

# Refresh cash and industry counts
account_refreshed = alpaca_client.get_account()
available_cash = float(account_refreshed.cash)

current_industry_counts = {}
for p in alpaca_client.get_all_positions():
  try:
    ind = (
        yf.Ticker(p.symbol.replace(".", "-")).info.get("industry", "Unknown")
    )
  except:
    ind = "Unknown"
  current_industry_counts[ind] = current_industry_counts.get(ind, 0) + 1

print(f"\n💰 Post-Rebalance Available Cash: ${available_cash:,.2f}")
print(f"📊 Post-Rebalance Industry Counts: {current_industry_counts}\n")


# ==========================================
# 8. EXECUTE BRACKET ORDERS
# ==========================================
if not qualified_stocks:
  print("No stocks passed filters today. Exiting safely.")
  exit()

print(f"\nEvaluating bracket orders with cash & industry limits...")

for chosen_ticker in qualified_stocks:
  if available_cash < TARGET_ALLOCATION:
    print(
        "\n❌ Stopping execution: Available cash"
        f" (${available_cash:,.2f}) is below target allocation"
        f" (${TARGET_ALLOCATION:,.2f})."
    )
    log_trade_decision(
        current_time,
        chosen_ticker,
        "N/A",
        "N/A",
        "N/A",
        "SKIPPED",
        f"Insufficient cash",
    )
    break

  try:
    stock_obj = yf.Ticker(chosen_ticker)
    stock_industry = stock_obj.info.get("industry", "Unknown")

    existing_industry_count = current_industry_counts.get(stock_industry, 0)
    if existing_industry_count >= MAX_STOCKS_PER_INDUSTRY:
      print(
          f"    🚫 Skipping {chosen_ticker}: Industry '{stock_industry}' has"
          " reached its limit."
      )
      log_trade_decision(
          current_time,
          chosen_ticker,
          "N/A",
          "N/A",
          "N/A",
          "SKIPPED",
          f"Industry cap reached for {stock_industry}",
      )
      continue

    current_price = stock_obj.history(period="1d")["Close"].iloc[-1]
    shares_to_buy = round(TARGET_ALLOCATION / current_price, 4)

    take_profit_price = round(current_price * (1 + TAKE_PROFIT_PCT), 2)
    stop_loss_price = round(current_price * (1 - STOP_LOSS_PCT), 2)

    print(
        f"\n-> Placing order for {chosen_ticker} ({stock_industry}):"
        f" ~${current_price:.2f}/share ({shares_to_buy} shares)"
    )
    print(
        f"    Take Profit: ${take_profit_price} (+25%) | Stop Loss:"
        f" ${stop_loss_price} (-15%)"
    )

    bracket_order_data = MarketOrderRequest(
        symbol=chosen_ticker,
        qty=shares_to_buy,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=take_profit_price),
        stop_loss=StopLossRequest(stop_price=stop_loss_price),
    )

    submitted_order = alpaca_client.submit_order(order_data=bracket_order_data)
    print(f"    Successfully placed! Order ID: {submitted_order.id}")

    current_industry_counts[stock_industry] = existing_industry_count + 1

    account_refreshed = alpaca_client.get_account()
    available_cash = float(account_refreshed.account.cash
                           if hasattr(account_refreshed, "account")
                           else account_refreshed.cash)
    portfolio_val = float(account_refreshed.equity)

    print(f"    💰 Remaining Cash: ${available_cash:,.2f}")
    print(f"    📈 Total Portfolio Value: ${portfolio_val:,.2f}")

    log_notes = (
        f"Bought {shares_to_buy} shares in {stock_industry}. TP"
        f" @ ${take_profit_price}, SL @ ${stop_loss_price}."
    )
    log_trade_decision(
        current_time,
        chosen_ticker,
        "N/A",
        "N/A",
        "N/A",
        "BRACKET_BUY",
        log_notes,
    )

  except Exception as e:
    print(f"    ❌ Order failed for {chosen_ticker}: {e}")
    log_trade_decision(
        current_time,
        chosen_ticker,
        "N/A",
        "N/A",
        "N/A",
        "ERROR",
        f"Order failed: {e}",
    )

print("\nAll qualified orders processed!")
