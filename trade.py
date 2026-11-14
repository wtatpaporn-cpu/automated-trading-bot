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
                    model="gemini-3.8-flash",
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

    positions = alpaca_client.get_all_positions()
    for p in positions:
        pos_ticker = p.symbol.replace(".", "-")
        pos_ind = "Unknown"
        try:
            pos_ind = yf.Ticker(pos_ticker).info.get("industry", "Unknown")
        except Exception:
            pass
        current_industry_counts[pos_ind] = current_industry_counts.get(pos_ind, 0) + 1

    print(f"💰 Cash: ${available_cash:,.2f} | Open Positions: {len(positions)} | Industries: {len(current_industry_counts)} tracked")
except Exception as e:
    print(f"⚠️ Account fetch error: {e}")
    exit()


# ==========================================
# 7. PORTFOLIO REBALANCING & AUDIT
# ==========================================
print("\nAuditing portfolio...")

open_positions = alpaca_client.get_all_positions()
print(f"  ℹ️ Inspecting {len(open_positions)} open position(s)...")

if not open_positions:
    print("  ℹ️ Portfolio is empty. Skipping rebalance/prune checks.")
else:
    industry_positions = {}
    for p in open_positions:
        pos_ticker = p.symbol.replace(".", "-")
        pos_ind = "Unknown"
        try:
            pos_ind = yf.Ticker(pos_ticker).info.get("industry", "Unknown")
        except Exception:
            pass
        industry_positions.setdefault(pos_ind, []).append((p, pos_ticker))

    # Enforce Industry Caps
    for ind, positions in industry_positions.items():
        if len(positions) > MAX_STOCKS_PER_INDUSTRY:
            excess = len(positions) - MAX_STOCKS_PER_INDUSTRY
            print(f"  ⚠️ Industry '{ind}' exceeds cap ({len(positions)}/{MAX_STOCKS_PER_INDUSTRY}). Pruning {excess}...")

            for p, ticker in positions[-excess:]:
                try:
                    for order in alpaca_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[ticker])):
                        try: 
                            alpaca_client.cancel_order_by_id(order.id)
                        except: 
                            pass
                    time.sleep(1.0)
                    alpaca_client.submit_order(order_data=MarketOrderRequest(symbol=ticker, qty=p.qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
                    print(f"    📉 Sold {ticker} (Industry Cap)")
                    log_trade_decision(current_time, ticker, "N/A", "N/A", "N/A", "PRUNE_SELL", f"Cap exceeded for {ind}")
                except Exception as e:
                    print(f"    ❌ Failed to sell {ticker}: {e}")

    # Hold Audit
    for p in open_positions:
        pos_ticker = p.symbol.replace(".", "-")
        try:
            stock_obj = yf.Ticker(pos_ticker)
            rev_growth = 0
            try:
                rev_growth = stock_obj.info.get("revenueGrowth", 0) or 0
            except Exception:
                pass
                
            df_pos = stock_obj.history(period="60d")
            uptrend = (df_pos["Close"].iloc[-1] > df_pos["Close"].rolling(window=50).mean().iloc[-1]) if len(df_pos) >= 50 else True

            if not ((rev_growth >= SELL_MIN_REV_GROWTH) and uptrend):
                print(f"  📉 Selling {pos_ticker} (Failed growth/trend audit)")
                for order in alpaca_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[pos_ticker])):
                    try: 
                        alpaca_client.cancel_order_by_id(order.id)
                    except: 
                        pass
                time.sleep(1.0)
                alpaca_client.submit_order(order_data=MarketOrderRequest(symbol=p.symbol, qty=p.qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY))
                log_trade_decision(current_time, pos_ticker, "N/A", "N/A", "N/A", "FILTER_SELL", "Audit failed")
        except Exception as e:
            pass

# Refresh cash
try:
    account_refreshed = alpaca_client.get_account()
    available_cash = float(account_refreshed.cash)
    print(f"💰 Post-Audit Cash: ${available_cash:,.2f}")
except Exception as e:
    print(f"⚠️ Cash refresh warning: {e}")


# ==========================================
# 8. EXECUTE ORDERS & SUMMARY REPORT
# ==========================================
if not qualified_stocks:
    print("No stocks passed filters. Exiting.")
    exit()

print(f"\nPlacing orders...")

trades_executed_count = 0

for chosen_ticker in qualified_stocks:
    if available_cash < TARGET_ALLOCATION:
        print(f"❌ Stop: Insufficient cash (${available_cash:,.2f}) for target (${TARGET_ALLOCATION:,.2f}).")
        break

    try:
        stock_obj = yf.Ticker(chosen_ticker)
        stock_industry = "Unknown"
        try:
            stock_industry = stock_obj.info.get("industry", "Unknown")
        except Exception:
            pass

        existing_count = current_industry_counts.get(stock_industry, 0)
        if existing_count >= MAX_STOCKS_PER_INDUSTRY:
            print(f"  🚫 Skipping {chosen_ticker}: Industry cap reached.")
            continue

        current_price = stock_obj.history(period="1d")["Close"].iloc[-1]
        shares_to_buy = round(TARGET_ALLOCATION / current_price, 4)
        is_fractional = not shares_to_buy.is_integer()

        if is_fractional:
            order_data = MarketOrderRequest(
                symbol=chosen_ticker, qty=shares_to_buy, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY, order_class=OrderClass.SIMPLE
            )
            print(f"  -> Buying ~{shares_to_buy} shares of {chosen_ticker} @ ${current_price:.2f} (Simple)")
            log_notes = f"Bought {shares_to_buy} fractional shares."
        else:
            tp = round(current_price * (1 + TAKE_PROFIT_PCT), 2)
            sl = round(current_price * (1 - STOP_LOSS_PCT), 2)
            order_data = MarketOrderRequest(
                symbol=chosen_ticker, qty=int(shares_to_buy), side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=tp),
                stop_loss=StopLossRequest(stop_price=sl),
            )
            print(f"  -> Buying {int(shares_to_buy)} shares of {chosen_ticker} @ ${current_price:.2f} [TP: ${tp} | SL: ${sl}]")
            log_notes = f"Bought {int(shares_to_buy)} whole shares. TP @ ${tp}, SL @ ${sl}."

        submitted = alpaca_client.submit_order(order_data=order_data)
        print(f"     Success! ID: {submitted.id[:8]}...")

        trades_executed_count += 1
        current_industry_counts[stock_industry] = existing_count + 1
        
        account_refreshed = alpaca_client.get_account()
        available_cash = float(account_refreshed.cash)
        log_trade_decision(current_time, chosen_ticker, "N/A", "N/A", "N/A", "BUY", log_notes)

    except Exception as e:
        print(f"  ❌ Order failed for {chosen_ticker}: {e}")
        log_trade_decision(current_time, chosen_ticker, "N/A", "N/A", "N/A", "ERROR", str(e))

# ==========================================
# 9. FINAL EXECUTION SUMMARY REPORT
# ==========================================
try:
    final_account = alpaca_client.get_account()
    final_equity = float(final_account.equity)
    final_cash = float(final_account.cash)
except Exception:
    final_equity = 0.0
    final_cash = available_cash

print("\n" + "="*40)
print("📊 TRADING BOT EXECUTION SUMMARY")
print("="*40)
print(f"  • Stocks Screened:     {len(watchlist)}")
print(f"  • Stocks Approved:     {len(qualified_stocks)}")
print(f"  • Trades Executed:     {trades_executed_count}")
print(f"  • Ending Cash:         ${final_cash:,.2f}")
print(f"  • Total Portfolio Eq:  ${final_equity:,.2f}")
print("="*40)
print("All tasks completed successfully!")
