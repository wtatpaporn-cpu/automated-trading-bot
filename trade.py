import os
import csv
from datetime import datetime
import pandas as pd
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from google import genai

# ==========================================
# 1. CONFIGURATION & CREDENTIALS (SECURE)
# ==========================================
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# Risk Management Limits
MAX_STOCKS_PER_SECTOR = 5   # Hard cap: no more than 5 open positions in the same sector
TARGET_ALLOCATION = 1000.0   # Dollar amount per stock

print("Initializing clients...")
ai_client = genai.Client(api_key=GEMINI_API_KEY)
alpaca_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)


# ==========================================
# 2. LOGGING HELPER FUNCTION
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILENAME = os.path.join(script_dir, "trade_log.csv")

def log_trade_decision(timestamp, ticker, pe, pb, roe, decision, notes=""):
    """Appends a decision row to trade_log.csv right next to the script."""
    file_exists = os.path.isfile(LOG_FILENAME)
    with open(LOG_FILENAME, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Ticker", "P/E", "P/B", "ROE (%)", "Decision", "Notes"])
        writer.writerow([timestamp, ticker, pe, pb, roe, decision, notes])


# ==========================================
# 3. LOAD WATCHLIST FROM CSV AUTOMATICALLY
# ==========================================
csv_filename = os.path.join(script_dir, "watchlist.csv")

watchlist = []
try:
    with open(csv_filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    header_idx = None
    for idx, line in enumerate(lines):
        if "Symbol" in line:
            header_idx = idx
            break

    if header_idx is not None:
        df_watchlist = pd.read_csv(csv_filename, skiprows=header_idx, on_bad_lines='skip')
        df_watchlist.columns = df_watchlist.columns.str.strip()
        watchlist = df_watchlist['Symbol'].dropna().tolist()
        watchlist = [str(t).strip() for t in watchlist if str(t) != "Symbol" and len(str(t)) <= 5]
        print(f"Successfully loaded {len(watchlist)} tickers from CSV.")
    else:
        raise ValueError("Could not locate 'Symbol' row in the file.")

except Exception as e:
    print(f"Error reading CSV file: {e}")
    watchlist = ["AAPL", "MSFT", "NVDA"]


# ==========================================
# 4. MULTI-FACTOR SCREENING & LOGGING
# ==========================================
print(f"\nRunning Multi-Factor Screen on {len(watchlist)} stocks...")
qualified_stocks = []
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

for ticker in watchlist:
    try:
        stock = yf.Ticker(ticker)
        
        # Fundamental Value Check
        info = stock.info
        pe_ratio = info.get("trailingPE", float("inf"))
        pb_ratio = info.get("priceToBook", float("inf"))
        roe = info.get("returnOnEquity", 0)
        
        pe_ratio = pe_ratio if pe_ratio is not None else float("inf")
        pb_ratio = pb_ratio if pb_ratio is not None else float("inf")
        roe = roe if roe is not None else 0
        
        is_valuable = (0 < pe_ratio < 25) and (pb_ratio < 5) and (roe > 0.10)
        if not is_valuable:
            log_trade_decision(current_time, ticker, f"{pe_ratio:.2f}", f"{pb_ratio:.2f}", f"{roe*100:.1f}%", "REJECTED", "Failed Value Filter")
            continue
            
        # Technical Momentum Check
        df = stock.history(period="50d")
        if len(df) < 20:
            log_trade_decision(current_time, ticker, f"{pe_ratio:.2f}", f"{pb_ratio:.2f}", f"{roe*100:.1f}%", "REJECTED", "Insufficient Data")
            continue
            
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        current_price = df['Close'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        
        if current_price > sma_20:
            qualified_stocks.append(ticker)
            print(f"  [PASS BOTH] {ticker} is Undervalued AND in an Uptrend!")
        else:
            log_trade_decision(current_time, ticker, f"{pe_ratio:.2f}", f"{pb_ratio:.2f}", f"{roe*100:.1f}%", "REJECTED", "Failed Momentum Filter")
            
    except Exception as e:
        print(f"  [ERROR] Processing {ticker}: {e}")

print(f"\nScreening complete! {len(qualified_stocks)} stock(s) passed both filters: {qualified_stocks}")


# ==========================================
# 5. FETCH EXISTING PORTFOLIO & CASH BALANCE
# ==========================================
current_sector_counts = {}
available_cash = 0.0

try:
    account = alpaca_client.get_account()
    available_cash = float(account.cash)
    
    open_positions = alpaca_client.get_all_positions()
    for p in open_positions:
        pos_ticker = p.symbol
        pos_info = yf.Ticker(pos_ticker).info
        pos_sector = pos_info.get('sector', 'Unknown')
        current_sector_counts[pos_sector] = current_sector_counts.get(pos_sector, 0) + 1
        
    print(f"\n💰 Current Available Cash: ${available_cash:,.2f}")
    print(f"📊 Current Portfolio Sector Counts: {current_sector_counts}")
except Exception as e:
    print(f"⚠️ Could not fetch Alpaca account details: {e}")
    exit()

# ==========================================
# 5.5 PORTFOLIO REBALANCING & SECTOR PRUNING
# ==========================================
print("\n🔄 Auditing portfolio for sector caps and filter compliance...")

# 1. Group current open positions by sector
sector_positions = {}
for p in open_positions:
    pos_ticker = p.symbol
    try:
        pos_info = yf.Ticker(pos_ticker).info
        pos_sector = pos_info.get('sector', 'Unknown')
    except Exception:
        pos_sector = 'Unknown'
        
    if pos_sector not in sector_positions:
        sector_positions[pos_sector] = []
    sector_positions[pos_sector].append((p, pos_ticker))

# 2. Enforce Sector Caps (Sell excess positions if a sector is over limit)
for sector, positions in sector_positions.items():
    if len(positions) > MAX_STOCKS_PER_SECTOR:
        excess_count = len(positions) - MAX_STOCKS_PER_SECTOR
        print(f"  ⚠️ Sector '{sector}' has {len(positions)} positions (Cap: {MAX_STOCKS_PER_SECTOR}). Pruning {excess_count} excess...")
        
        # Sell the excess positions (taking the newest or last added in the list)
        for p, ticker in positions[-excess_count:]:
            print(f"    📉 [SECTOR PRUNE] Selling {ticker} to bring '{sector}' back down to cap.")
            try:
                # Cancel open bracket orders (TP/SL) so shares become available to sell
                open_orders = alpaca_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[ticker]))
                for order in open_orders:
                    alpaca_client.cancel_order_by_id(order.id)

                sell_order_data = MarketOrderRequest(
                    symbol=ticker,
                    qty=p.qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC
                )
                alpaca_client.submit_order(order_data=sell_order_data)
                log_trade_decision(current_time, ticker, "N/A", "N/A", "N/A", "SECTOR_PRUNE_SELL", f"Exceeded sector cap of {MAX_STOCKS_PER_SECTOR} for {sector}")
            except Exception as e:
                print(f"    ❌ Failed to sell {ticker}: {e}")

# 3. Audit remaining positions for fundamental/momentum decay
print("\n🔍 Checking remaining positions against value & momentum filters...")
remaining_positions = alpaca_client.get_all_positions()
for p in remaining_positions:
    pos_ticker = p.symbol
    try:
        stock_obj = yf.Ticker(pos_ticker)
        info = stock_obj.info
        
        pe = info.get("trailingPE", float("inf")) or float("inf")
        pb = info.get("priceToBook", float("inf")) or float("inf")
        roe = info.get("returnOnEquity", 0) or 0
        
        is_still_valuable = (0 < pe < 25) and (pb < 5) and (roe > 0.10)
        
        df_pos = stock_obj.history(period="50d")
        is_in_uptrend = True
        if len(df_pos) >= 20:
            df_pos['SMA_20'] = df_pos['Close'].rolling(window=20).mean()
            is_in_uptrend = df_pos['Close'].iloc[-1] > df_pos['SMA_20'].iloc[-1]
            
        if not (is_still_valuable and is_in_uptrend):
            print(f"  📉 [FILTER SELL] {pos_ticker} failed current filters. Selling position.")
            
            # Cancel open bracket orders (TP/SL) so shares become available to sell
            open_orders = alpaca_client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[pos_ticker]))
            for order in open_orders:
                alpaca_client.cancel_order_by_id(order.id)

            sell_order_data = MarketOrderRequest(
                symbol=p.symbol,
                qty=p.qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            alpaca_client.submit_order(order_data=sell_order_data)
            log_trade_decision(current_time, pos_ticker, f"{pe:.2f}", f"{pb:.2f}", f"{roe*100:.1f}%", "FILTER_SELL", "Failed ongoing screen filters")
            
    except Exception as e:
        print(f"  ⚠️ Error auditing {pos_ticker}: {e}")

# Refresh cash and sector counts after all rebalancing sales
account_refreshed = alpaca_client.get_account()
available_cash = float(account_refreshed.cash)

# Re-calculate open sector counts for the buying loop later
current_sector_counts = {}
for p in alpaca_client.get_all_positions():
    try:
        sec = yf.Ticker(p.symbol).info.get('sector', 'Unknown')
    except:
        sec = 'Unknown'
    current_sector_counts[sec] = current_sector_counts.get(sec, 0) + 1

print(f"\n💰 Post-Rebalance Available Cash: ${available_cash:,.2f}")
print(f"📊 Post-Rebalance Sector Counts: {current_sector_counts}\n")


# ==========================================
# 6. EXECUTE BRACKET ORDERS WITH CASH & SECTOR CHECKS
# ==========================================
if not qualified_stocks:
    print("No stocks passed filters today. Exiting safely.")
    exit()

print(f"\nEvaluating bracket orders with cash & sector limits...")

for chosen_ticker in qualified_stocks:
    # Check if we have enough cash left for another position
    if available_cash < TARGET_ALLOCATION:
        print(f"\n❌ Stopping execution: Available cash (${available_cash:,.2f}) is below target allocation (${TARGET_ALLOCATION:,.2f}).")
        log_trade_decision(current_time, chosen_ticker, "N/A", "N/A", "N/A", "SKIPPED", f"Insufficient cash (${available_cash:,.2f})")
        break

    try:
        stock_obj = yf.Ticker(chosen_ticker)
        stock_sector = stock_obj.info.get('sector', 'Unknown')
        
        # Check Sector Cap
        existing_sector_count = current_sector_counts.get(stock_sector, 0)
        if existing_sector_count >= MAX_STOCKS_PER_SECTOR:
            print(f"    🚫 Skipping {chosen_ticker}: Sector '{stock_sector}' has reached its limit ({existing_sector_count}/{MAX_STOCKS_PER_SECTOR} active).")
            log_trade_decision(current_time, chosen_ticker, "N/A", "N/A", "N/A", "SKIPPED", f"Sector cap reached for {stock_sector}")
            continue

        current_price = stock_obj.history(period="1d")['Close'].iloc[-1]
        shares_to_buy = int(TARGET_ALLOCATION / current_price)
        if shares_to_buy < 1:
            shares_to_buy = 1
            
        take_profit_price = round(current_price * 1.05, 2)  # +5%
        stop_loss_price = round(current_price * 0.97, 2)    # -3%
        
        print(f"\n-> Placing order for {chosen_ticker} ({stock_sector}): ~${current_price:.2f}/share ({shares_to_buy} shares)")
        print(f"   Take Profit: ${take_profit_price} | Stop Loss: ${stop_loss_price}")

        bracket_order_data = MarketOrderRequest(
            symbol=chosen_ticker,
            qty=shares_to_buy,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=take_profit_price),
            stop_loss=StopLossRequest(stop_price=stop_loss_price)
        )
        
        submitted_order = alpaca_client.submit_order(order_data=bracket_order_data)
        print(f"   Successfully placed! Order ID: {submitted_order.id}")
        
        # Update local tracking variables after successful order
        current_sector_counts[stock_sector] = existing_sector_count + 1
        available_cash -= TARGET_ALLOCATION  # Approximate deduct
        
        # Re-fetch live account balance to be precise
        account_refreshed = alpaca_client.get_account()
        available_cash = float(account_refreshed.cash)
        portfolio_val = float(account_refreshed.equity)
        
        print(f"   💰 Remaining Cash: ${available_cash:,.2f}")
        print(f"   📈 Total Portfolio Value: ${portfolio_val:,.2f}")
        
        # Log to CSV
        log_notes = f"Bought {shares_to_buy} shares in {stock_sector}. TP @ ${take_profit_price}, SL @ ${stop_loss_price}. Cash Left: ${available_cash:,.2f}"
        log_trade_decision(current_time, chosen_ticker, "N/A", "N/A", "N/A", "BRACKET_BUY", log_notes)
        
    except Exception as e:
        print(f"   ❌ Order failed for {chosen_ticker}: {e}")
        log_trade_decision(current_time, chosen_ticker, "N/A", "N/A", "N/A", "ERROR", f"Order failed: {e}")

print("\nAll qualified orders processed!")
