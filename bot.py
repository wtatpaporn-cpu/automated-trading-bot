import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from google import genai

# ==========================================
# 1. CONFIGURATION & KEYS
# ==========================================
ALPACA_API_KEY = "PKVA7TNPF34VY2HKXVXDSGVMQY"
ALPACA_SECRET_KEY = "8cpd3CxabNDCErzWQp78yqHDQgjFVL6qeyk3F3y1W9T4"
GEMINI_API_KEY = "AQ.Ab8RN6J3ibiFd2eaxPK4nSXg30_2oqe1J4uHsCnUQS3HXNN29g"

ticker_symbol = "NVDA"

print("Initializing clients...")
ai_client = genai.Client(api_key=GEMINI_API_KEY)
alpaca_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)


# ==========================================
# 2. FETCH REAL MARKET DATA USING YFINANCE
# ==========================================
print(f"Fetching live market data for {ticker_symbol}...")
stock = yf.Ticker(ticker_symbol)
df = stock.history(period="5d")

# Convert recent price history into text for Gemini
market_data_summary = df[['Open', 'High', 'Low', 'Close', 'Volume']].to_string()


# ==========================================
# 3. ASK GEMINI TO ANALYZE THE DATA
# ==========================================
print("Sending data to Gemini for analysis...")
prompt = f"""
You are an expert quantitative stock analyst. 
Here is the recent 5-day price and volume data for {ticker_symbol}:

{market_data_summary}

Analyze this recent price action and trend. Based strictly on this data, 
evaluate whether the momentum indicates a short-term buying opportunity.
Reply with ONLY one word: BUY, SELL, or HOLD.
"""

ai_signal = "HOLD"  # Default fallback
try:
    print("DEBUG: Making API call to Gemini...")
    response = ai_client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )
    print("DEBUG: API call finished successfully.")
    
    if response and response.text:
        ai_signal = response.text.strip().upper()
        print(f"Gemini Data-Driven Decision: {ai_signal}")
except Exception as e:
    print(f"AI request failed with error: {e}")


# ==========================================
# 4. EXECUTE TRADE ON ALPACA IF SIGNAL IS BUY
# ==========================================
if "BUY" in ai_signal:
    print(f"Gemini signaled BUY for {ticker_symbol}. Placing paper market order...")
    try:
        order_request = MarketOrderRequest(
            symbol=ticker_symbol,
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        submitted_order = alpaca_client.submit_order(order_data=order_request)
        print(f"Successfully placed paper order! Order ID: {submitted_order.id}")
    except Exception as e:
        print(f"Order submission failed (Make sure market hours are open): {e}")
else:
    print(f"Gemini signal was {ai_signal}. No trade was placed.")