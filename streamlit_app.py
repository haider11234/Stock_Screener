import warnings
import csv
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
import anthropic
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
from cryptography.fernet import Fernet
import requests
import time



st.set_page_config(page_title="AI-Powered Stock Screener", layout="wide")

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

def fetch_open_interest(ticker):
    url = f'https://www.alphavantage.co/query?function=REALTIME_OPTIONS&symbol={ticker}&apikey=2SP6D9RX4PUYOWQA'
    response = requests.get(url)
    data = response.json()
    
    if 'data' in data:
        # Sum up all open interest values
        total_open_interest = sum(int(contract['open_interest']) for contract in data['data'] if 'open_interest' in contract)
        return total_open_interest
    else:
        return None



# Function to calculate SMAs
def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

# Function to calculate MACD
def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to check if a stock meets the criteria
def meets_criteria(data, use_sma, use_price, use_wick, use_macd, use_rsi, use_ltp, use_open_interest, use_volume, rsi_threshold, ltp_threshold, open_interest_threshold, volume_threshold):
    if len(data) < 201:
        return False
    
    current_data = data.iloc[-1]
    previous_data = data.iloc[-2]
    
    if use_sma:
        # SMA Criteria
        sma_20 = calculate_sma(data, 20)
        sma_200 = calculate_sma(data, 200)

        if sma_20.iloc[-1] <= sma_200.iloc[-1]:
            return False
        if sma_20.iloc[-1] <= sma_20.iloc[-2]:
            return False
    
    if use_price:
        # Price Criteria
        sma_20 = calculate_sma(data, 20)
        if current_data['Close'] > 1.01 * sma_20.iloc[-1]:
            return False
        if current_data['Close'] <= current_data['Open']:
            return False
        if (current_data['Close'] - current_data['Open']) / current_data['Open'] <= 0.005:
            return False
    
    if use_wick:
        # Wick Criteria
        body = abs(current_data['Close'] - current_data['Open'])
        upper_wick = current_data['High'] - max(current_data['Close'], current_data['Open'])
        lower_wick = min(current_data['Close'], current_data['Open']) - current_data['Low']
        
        if upper_wick >= 0.2 * body:
            return False
        if lower_wick >= 0.2 * body:
            return False

    if use_macd:
        # MACD Criteria
        macd, signal_line = calculate_macd(data)
        if macd.iloc[-1] <= signal_line.iloc[-1]:
            return False

    if use_rsi:
        # RSI Criteria
        rsi = calculate_rsi(data)
        if rsi.iloc[-1] <= rsi_threshold:
            return False

    if use_ltp:
        # LTP above 20D SMA Criteria
        sma_20 = calculate_sma(data, 20)
        ltp_above_sma = (current_data['Close'] - sma_20.iloc[-1]) / sma_20.iloc[-1] * 100
        if ltp_above_sma <= ltp_threshold:
            return False

    if use_open_interest:
        # Open Interest Criteria
        if 'Open Interest' in data.columns and data['Open Interest'].iloc[-1] < open_interest_threshold:
            return False
            
    if use_volume:
        # Volume Criteria
        if current_data['Volume'] < volume_threshold:
            return False

    return True

def process_stock(ticker, end_date, use_sma, use_price, use_wick, use_macd, use_rsi, use_ltp, use_volume, rsi_threshold, ltp_threshold, volume_threshold):
    try:
        start_date = end_date - timedelta(days=365)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if not data.empty and meets_criteria(data, use_sma, use_price, use_wick, use_macd, use_rsi, use_ltp, False, use_volume, rsi_threshold, ltp_threshold, 0, volume_threshold):
            return ticker, data
    except Exception as e:
        st.error(f"Error processing {ticker}: {str(e)}")
    return None, None


def apply_open_interest_criterion(results, open_interest_threshold):
    final_results = {}
    total_stocks = len(results)
    for i, (ticker, data) in enumerate(results.items()):
        if i > 0 and i % 60 == 0:
            time.sleep(60)  # Wait for 60 seconds after every 60 requests
        
        open_interest = fetch_open_interest(ticker)
        if open_interest is not None and open_interest >= open_interest_threshold:
            data['Open Interest'] = open_interest  # Store the open interest data
            final_results[ticker] = data
        
        # Update progress
        status_text.text(f"Open Interest: {i+1}/{total_stocks} stocks. Found {len(final_results)} matching all criteria.")
        progress = (current_step + (i+1)/total_stocks) / total_steps
        progress_bar.progress(progress)
    
    return final_results

@st.cache_data
def fetch_all_companies():
    sp500 = []
    with open("ciks.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            sp500.append(row[0])
    
    tickers_list = sp500
    tickers_list = [ticker.replace(".", "-") for ticker in tickers_list]
    return tickers_list

# Login function
def login(username, password):
    return username == "admin" and password == "pass123"

# Function to encode image to base64
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def decrypt_message(encrypted_message: bytes, key: bytes) -> str:

    f = Fernet(key)
    decrypted_message = f.decrypt(encrypted_message).decode()
    return decrypted_message

# Function to perform ChatGPT analysis
def chatgpt_analysis(image, prompt):
    key = b'bXlfc3VwZXJfc2VjcmV0X3Bhc3N3b3JkAAAAAAAAAAA='
    enc = b'gAAAAABmq8GcHEugg0vzTTkuHlSenaVJ9_Wf26SVw9B_u2hOB8kQlhCDptGVk2UK4G-q80S6JNEPgvTgPy7db_0CHwRyuh1SjE7sz_aeylcXSjTosMYG65SUYLSrk2sRz6qMLpKrqZzDgUg7_B8cnRsCCSK--bWQYw=='

    try:
        client = OpenAI(api_key=decrypt_message(enc, key))
        MODEL = "gpt-4o"

        base64_image = encode_image(image)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes stock charts."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred with ChatGPT analysis: {str(e)}"

# Function to perform Claude AI analysis
def claude_analysis(image, prompt):
    key = b'bXlfc3VwZXJfc2VjcmV0X3Bhc3N3b3JkAAAAAAAAAAA='

    enc = b'gAAAAABmq7Fssh5Hp7yCwi-lpm9In-jvNw1BO1RI5dXm1Sp19V-yFsRBUO-aD8AUamS9xfRRKFFqSwWoUKx_s5ltChFNRbR31_-3-w2xjQsA7yzR7rV2FfFyU3XaJ-Ru80NrncpgO2Xxn_qHrGj9K6wiJn4uiISmwgLVzeEilmb9yo8TldozFOvwdragl6T5mh-8Wd_F6P7lxO-U1HtIDAlruk0wUvRS8A=='
    try:
        client = anthropic.Anthropic(
            api_key=decrypt_message(enc, key),
        )

        image_encode = encode_image(image)

        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_encode,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
        return message.content[0].text
    except Exception as e:
        return f"An error occurred with Claude AI analysis: {str(e)}"

# Function to create AI analysis PDF
# Modify the create_ai_analysis_pdf function to improve formatting
def create_ai_analysis_pdf(results, ai_results, use_open_interest, use_volume):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=20, bottomMargin=20)
    styles = getSampleStyleSheet()
    story = []

    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']

    for ticker, data in results.items():
        story.append(Paragraph(f"Ticker Name: {ticker}", title_style))
        
        if use_open_interest:
            if 'Open Interest' in data.columns:
                story.append(Paragraph(f"Open Interest: {data['Open Interest'].iloc[-1]}", normal_style))
            else:
                story.append(Paragraph("Open Interest: Data not available", normal_style))
        
        if use_volume:
            story.append(Paragraph(f"Volume: {data['Volume'].iloc[-1]}", normal_style))
        
        # Create original colorful candlestick chart
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                                             open=data['Open'],
                                             high=data['High'],
                                             low=data['Low'],
                                             close=data['Close'])])
        fig.update_layout(title=f"{ticker} Stock Chart", 
                          width=800, height=500,
                          xaxis_rangeslider_visible=False)
        
        # Ensure high quality image
        img_bytes = fig.to_image(format="png", scale=4, engine="kaleido")
        img_stream = BytesIO(img_bytes)
        img = ReportLabImage(img_stream, width=500, height=300)
        story.append(img)
        
        def format_ai_analysis(text):
            # Split the text into lines
            lines = text.split('\n')
            formatted_text = []
            
            for line in lines:
                if line.startswith('#'):
                    # Heading 1
                    formatted_text.append(Paragraph(line.strip('# '), title_style))
                elif line.startswith('##'):
                    # Heading 2
                    formatted_text.append(Paragraph(line.strip('# '), subtitle_style))
                elif line.startswith('*') or line.startswith('-'):
                    # Bullet points
                    formatted_text.append(Paragraph(f"• {line.strip('* -')}", normal_style))
                else:
                    # Normal text
                    formatted_text.append(Paragraph(line, normal_style))
            
            return formatted_text
        
        if f"{ticker}_chatgpt" in ai_results:
            story.append(Paragraph("ChatGPT Analysis:", subtitle_style))
            story.extend(format_ai_analysis(ai_results[f"{ticker}_chatgpt"]))
        
        if f"{ticker}_claude" in ai_results:
            story.append(Paragraph("Claude Analysis:", subtitle_style))
            story.extend(format_ai_analysis(ai_results[f"{ticker}_claude"]))
        
        story.append(Spacer(1, 20))
    
    doc.build(story)
    buffer.seek(0)
    return buffer
# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = []
if 'ai_results' not in st.session_state:
    st.session_state.ai_results = {}

# Login screen
if not st.session_state.logged_in:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(username, password):
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid username or password")
else:
    # Main app content
    st.title("📊 AI-Powered Stock Screener")

    # Sidebar
    st.sidebar.header("Settings")

    # Fetch all companies
    all_companies = fetch_all_companies()

    # Multi-select dropdown for companies
    selected_companies = st.sidebar.multiselect(
        "Select companies:",
        options=all_companies,
        default=all_companies[:5]  # Default to first 5 companies
    )

    if st.sidebar.checkbox("Select All Companies"):
        selected_companies = all_companies

    # Set end date to today and start date to 365 days ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    st.sidebar.write(f"Data range: {start_date.date()} to {end_date.date()}")
    st.sidebar.write("(Fixed to ensure 200-day SMA calculation)")

    # Criteria selection
    st.sidebar.subheader("Select Criteria")
    use_sma = st.sidebar.checkbox("SMA Criteria", value=True)
    use_price = st.sidebar.checkbox("Price Criteria")
    use_wick = st.sidebar.checkbox("Wick Criteria")
    use_macd = st.sidebar.checkbox("MACD Criteria")
    use_rsi = st.sidebar.checkbox("RSI Criteria")
    use_ltp = st.sidebar.checkbox("LTP above 20D SMA Criteria")
    use_open_interest = st.sidebar.checkbox("Open Interest Criteria", value=True)
    use_volume = st.sidebar.checkbox("Volume Criteria", value=True)

    # Threshold inputs
    rsi_threshold = 50
    ltp_threshold = 10
    open_interest_threshold = 500
    volume_threshold = 500000

    if use_rsi:
        rsi_threshold = st.sidebar.number_input("RSI Threshold", min_value=0, max_value=100, value=50)
    if use_ltp:
        ltp_threshold = st.sidebar.number_input("LTP above 20D SMA Threshold (%)", min_value=0, max_value=100, value=10)
    if use_open_interest:
        open_interest_threshold = st.sidebar.number_input("Open Interest Threshold", min_value=0, value=500)
    if use_volume:
        volume_threshold = st.sidebar.number_input("Volume Threshold", min_value=0, value=500000)

    # AI Analysis options
    use_chatgpt = st.sidebar.checkbox("ChatGPT AI Analysis")
    use_claude = st.sidebar.checkbox("Claude AI Analysis")

    if use_chatgpt:
        prompt_gpt =  """You are an expert in technical analysis, specializing in candlestick patterns as described in the Candlestick Trading Bible and the 20/200 SMA strategy. Your task is to analyze each stock chart provided in the PNG screenshot to determine the current trend, identify present candlestick patterns, and evaluate the stock based on the 20/200 SMA strategy. For each stock, provide a detailed analysis that includes the following:

1. **Stock Name:**
2. **Current Trend:** Describe whether the stock is in an uptrend, downtrend, or consolidating.
3. **Present Candlestick Patterns:** List the key candlestick patterns observed in the chart (e.g., Doji, Hammer, Engulfing, etc.), and provide a brief explanation of each pattern.
4. **Reversal Markers:** Identify any potential reversal patterns (e.g., Bearish Engulfing, Evening Star, etc.), explain what they signify, and provide the confidence percentage in a bearish pattern forming.
5. **Continuation Patterns:** Identify any continuation patterns (e.g., Three White Soldiers, Rising Three Methods, etc.), explain what they signify, and provide the confidence percentage in a bullish pattern forming.
6. **Sentiment:** Based on the identified patterns and current trend, indicate whether the sentiment is bullish, bearish, or neutral.
7. **Recommendation:** Provide a recommendation (BUY, SELL, NEUTRAL) based on the analysis.
8. **20/200 SMA Strategy Evaluation:**
   - **SMA Position:** Evaluate if the 20 SMA is above or below the 200 SMA.
   - **Trend Evaluation:** Determine if the 20 SMA is in an uptrend or downtrend.
   - **Candle Position:** Check if the green buy candle is within 1% of the 20 SMA with little to no wicks.
   - **Strategy Fit:** Assess how well the stock aligns with the 20/200 SMA strategy and provide specific observations.

Format the output as follows for each stock:

---

<STOCK NAME HERE>:
- **Current Trend:** <Describe the current trend>
- **Present Candlestick Patterns:**
  - Example: Bullish Engulfing: A larger bullish candle engulfs the previous smaller bearish candle, indicating a potential bullish reversal.
- **Reversal Markers:**
  - Example: Bearish Engulfing: Indicates a potential bearish reversal. Confidence: 75%
- **Continuation Patterns:**
  - Example: Three White Soldiers: Indicates a continuation of the uptrend. Confidence: 80%
- **Sentiment:** <Bullish/Bearish/Neutral>
- **Recommendation:** <BUY/SELL/NEUTRAL>
- **20/200 SMA Strategy Evaluation:**
  - **SMA Position:** <Describe if 20 SMA is above/below 200 SMA>
  - **Trend Evaluation:** <Describe if 20 SMA is in an uptrend/downtrend>
  - **Candle Position:** <Describe if the green buy candle is within 1% of the 20 SMA with little to no wicks>
  - **Strategy Fit:** <Assess how well the stock aligns with the 20/200 SMA strategy and provide specific observations>

---

Use this format to analyze each stock chart and provide the output in a plain text file, listing each stock's analysis sequentially.

### Example of Analysis for Sample Stock Chart

**Example Analysis:**"""

        chatgpt_prompt = st.sidebar.text_area("ChatGPT Prompt", value=prompt_gpt)

    if use_claude:
        prompt_claude = """You are an AI assistant tasked with analyzing stock charts based on the teachings of the book "Candlestick Bible". Your goal is to examine a given chart and provide an analysis of its performance according to the principles outlined in the book.

One Analysis per image uploaded please...  Delinete them by a line break

As part of your analysis, also please look into my trading strategy, make a second section in your response that... Call that section:  20/200 SMA method, and let me know if this stick qualifies with a STRONG YES, YES, NO, STRONG NO options...

Please also indicate strong resitance and support prices

Also when you analyze this, at the vrery bottom I like to trade options, in fact I SELL options, so I nee to know for this stock which options strategies have th ehighest probabilty of successs based on its trend and rating. lets go with the simplest plan, Selling a call or selling a put with X Days to Expiration, at X price, lets say we have resitance at 16.50 , lets try to sell an option that makes the msot sense based on key resitances and support.



My strategy:

Introduction to my stock market thesis
1. Imagine banks have special programs (SMA outfits) that work during a specific time when big financial companies are active (institutional hour sessions).  These banks want to stay ahead, so they sometimes use different ways of studying the stock market (technical analysis) during regular hours and also during extended hours.  This helps them make smart decisions when they want to buy things in the stock market.  Its like they have a plan to be really good at buying and selling stocks.
2. In this context “sma outfits” refers to trading algorithms that use simple moving averages (SMA) in technical analysis. SMA is a way to look at the average price of a stock over a specific period. These algorithms are programmed by banks and financial institutions to analyse stock market trends.
3. Algorithms are active during institutional hours and these institutions may switch between regular hours and extended hours for their technical analysis strategies (this is why you will see on my charts I always analyse stocks with extended hours zones included, both pre-market and after hours) to make informed buying decisions in the stock market.
4. Banks and institutional investors often have sophisticated tools, algorithms and resources that give them an advantage in the stock market over retail traders. They can use many advanced strategies, like the mentioned SMA outfits and technical analysis, to make more informed  decisions and potentially gain an edge in trading. This is why as retail traders we have to follow how banks and market makers trade the markets and understand when algorithms are triggering buying and selling pressure. I have many SMA outfits but this is the follow concept is the easiest to understand and simplest to trade for a beginner.
Contents:
1)	Overview of strategy
1)	SMA’s
1)	Candlesticks
1)	Signal
1)	Examples
1)	Personal trades
1)	Useful Materials
Overview:
Let's make the strategy very easy to understand and break it down to its simplest form.
Components:
1) 20 SMA.
2) 200 SMA.
3) Green buy candle with little/no wicks on the candlestick.
4) Support/resistance levels.
These are the 4 main components to focus on when trading this strategy. Over complicating can lead to overthinking, which will affect entry and exit to a trade. Stay calm, spot the set up, confirm with your analysis and take the trade.
I will be simplifying this version as it is best to start with the basics.
From my own personal trading data which I can share, out of the last 43 trades, 2 have been losers, 3 break even and 37 have been profitable trades in the last 6-7 weeks. 94.5% success rate. When this strategy is done properly, it works very well.
Strategy background/context
1) The U.S. public equity market is operated by every megalodon wealth firm, bank, family office, hedge fund etc. These are the proprietary simple moving average operations used to vehiculate wealth in and out of the Nasdaq S&P500 Dow VIX and every equity on the NYSE NASDAQ AMS.
2) Elite firms have astute precision detection systems that use the preceding Simple Moving Average combinations on multiple timeframes to collectively execute bids, asks, and maximize on every singular point and penny move higher and lower on liquid trading vehicles
3) The only utility that the following information has is the democratization of institutional knowledge, such as how the largest financial entities create selections on the "outfits" for other firms to elect liquidity on for bets, & understanding how financiers transfer capital.
4) There are several pairs of simple moving average outfits, but they are more for index’s where big institutions and whales have alot more liquidity to enter and exit trades with large positions.
5) From my personal trading experience I believe this simple moving average concept involving the 20/200 is the most simple and easy to understand for the positive yielding effects it has. Especially as I have used it across a wide range of stocks from small illiquid market caps to large blue chip stocks.
Simple moving averages background/context
1) 20-Day Simple Moving Average (20-day SMA):
a) The 20-day SMA is a calculation that represents the average closing price of a stock over the last 20 trading days.
b) It is a short-term moving average, providing a more sensitive reflection of recent price changes.
c) Traders use the 20-day SMA to identify short-term trends and potential entry or exit points for trades.
d) When the stock price is above the 20-day SMA, it may be considered a bullish signal, indicating potential upward momentum.
e) Conversely, if the price is below the 20-day SMA, it may be viewed as bearish, suggesting potential downward momentum.
2) 200-Day Simple Moving Average (200-day SMA):
a) The 200-day SMA represents the average closing price of a stock over the last 200 trading days.
b) It is a long-term moving average, providing a smoother and less sensitive reflection of price changes over an extended period.
c) Investors often use the 200-day SMA to identify the overall trend of a stock or the broader market.
d) When the stock price is above the 200-day SMA, it may be considered a bullish signal, indicating a potential long-term uptrend.
e) If the price is below the 200-day SMA, it may be seen as bearish, suggesting a potential long-term downtrend.
Simple moving averages background/context
1) Trend Identification: Both moving averages help traders and investors identify the prevailing trend in a stock's price, whether it's short-term (20-day) or long-term (200-day).
2) Support and Resistance Levels: Moving averages can act as support or resistance levels. Prices often bounce off these averages, providing potential entry or exit points.
3) Signal Confirmation: Crosses between the stock price and the moving averages (e.g., price crossing above or below the moving average) can be used as signals for potential changes in trend direction.
4) Risk Management: Traders may use the relationship between the current price and moving averages to assess risk and set stop-loss levels.
5) Market Sentiment: Moving averages can reflect market sentiment, helping traders gauge whether the market is currently bullish or bearish.
Rules of moving averages - my strategy
1) This strategy works best on the 5 minute chart and the daily chart.
2) For the strategy to work, the 20 sma has to be above the 200 sma.
3) The 20 sma has to be in an uptrend, do not focus on crossovers, the strategy works best when the 20 sma is already above the 200 sma and is an uptrend.
4) Green candlestick just below, equal too, or above the 20 sma (within 1% max) with little to no wicks is the main indicator of buy algorithm. Candlesticks with large wicks, above (shadow) or below is not a strong buy signal and is a 50/50 trade.
5) Do not focus on set ups where the 20 sma is not in an uptrend and very close to the 200 sma.
6) I will go through some examples on both the 5 minute chart and day chart.





This strategy works very similar on the day chart as it does on the 5 minute chart. The exact same principles and rules apply. This is a 6 month trend on meta last year after the October 2022 major bottom across markets and growth tech stocks. 20 sma above the 200 sma and in an uptrend. The green buy candle HAS to be on the 20 sma in an UPTREND!


$PYPL - you can see the moving average outfit start to break down. Price breaks below the 20 sma, opening above and closing below, also causing a crossover of the 20 sma and 200 sma signalling short term selling pressure.




Personal trades
I was going to include screenshots of personal trades I have taken using this method however I don't know whether that will skew your own interpretation and strategy using this. Where I buy and sell might be different to where you would like to buy and sell.I am a micro trader and take profits very quickly - If you would like me to share some let me know and I will add them into a new version of this powerpoint presentation.

Useful tools!
If you follow stock gurus for entries and get dumped on then you are simply not a trader. You are exit liquidity. I think this is the time to be very clear. If you are learning via social media, then the goal is to expedite your ability to trade 100% independently of any stock alert, chat room, or scammer FURU. This is your opportunity to get the tools you need and never look back. NEVER pay to learn. Block every FURU. Do not engage in chat services.Figure out what works. Take it and run.
Tools
Finviz
SEC EDGAR
CNBC Business
TweetDeck
Tradingview
X - best accounts to follow for stock market/financial news = @cnnbrk @CNBCnow @Financialjuice1 @tier10k @reportaznews
@PatentGrants @SPACtrack @DilutionTracker @financialnews @walterbloomberg
Best brokerages (IMO)
Thinkorswim TDA Ameritrade
WeBull
Lightspeed
Interactive Broker
Conclusion
In my entire experience as a day trader, I’ve become exceptionally aware of the calculated behavior of stock manipulation and how it will always coalesse with algorithmic trading.
Let this be an affirmation, technical analysis is the most fundamental aspect of the Live Trade. Technical analysis needs to be simple and a trader needs to have the ability to detail the reason for a trade execution without adjustment to tool parameters.
Being mindful of volatility and in-depth manipulation, I’ve been able to use the following to establish a career as an independent security trader.
Ultimately, this is the heart of day trading.
Key Simple Moving Averages
Only focus on the bullish set up I went through on this video and for the first couple of days instead of live trading, paper trade or watch the market moving against this thesis.
Create a trading journal and document everything useful you see.
Track every single one of your trades and document your mental state when taking a trade - this data will be key to look back in after months of trading to highlight your own trading trends and stock market data. Example - From my first 31 trades this year I have noticed I make the most money between 9  10am & 2 - 3pm EST.  I noticed I dont trade well when my I score my mental state below 6.
You don't need to trade everyday and dont force trades and be greedy. Don't also think you are not making enough money, trading is supposed to be boring but relaxing. The less stress you put on yourself the better.
A clear mind will always trade better than a junk filled mind, sort your personal life out first and get those errands ticked off.






Also when you analyze the stok base don the overall ycandlesick bible, give me a BUY or SELL confidence percent,  so if its a strongest buy its Buy 100 %. If its a sell strongest, its SELL 100 an dso on....
"""
        claude_prompt = st.sidebar.text_area("Claude Prompt", value=prompt_claude)

    # Display selected criteria
    selected_criteria = []
    if use_sma:
        selected_criteria.append("SMA")
    if use_price:
        selected_criteria.append("Price")
    if use_wick:
        selected_criteria.append("Wick")
    if use_macd:
        selected_criteria.append("MACD")
    if use_rsi:
        selected_criteria.append(f"RSI > {rsi_threshold}")
    if use_ltp:
        selected_criteria.append(f"LTP > {ltp_threshold}% above 20D SMA")
    if use_open_interest:
        selected_criteria.append(f"Open Interest > {open_interest_threshold}")
    if use_volume:
        selected_criteria.append(f"Volume > {volume_threshold}")
    
    st.sidebar.write(f"Selected criteria: {', '.join(selected_criteria)}")

    if st.sidebar.button("Run Screener"):
        if not selected_companies:
            st.warning("Please select at least one company.")
        else:
            st.session_state.results = {}
            st.session_state.ai_results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_steps = 1  # Initial screening
            if use_open_interest:
                total_steps += 1  # Open interest screening
            if use_chatgpt or use_claude:
                total_steps += 1  # AI analysis
            current_step = 0

            # Initial screening process (without open interest)
            for i, ticker in enumerate(selected_companies):
                ticker, data = process_stock(ticker, end_date, use_sma, use_price, use_wick, use_macd, use_rsi, use_ltp, use_volume, rsi_threshold, ltp_threshold, volume_threshold)
                if ticker and not data.empty:
                    st.session_state.results[ticker] = data
                
                progress = (i+1) / len(selected_companies)
                progress_bar.progress(progress)
                status_text.text(f"Screening: {i+1}/{len(selected_companies)} stocks. Found {len(st.session_state.results)} matching initial criteria.")

            current_step += 1

            # Apply open interest criterion if selected and if there are any matches
            if use_open_interest and st.session_state.results:
                status_text.text(f"{len(st.session_state.results)} stocks matched initial criteria. Applying Open Interest criterion...")
                st.session_state.results = apply_open_interest_criterion(st.session_state.results, open_interest_threshold)
                current_step += 1

            # AI Analysis process
            if (use_chatgpt or use_claude) and st.session_state.results:
                total_ai_stocks = len(st.session_state.results)
                status_text.text(f"Performing AI analysis on {total_ai_stocks} stocks...")
                for i, (ticker, data) in enumerate(st.session_state.results.items()):
                    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                                        open=data['Open'],
                                                        high=data['High'],
                                                        low=data['Low'],
                                                        close=data['Close'])])
                    fig.update_layout(title=f"{ticker} Stock Chart", width=1200, height=800)
                    img_bytes = fig.to_image(format="png", scale=2)
                    img = Image.open(BytesIO(img_bytes))

                    if use_chatgpt:
                        status_text.text(f"ChatGPT Analysis: {i+1}/{total_ai_stocks} stocks")
                        chatgpt_result = chatgpt_analysis(img, chatgpt_prompt)
                        st.session_state.ai_results[f"{ticker}_chatgpt"] = chatgpt_result

                    if use_claude:
                        status_text.text(f"Claude Analysis: {i+1}/{total_ai_stocks} stocks")
                        claude_result = claude_analysis(img, claude_prompt)
                        st.session_state.ai_results[f"{ticker}_claude"] = claude_result
                        # Add a 60-second delay after each Claude AI request
                        time.sleep(60)

                    progress = (current_step + (i+1)/total_ai_stocks) / total_steps
                    progress_bar.progress(progress)

                current_step += 1

            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")

            if st.session_state.results:
                st.success(f"Found {len(st.session_state.results)} stocks meeting all criteria:")
                st.write(", ".join(st.session_state.results.keys()))
            else:
                st.warning("No stocks found meeting all criteria.")

    # Display download buttons for AI analysis results
    if st.session_state.ai_results:
        st.subheader("Download AI Analysis Results")
        
        col1, col2 = st.columns(2)
        
        # Check if there are any ChatGPT results
        chatgpt_results = {k: v for k, v in st.session_state.ai_results.items() if k.endswith("_chatgpt")}
        if chatgpt_results:
            chatgpt_pdf = create_ai_analysis_pdf(st.session_state.results, 
                                                chatgpt_results, 
                                                use_open_interest, use_volume)
            col1.download_button(
                label="Download ChatGPT Analysis PDF",
                data=chatgpt_pdf,
                file_name="chatgpt_analysis_results.pdf",
                mime="application/pdf"
            )
        
        # Check if there are any Claude AI results
        claude_results = {k: v for k, v in st.session_state.ai_results.items() if k.endswith("_claude")}
        if claude_results:
            claude_pdf = create_ai_analysis_pdf(st.session_state.results, 
                                                claude_results, 
                                                use_open_interest, use_volume)
            col2.download_button(
                label="Download Claude AI Analysis PDF",
                data=claude_pdf,
                file_name="claude_analysis_results.pdf",
                mime="application/pdf"
            )

    # Display detailed results
    if st.session_state.results:
        st.subheader("Detailed Results")
        
        for ticker in st.session_state.results.keys():
            data = st.session_state.results[ticker]
            
            # Create a column layout for the checkbox and graph
            col1, col2 = st.columns([1, 20])
            
            # Add a checkbox in the first column
            with col1:
                is_selected = st.checkbox("", key=f"select_{ticker}", value=ticker in st.session_state.selected_tickers)
                if is_selected and ticker not in st.session_state.selected_tickers:
                    st.session_state.selected_tickers.append(ticker)
                elif not is_selected and ticker in st.session_state.selected_tickers:
                    st.session_state.selected_tickers.remove(ticker)
            
            # Display the graph in the second column
            with col2:
                # Prepare the plot data
                plot_data = [go.Candlestick(x=data.index,
                                            open=data['Open'],
                                            high=data['High'],
                                            low=data['Low'],
                                            close=data['Close'],
                                            name="Candlesticks")]
                
                # Always show SMA 20 and 200
                sma_20 = calculate_sma(data, 20)
                sma_200 = calculate_sma(data, 200)
                plot_data.append(go.Scatter(x=data.index, y=sma_20, name="20 SMA", line=dict(color='blue')))
                plot_data.append(go.Scatter(x=data.index, y=sma_200, name="200 SMA", line=dict(color='red')))
                
                # Create and display the plot
                fig = go.Figure(data=plot_data)
                fig.update_layout(title=f"{ticker} Stock Price and SMAs", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
        
        # Display selected tickers at the bottom
        st.subheader("Selected Tickers")
        st.write(", ".join(st.session_state.selected_tickers))

    st.markdown("---")
    st.markdown("Created with ❤️ using Streamlit and yfinance")

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
