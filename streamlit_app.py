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



st.set_page_config(page_title="AI-Powered Stock Screener", layout="wide")

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

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
        if 'Open Interest' in data.columns and current_data['Open Interest'] < open_interest_threshold:
            return False

    if use_volume:
        # Volume Criteria
        if current_data['Volume'] < volume_threshold:
            return False

    return True

def process_stock(ticker, end_date, use_sma, use_price, use_wick, use_macd, use_rsi, use_ltp, use_open_interest, use_volume, rsi_threshold, ltp_threshold, open_interest_threshold, volume_threshold):
    try:
        start_date = end_date - timedelta(days=365)  # Fetch 1 year of data to ensure we have enough for 200-day SMA
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not data.empty and meets_criteria(data, use_sma, use_price, use_wick, use_macd, use_rsi, use_ltp, use_open_interest, use_volume, rsi_threshold, ltp_threshold, open_interest_threshold, volume_threshold):
            return ticker, data
    except Exception as e:
        st.error(f"Error processing {ticker}: {str(e)}")
    return None, None

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
    enc = b'gAAAAABmq6-p9GnMYojabe5kf7qtzlldt9QENiov8KJSaf6_Za8F5OH22fjPYmwqCtseXM9IO-Dth8SgHjAmK41zpFR7yEyQRQLtfNO3Y-2gpgRifMv0haNA3-eDColPcRLpGXVtM4baI-RWQVj18ovRRPijGF5nYQ=='

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
                story.append(Paragraph("Open Interest: Not available", normal_style))
        
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
        chatgpt_prompt = st.sidebar.text_area("ChatGPT Prompt", value="Analyze this stock chart and provide insights.")

    if use_claude:
        claude_prompt = st.sidebar.text_area("Claude Prompt", value="Analyze this stock chart and provide insights.")

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

            total_steps = len(selected_companies)
            if use_chatgpt or use_claude:
                total_steps *= 2  # Double the steps to account for AI analysis
            current_step = 0

            # Screening process
            for i, ticker in enumerate(selected_companies):
                ticker, data = process_stock(ticker, end_date, use_sma, use_price, use_wick, use_macd, use_rsi, use_ltp, use_open_interest, use_volume, rsi_threshold, ltp_threshold, open_interest_threshold, volume_threshold)
                if ticker and not data.empty:
                    st.session_state.results[ticker] = data
                
                current_step += 1
                progress = current_step / total_steps
                progress_bar.progress(progress)
                status_text.text(f"Screening: {i+1}/{len(selected_companies)} stocks. Found {len(st.session_state.results)} matching criteria.")

            # AI Analysis process
            if (use_chatgpt or use_claude) and st.session_state.results:
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
                        status_text.text(f"ChatGPT Analysis: {i+1}/{len(st.session_state.results)} stocks")
                        chatgpt_result = chatgpt_analysis(img, chatgpt_prompt)
                        st.session_state.ai_results[f"{ticker}_chatgpt"] = chatgpt_result
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)

                    if use_claude:
                        status_text.text(f"Claude Analysis: {i+1}/{len(st.session_state.results)} stocks")
                        claude_result = claude_analysis(img, claude_prompt)
                        st.session_state.ai_results[f"{ticker}_claude"] = claude_result
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)

            progress_bar.progress(1.0)  # Ensure the progress bar reaches 100%
            status_text.text("Analysis complete!")

            if st.session_state.results:
                st.success(f"Found {len(st.session_state.results)} stocks meeting the criteria:")
                st.write(", ".join(st.session_state.results.keys()))
            else:
                st.warning("No stocks found meeting the criteria.")

    # Display download buttons for AI analysis results
    if st.session_state.ai_results:
        st.subheader("Download AI Analysis Results")
        
        col1, col2 = st.columns(2)
        
        if use_chatgpt:
            chatgpt_pdf = create_ai_analysis_pdf(st.session_state.results, 
                                                 {k: v for k, v in st.session_state.ai_results.items() if k.endswith("_chatgpt")}, 
                                                 use_open_interest, use_volume)
            col1.download_button(
                label="Download ChatGPT Analysis PDF",
                data=chatgpt_pdf,
                file_name="chatgpt_analysis_results.pdf",
                mime="application/pdf"
            )
        
        if use_claude:
            claude_pdf = create_ai_analysis_pdf(st.session_state.results, 
                                                {k: v for k, v in st.session_state.ai_results.items() if k.endswith("_claude")}, 
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
