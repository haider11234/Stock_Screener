import warnings
import csv
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

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
def meets_criteria(data, use_sma, use_price, use_wick, use_macd, use_rsi, use_ltp, rsi_threshold, ltp_threshold):
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

    return True

def process_stock(ticker, end_date, use_sma, use_price, use_wick, use_macd, use_rsi, use_ltp, rsi_threshold, ltp_threshold):
    try:
        start_date = end_date - timedelta(days=365)  # Fetch 1 year of data to ensure we have enough for 200-day SMA
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not data.empty and meets_criteria(data, use_sma, use_price, use_wick, use_macd, use_rsi, use_ltp, rsi_threshold, ltp_threshold):
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

# Streamlit app
st.set_page_config(page_title="Optimized Stock Screener", layout="wide")

# Initialize session state for login and results
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = []

# Login screen
if not st.session_state.logged_in:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(username, password):
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")
else:
    # Main app content
    st.title("📊 Optimized Stock Screener")

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

    # Threshold inputs
    rsi_threshold = 50
    ltp_threshold = 10
    if use_rsi:
        rsi_threshold = st.sidebar.number_input("RSI Threshold", min_value=0, max_value=100, value=50)
    if use_ltp:
        ltp_threshold = st.sidebar.number_input("LTP above 20D SMA Threshold (%)", min_value=0, max_value=100, value=10)

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
    
    st.sidebar.write(f"Selected criteria: {', '.join(selected_criteria)}")

    if st.sidebar.button("Run Screener"):
        if not selected_companies:
            st.warning("Please select at least one company.")
        else:
            st.session_state.results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, ticker in enumerate(selected_companies):
                ticker, data = process_stock(ticker, end_date, use_sma, use_price, use_wick, use_macd, use_rsi, use_ltp, rsi_threshold, ltp_threshold)
                if ticker and not data.empty:
                    st.session_state.results[ticker] = data
                
                # Update progress
                progress = (i + 1) / len(selected_companies)
                progress_bar.progress(progress)
                status_text.text(f"Processed {i+1}/{len(selected_companies)} stocks. Found {len(st.session_state.results)} matching criteria.")
            
            if st.session_state.results:
                st.success(f"Found {len(st.session_state.results)} stocks meeting the criteria:")
                st.write(", ".join(st.session_state.results.keys()))
            else:
                st.warning("No stocks found meeting the criteria.")

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
                # Prepare the plot data based on selected criteria
                plot_data = [go.Candlestick(x=data.index,
                                            open=data['Open'],
                                            high=data['High'],
                                            low=data['Low'],
                                            close=data['Close'],
                                            name="Candlesticks")]
                
                if use_sma:
                    sma_20 = calculate_sma(data, 20)
                    sma_200 = calculate_sma(data, 200)
                    plot_data.append(go.Scatter(x=data.index, y=sma_20, name="20 SMA", line=dict(color='blue')))
                    plot_data.append(go.Scatter(x=data.index, y=sma_200, name="200 SMA", line=dict(color='red')))
                
                if use_macd:
                    macd, signal_line = calculate_macd(data)
                    plot_data.append(go.Scatter(x=data.index, y=macd, name="MACD", line=dict(color='blue')))
                    plot_data.append(go.Scatter(x=data.index, y=signal_line, name="Signal Line", line=dict(color='red')))
                
                if use_rsi:
                    rsi = calculate_rsi(data)
                    plot_data.append(go.Scatter(x=data.index, y=rsi, name="RSI", line=dict(color='purple')))
                
                if use_ltp:
                    sma_20 = calculate_sma(data, 20)
                    ltp_above_sma = ((data['Close'] - sma_20) / sma_20) * 100
                    plot_data.append(go.Scatter(x=data.index, y=ltp_above_sma, name="LTP above 20D SMA (%)", line=dict(color='green')))
                
                # Create and display the plot
                fig = go.Figure(data=plot_data)
                fig.update_layout(title=f"{ticker} Stock Price and Selected Indicators", xaxis_title="Date", yaxis_title="Price/Value")
                st.plotly_chart(fig, use_container_width=True)
        
        # Display selected tickers at the bottom
        st.subheader("Selected Tickers")
        st.write(", ".join(st.session_state.selected_tickers))

    st.markdown("---")
    st.markdown("Created with ❤️ using Streamlit and yfinance")

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()
