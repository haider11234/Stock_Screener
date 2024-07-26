import warnings
import csv
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# Function to calculate SMAs
def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

# Function to check if a stock meets the criteria
def meets_criteria(data, use_sma, use_price, use_wick):
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

    return True

def process_stock(ticker, end_date, use_sma, use_price, use_wick):
    try:
        start_date = end_date - timedelta(days=365)  # Fetch 1 year of data to ensure we have enough for 200-day SMA
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if not data.empty and meets_criteria(data, use_sma, use_price, use_wick):
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

    # Display selected criteria
    selected_criteria = []
    if use_sma:
        selected_criteria.append("SMA")
    if use_price:
        selected_criteria.append("Price")
    if use_wick:
        selected_criteria.append("Wick")
    
    st.sidebar.write(f"Selected criteria: {', '.join(selected_criteria)}")

    if st.sidebar.button("Run Screener"):
        if not selected_companies:
            st.warning("Please select at least one company.")
        else:
            st.session_state.results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, ticker in enumerate(selected_companies):
                ticker, data = process_stock(ticker, end_date, use_sma, use_price, use_wick)
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
        selected_ticker = st.selectbox("Select a stock for detailed view:", list(st.session_state.results.keys()))
        
        if selected_ticker:
            data = st.session_state.results[selected_ticker]
            sma_20 = calculate_sma(data, 20)
            sma_200 = calculate_sma(data, 200)
            
            # Candlestick chart
            fig = go.Figure(data=[go.Candlestick(x=data.index,
                                                 open=data['Open'],
                                                 high=data['High'],
                                                 low=data['Low'],
                                                 close=data['Close'],
                                                 name="Candlesticks"),
                                  go.Scatter(x=data.index, y=sma_20, name="20 SMA", line=dict(color='blue')),
                                  go.Scatter(x=data.index, y=sma_200, name="200 SMA", line=dict(color='red'))])
            
            fig.update_layout(title=f"{selected_ticker} Stock Price", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display last 5 days of data with SMA
            st.subheader("Last 5 Days of Data (including SMA)")
            display_data = data.tail().copy()
            display_data['SMA_20'] = sma_20.tail()
            display_data['SMA_200'] = sma_200.tail()
            st.dataframe(display_data.style.format({
                "Open": "${:.2f}", 
                "High": "${:.2f}", 
                "Low": "${:.2f}", 
                "Close": "${:.2f}", 
                "Volume": "{:,.0f}",
                "SMA_20": "${:.2f}",
                "SMA_200": "${:.2f}"
            }))

    st.markdown("---")
    st.markdown("Created with ❤️ using Streamlit and yfinance")

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()
