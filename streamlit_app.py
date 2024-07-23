import warnings
import csv
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# Function to calculate SMAs
def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

# Function to check if a stock meets the criteria
def meets_criteria(data):
    if len(data) < 201:
        return False
    
    current_data = data.iloc[-1]
    previous_data = data.iloc[-2]
    
    # SMA Criteria
    sma_20 = calculate_sma(data, 20)
    sma_200 = calculate_sma(data, 200)

    if sma_20.iloc[-1] <= sma_200.iloc[-1]:
        return False
    if sma_20.iloc[-1] <= sma_20.iloc[-2]:
        return False
    
    # Price Criteria
    if current_data['Close'] > 1.01 * sma_20.iloc[-1]:
        return False
    if current_data['Close'] <= current_data['Open']:
        return False
    if (current_data['Close'] - current_data['Open']) / current_data['Open'] <= 0.005:
        return False
    
    return True

def process_stock(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if meets_criteria(data):
            return ticker
    except Exception as e:
        pass
    return None

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

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

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

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_range = st.sidebar.date_input("Select date range:", [start_date, end_date])

    if st.sidebar.button("Run Screener"):
        if not selected_companies:
            st.warning("Please select at least one company.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            total_stocks = len(selected_companies)
            processed_stocks = 0
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_stock = {executor.submit(process_stock, ticker, date_range[0], date_range[1]): ticker for ticker in selected_companies}
                for future in as_completed(future_to_stock):
                    processed_stocks += 1
                    result = future.result()
                    if result:
                        results.append(result)
                    
                    progress_bar.progress(processed_stocks / total_stocks)
                    status_text.text(f"Processed {processed_stocks}/{total_stocks} stocks. Found {len(results)} matching criteria.")
            
            if results:
                st.success(f"Found {len(results)} stocks meeting the criteria:")
                st.write(", ".join(results))
                
                # Display detailed results
                selected_ticker = st.selectbox("Select a stock for detailed view:", results)
                
                if selected_ticker:
                    data = yf.download(selected_ticker, start=date_range[0], end=date_range[1])
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
                    
                    # Display last 5 days of data
                    st.subheader("Last 5 Days of Data")
                    st.dataframe(data.tail().style.format({"Open": "${:.2f}", "High": "${:.2f}", "Low": "${:.2f}", "Close": "${:.2f}", "Volume": "{:,.0f}"}))
            else:
                st.warning("No stocks found meeting the criteria.")

    st.markdown("---")
    st.markdown("Created with ❤️ using Streamlit and yfinance")

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()
