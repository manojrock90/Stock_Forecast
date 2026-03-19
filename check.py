import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from EDA import Report

st.title("📊 Stock Market Dashboard")

st.header("1️⃣ Data Loading Section")
# st.write("Upload or fetch stock data here...")
ticker_df = pd.read_csv(r"C:\Users\Manoj\Downloads\Stock_Forecast\ind_nifty100list.csv")
nifty100_tickers = dict(zip(ticker_df['Company Name'], ticker_df['Symbol'] + ".NS"))

selected_stock = st.selectbox("Select a stock", list(nifty100_tickers.keys()))
price_type = st.selectbox("Select Price Type to Forecast", ["Open", "High", "Low", "Close"], index=3)

st.divider()  # horizontal line

st.header("2️⃣ Exploratory Data Analysis (EDA)")
@st.cache_data
def load_data(ticker, period=30):
    df = yf.Ticker(ticker).history(period=f"{period}d", interval="15m").reset_index()
    df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
    return df
df = load_data(nifty100_tickers[selected_stock])

df['Datetime'] = pd.to_datetime(df['Datetime'])
start_date = df['Datetime'].min()
end_date = df['Datetime'].max()

df_eda = df[(df['Datetime']).between(start_date, end_date)]
sma = st.number_input(
    "Simple moving average",
    min_value=1, max_value=20, value=7, key = 'sma_input')
ewa = st.number_input(
    "Exponential moving average",
    min_value=1, max_value=20, value=7,key = 'ema_input')
volatility = st.number_input(
    "Volatility",
    min_value=1, max_value=20, value=7,key = 'Volatility_input')
rsi = st.number_input(
    "RSI",
    min_value=1, max_value=20, value=7,key = 'key_input')

if st.button("Plot EDA"):
    with st.spinner("Please be waited,your high level EDA is coming out.. ⏳"):

    # st.write("Show plots and statistics here...")
        report = Report(df_eda, price_type, sma, ewa,volatility,rsi)
        result = report.technical_indicators()
        result.head(2)
    st.success("EDA completed Successfully ✅")    

st.divider()

st.header("3️⃣ Model Forecasting Section")
st.write("Display model results and predictions here...")



tab1, tab2, tab3 = st.tabs(["📥 Data", "📈 EDA", "🤖 Forecast"])

with tab1:
    st.subheader("Upload or fetch your data")
    st.file_uploader("Upload CSV file")

with tab2:
    st.subheader("Data Exploration")
    st.line_chart([1, 2, 3])

with tab3:
    st.subheader("Forecast Results")
    st.write("Coming soon...")

