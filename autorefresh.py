import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Live Forecast Dashboard", layout="wide")
st.title("📈 Live Stock Price vs Forecast")

# --- Inputs ---
ticker = st.selectbox("Select Stock", ["TCS.NS", "INFY.NS", "RELIANCE.NS"])
forecast_steps = 4

# --- Refresh every 1 min ---
st_autorefresh = st_autorefresh(interval=60*1000, key="refresh")

# --- Load latest intraday data (last 1 day) ---
df = yf.download(ticker, period="1d", interval="1m").reset_index()
df['Datetime'] = pd.to_datetime(df['Datetime'])

# --- Dummy Forecast (replace with SARIMA/Prophet) ---
last_time = df['Datetime'].iloc[-1]
forecast_index = pd.date_range(start=last_time + pd.Timedelta(minutes=1),
                               periods=forecast_steps, freq="1min")
forecast_values = [df['Close'].iloc[-1] * (1 + i*0.001) for i in range(forecast_steps)]
forecast_df = pd.DataFrame({"Datetime": forecast_index, "Forecast": forecast_values})

# --- Plot ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Datetime"], y=df["Close"], mode="lines", name="Actual Price", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=forecast_df["Datetime"], y=forecast_df["Forecast"], mode="lines+markers", 
                         name="Forecast", line=dict(color="red", dash="dash")))

fig.update_layout(title=f"Live {ticker} Price vs Forecast",
                  xaxis_title="Time", yaxis_title="Price",
                  template="plotly_white", hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)
