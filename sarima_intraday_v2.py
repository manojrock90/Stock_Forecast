# streamlit_sarima_moving_window.py
import streamlit as st
import yfinance as yf
import pandas as pd
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import io
# session = requests.Session()
# session.headers.update({'User-Agent': 'Mozilla/5.0'})
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Intraday Forecast", layout="wide")
st.title("📈 Intraday Forecast (15-min intervals)")

# -----------------------------
# Load NIFTY100 tickers
# -----------------------------
# ticker_df = pd.read_csv(r"C:\Users\Manoj\Downloads\Stock_Forecast\ind_nifty100list.csv")
ticker_df = pd.read_csv("s3://niftystock/data/")
nifty100_tickers = dict(zip(ticker_df['Company Name'], ticker_df['Symbol'] + ".NS"))

# -----------------------------
# User Inputs
# -----------------------------
selected_stock = st.selectbox("Select a stock", list(nifty100_tickers.keys()))
price_type = st.selectbox("Select Price Type to Forecast", ["Open", "High", "Low", "Close"], index=3)

window_days = st.number_input(
    "How many past days to use for training your Model?", min_value=1, max_value=60, value=30
)

forecast_steps = st.number_input(
    "Forecast steps ahead (15-min intervals)", min_value=1, max_value=12, value=4
)

# -----------------------------
# Load historical intraday data
# -----------------------------
@st.cache_data
def load_data(ticker, period):
    df = yf.Ticker(ticker).history(period=f"{period}d", interval="15m").reset_index()
    df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
    return df

# Download extra 2 days as buffer in case of missing data
df = load_data(nifty100_tickers[selected_stock], window_days + 2)

if df.empty:
    st.warning("No intraday data available for the selected period.")
    st.stop()

# -----------------------------
# Dynamic moving window: last N days
# -----------------------------
last_date = df['Datetime'].max().date()
start_date = last_date - pd.Timedelta(days=window_days-1)
df_window = df[df['Datetime'].dt.date >= start_date].copy()

train_start = df_window['Datetime'].min()
train_end = df_window['Datetime'].max()
st.info(f"🕒 **Training Data Range:** {train_start.strftime('%Y-%m-%d %H:%M')} → {train_end.strftime('%Y-%m-%d %H:%M')}")

if df_window.empty:
    st.warning("Not enough data in the selected window.")
    st.stop()

train = df_window[price_type]


# -----------------------------
# SARIMA parameter ranges
# -----------------------------
p_range = range(0, 2)
d_range = range(0, 2)
q_range = range(0, 2)
P_range = range(0, 2)
D_range = range(0, 2)
Q_range = range(0, 2)

# Dynamic seasonality: number of 15-min intervals in a trading day (09:15–15:30)
trading_minutes = 6 * 60 + 15  # 6h15m
# s = trading_minutes // 15  # ~25 intervals
s = 7

# -----------------------------
# Find best SARIMA order
# -----------------------------
def get_best_sarima(y):
    best_aic = float("inf")
    best_order = None
    best_seasonal = None
    best_res = None
    for p, d, q in itertools.product(p_range, d_range, q_range):
        for P, D, Q in itertools.product(P_range, D_range, Q_range):
            try:
                mod = SARIMAX(
                    y,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=True,
                    enforce_invertibility=True
                )
                out = mod.fit(disp=False)
                if out.aic < best_aic:
                    best_aic = out.aic
                    best_order = (p, d, q)
                    best_seasonal = (P, D, Q, s)
                    best_res = out
            except Exception:
                continue
    return best_order, best_seasonal, best_res

# -----------------------------
# Run forecast button
# -----------------------------
if st.button("Run Forecast"):

    best_order, best_seasonal, _ = get_best_sarima(train)

    model = SARIMAX(
        train,
        order=best_order,
        seasonal_order=best_seasonal,
        enforce_stationarity=True,
        enforce_invertibility=True
    )
    model_fit = model.fit(disp=False)

    # Forecast next 'forecast_steps'
    last_time = df_window['Datetime'].iloc[-1]
    forecast_index = pd.date_range(
        start=last_time + pd.Timedelta(minutes=15),
        periods=forecast_steps,
        freq="15min"
    )
    forecast_values = model_fit.forecast(steps=forecast_steps)
    forecast_df = pd.DataFrame({"Datetime": forecast_index, "Forecast": forecast_values.values})

    # -----------------------------
    # Plotting
    # -----------------------------
    today = datetime.now().date()
    df_today = df[df['Datetime'].dt.date == today]

    fig = go.Figure()

    if not df_today.empty:
        fig.add_trace(go.Scatter(
            x=df_today["Datetime"],
            y=df_today[price_type],
            mode='lines+markers',
            name="Today's Actuals",
            line=dict(color="blue")
        ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Datetime"],
        y=forecast_df["Forecast"],
        mode='lines+markers',
        name=f"Forecast ({forecast_steps*15} mins ahead)",
        line=dict(color="red", dash="dash")
    ))

    fig.update_layout(
        title=f"{selected_stock} Forecast (Next {forecast_steps*15} mins) with {window_days}-day Moving Window",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0)")
    )

    st.plotly_chart(fig, use_container_width=True)
    st.subheader(f"Forecast Table for {selected_stock} ({price_type})")
    st.dataframe(forecast_df)


    csv_buffer = io.StringIO()
    forecast_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    stock_safe_name = selected_stock.replace(" ", "_").replace("&", "and")
    filename = f"{stock_safe_name}_forecast_{timestamp}.csv"

    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv_data,
        file_name=filename,
        mime="text/csv"
    )


# https://chatgpt.com/share/68e01bd2-62a0-8006-92eb-ed6b88f2959a
