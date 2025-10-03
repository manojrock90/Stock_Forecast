# streamlit_sarima_intraday.py
import streamlit as st
import yfinance as yf
import pandas as pd
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Intraday Forecast", layout="wide")
st.title("📈 Intraday Forecast (15-min intervals)")

# -----------------------------
# Load NIFTY100 tickers
# -----------------------------
ticker_df = pd.read_csv(r"C:\Users\Manoj\Downloads\Stock_Forecast\ind_nifty100list.csv")
nifty100_tickers = dict(zip(ticker_df['Company Name'], ticker_df['Symbol'] + ".NS"))

selected_stock = st.selectbox("Select a stock", list(nifty100_tickers.keys()))
days = st.number_input("Number of past days to use for forecast", min_value=1, max_value=60, value=5)

price_type = st.selectbox(
    "Select Price Type to Forecast",
    ["Open", "High", "Low", "Close"],
    index=3  # default = Close
)

# Run forecast button
if st.button("Run Forecast"):
    # -----------------------------
    # Load historical data
    # -----------------------------
    @st.cache_data
    def load_data(ticker, period):
        df = yf.Ticker(ticker).history(period=f"{period}d", interval="15m").reset_index()
        df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
        return df

    df = load_data(nifty100_tickers[selected_stock], days)

    # Filter today's data
    today = datetime.now().date()
    df_today = df[df['Datetime'].dt.date == today].copy()

    if df_today.empty:
        st.warning("No intraday data available for today yet. Try again during market hours.")
        st.stop()

    # -----------------------------
    # SARIMA parameter ranges
    # -----------------------------
    p_range = range(0, 2)
    d_range = range(0, 2)
    q_range = range(0, 2)

    P_range = range(0, 2)
    D_range = range(0, 2)
    Q_range = range(0, 2)

    # For intraday 15-min data, daily seasonality = ~25 steps (09:15-15:30)
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

    train = df_today[price_type]
    best_order, best_seasonal, _ = get_best_sarima(train)

    # -----------------------------
    # Train SARIMA & forecast next 1 hour (4 x 15-min steps)
    # -----------------------------
    model = SARIMAX(train,
                    order=best_order,
                    seasonal_order=best_seasonal,
                    enforce_stationarity=True,
                    enforce_invertibility=True)
    model_fit = model.fit(disp=False)

    forecast = model_fit.forecast(steps=4)
    last_time = df_today['Datetime'].iloc[-1]
    forecast_index = pd.date_range(start=last_time + pd.Timedelta(minutes=15),
                                   periods=4, freq="15min")
    forecast_df = pd.DataFrame({"Datetime": forecast_index, "Forecast": forecast.values})

    # -----------------------------
    # Plot interactive chart
    # -----------------------------
    fig = go.Figure()

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
        name="Forecast (next 1h)",
        line=dict(color="red", dash="dash")
    ))

    fig.update_layout(
        title=f"{selected_stock} Forecast for {today} (Next 1 hour)",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0)")
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Show forecast table
    # -----------------------------
    st.subheader("Forecast Table (Next 1 Hour)")
    st.dataframe(forecast_df)
