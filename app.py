import os
import io
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sarima_forecast import sarima_forecast


st.set_page_config(page_title="Intraday Forecast", layout="wide")
st.title("📈 Intraday Forecast (15-min intervals)")

ticker_df = pd.read_csv(r"C:\Users\Manoj\Downloads\Stock_Forecast_V2\ind_nifty100list.csv")
nifty100_tickers = dict(zip(ticker_df['Company Name'], ticker_df['Symbol'] + ".NS"))

selected_stock = st.selectbox("Select a stock", list(nifty100_tickers.keys()))
price_type = st.selectbox("Select Price Type to Forecast", ["Open", "High", "Low", "Close"], index=3)

window_days = st.number_input(
    "How many previous days of data would you like to use for training your model(1 ≤ days ≤ 60)?",
    min_value=1, max_value=60, value=30
)

forecast_steps = st.number_input(
    "Forecast steps ahead (15-min intervals)",
    min_value=1, max_value=12, value=4
)


@st.cache_data
def load_data(ticker, period):
    df = yf.Ticker(ticker).history(period=f"{period}d", interval="15m").reset_index()
    df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
    return df

df = load_data(nifty100_tickers[selected_stock], window_days + 2)

if df.empty:
    st.warning("No intraday data available for the selected period.")
    st.stop()

last_date = df['Datetime'].max().date()
start_date = last_date - pd.Timedelta(days=window_days - 1)
df_window = df[df['Datetime'].dt.date >= start_date].copy()

train_start = df_window['Datetime'].min()
train_end = df_window['Datetime'].max()
st.info(f"🕒 **Training Data Range:** {train_start.strftime('%Y-%m-%d %H:%M')} → {train_end.strftime('%Y-%m-%d %H:%M')}")

if df_window.empty:
    st.warning("Not enough data in the selected window.")
    st.stop()
forecast_df = pd.DataFrame()   

if st.button("Run Forecast"):
    with st.spinner("Training your model... this may take a few seconds ⏳"):
        forecast_df = sarima_forecast(df_window, price_type, forecast_steps)
    st.success("Forecast completed ✅")

 

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

if not forecast_df.empty:
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
if not forecast_df.empty:
    st.dataframe(forecast_df)
else:
    st.info("Click 'Run Forecast' to generate predictions.")


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



# -----------------------------
# 🔹 After 15:15 — Generate full-day actual vs forecast CSV
# -----------------------------
now = datetime.now()
if now.time() >= time(15, 15):
    st.markdown("---")
    st.header("📊 End-of-Day Actual vs Forecast (09:15–15:15)")

    today = now.date()
    df_today = df[df['Datetime'].dt.date == today].copy()
    df_today = df_today[(df_today['Datetime'].dt.time >= time(9, 15)) &
                        (df_today['Datetime'].dt.time <= time(15, 15))]

    if not df_today.empty:
        # Train model on previous N days (excluding today's data)
        df_past = df[df['Datetime'].dt.date < today]
        train_past = df_past[price_type]

        if not train_past.empty:
            best_order, best_seasonal, _ = get_best_sarima(train_past)
            model = SARIMAX(train_past,
                            order=best_order,
                            seasonal_order=best_seasonal,
                            enforce_stationarity=True,
                            enforce_invertibility=True)
            model_fit = model.fit(disp=False)

            # Forecast for full trading session timestamps
            forecast_index = df_today['Datetime']
            forecast_values = model_fit.forecast(steps=len(forecast_index))

            final_df = pd.DataFrame({
                "Datetime": forecast_index,
                "Actual": df_today[price_type].values,
                "Forecast": forecast_values.values
            })

            # Compute MAPE
            final_df["MAPE (%)"] = abs(final_df["Actual"] - final_df["Forecast"]) / abs(final_df["Actual"]) * 100
            mean_mape = final_df["MAPE (%)"].mean()

            st.success(f"📈 Mean MAPE for today's session: **{mean_mape:.2f}%**")
            st.dataframe(final_df.style.format({"Actual": "{:.2f}", "Forecast": "{:.2f}", "MAPE (%)": "{:.2f}"}))

            # Download CSV
            csv_buffer = io.StringIO()
            final_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            filename = f"{selected_stock.replace(' ', '_')}_Actual_vs_Forecast_{today}.csv"

            st.download_button(
                label="📥 Download Full-Day Actual vs Forecast CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv"
            )

        else:
            st.warning("⚠️ Not enough past data to train SARIMA model for today's end-of-day forecast.")
    else:
        st.warning("⚠️ No data found for today's trading session (09:15–15:15).")
