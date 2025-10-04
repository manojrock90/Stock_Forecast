import itertools
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_forecast(df_today, price_type, forecast_steps=4):
    p_range = range(0, 2)
    d_range = range(0, 2)
    q_range = range(0, 2)
    P_range = range(0, 2)
    D_range = range(0, 2)
    Q_range = range(0, 2)
    s = 7  # seasonality (approx 1.75h cycle for 15-min intraday)

    y = df_today[price_type]
    best_aic = float("inf")
    best_order, best_seasonal = None, None

    # Grid search for best SARIMA
    for p, d, q in itertools.product(p_range, d_range, q_range):
        for P, D, Q in itertools.product(P_range, D_range, Q_range):
            try:
                model = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, s))
                results = model.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order, best_seasonal = (p, d, q), (P, D, Q, s)
            except:
                continue

    final_model = SARIMAX(y, order=best_order, seasonal_order=best_seasonal)
    fitted = final_model.fit(disp=False)
    forecast = fitted.forecast(steps=forecast_steps)

    last_time = df_today['Datetime'].iloc[-1]
    forecast_index = pd.date_range(start=last_time + pd.Timedelta(minutes=15),
                                   periods=forecast_steps, freq="15min")

    forecast_df = pd.DataFrame({"Datetime": forecast_index, "Forecast": forecast.values})
    return forecast_df, best_order, best_seasonal
