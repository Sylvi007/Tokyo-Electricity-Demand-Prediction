"""
Use the trained model (demand_model.joblib) to predict electricity demand for a whole day.

    python predict.py                                 # example: a hot summer workday and a mild Sunday
    python predict.py 2026-08-10 35 27 Workday         # date, daily max °C, daily min °C, day type

You give the date, the forecast maximum and minimum temperature, and whether it is a workday.
The script builds a simple hourly temperature curve (coolest around 05:00, warmest around 14:00)
and predicts the demand for every hour of that day.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
bundle = joblib.load(HERE / "demand_model.joblib")
MODEL, FEATURES = bundle["model"], bundle["features"]


def hourly_temperatures(t_max: float, t_min: float) -> np.ndarray:
    """Simple daily temperature curve: minimum at 05:00, maximum at 14:00."""
    hours = np.arange(24)
    shape = (np.cos((hours - 14) / 24 * 2 * np.pi) + 1) / 2          # 1 at 14:00, 0 at 02:00
    shape = (shape - shape.min()) / (shape.max() - shape.min())
    return t_min + (t_max - t_min) * shape


def make_features(date: str, t_max: float, t_min: float, workday: bool, humidity: float = 65.0) -> pd.DataFrame:
    idx = pd.date_range(date, periods=24, freq="h")
    temp = hourly_temperatures(t_max, t_min)
    f = pd.DataFrame(index=idx)
    f["temp_tokyo_c"] = temp
    f["temp_kanto_c"] = temp
    f["temp_kanto_24h_mean_c"] = temp.mean()
    f["temp_kanto_max_today_c"] = t_max
    f["cooling_degrees"] = np.clip(temp - 22, 0, None)
    f["heating_degrees"] = np.clip(16 - temp, 0, None)
    f["rh_tokyo_pct"] = humidity
    f["sun_kanto_h"] = np.where((idx.hour >= 7) & (idx.hour <= 16), 0.6, 0.0)
    f["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    f["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    f["hour"] = idx.hour
    f["dayofweek"] = idx.dayofweek
    f["month"] = idx.month
    f["is_workday"] = int(workday)
    f["is_holiday"] = 0
    return f[FEATURES]


def predict_day(date: str, t_max: float, t_min: float, workday: bool) -> pd.DataFrame:
    X = make_features(date, t_max, t_min, workday)
    return pd.DataFrame({"temp_c": X["temp_tokyo_c"].round(1).to_numpy(), "predicted_demand_gw": MODEL.predict(X).round(1)},
                        index=X.index.strftime("%H:00"))


def show(date, t_max, t_min, workday):
    p = predict_day(date, t_max, t_min, workday)
    peak = p["predicted_demand_gw"].idxmax()
    print(f"\n{date} ({'workday' if workday else 'weekend/holiday'}), forecast {t_min}–{t_max} °C")
    print(f"Predicted peak: {p.loc[peak, 'predicted_demand_gw']} GW at {peak} | "
          f"lowest: {p['predicted_demand_gw'].min()} GW | daily energy: {p['predicted_demand_gw'].sum():.0f} GWh")
    print(p.T.to_string())
    return p


if __name__ == "__main__":
    print(f"Model: {bundle['name']}")
    if len(sys.argv) == 5:
        show(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4].lower().startswith("work"))
    else:
        show("2026-08-10", 35.0, 27.0, True)     # hot summer workday
        show("2026-04-19", 21.0, 12.0, False)    # mild spring Sunday
        show("2027-01-20", 8.0, 1.0, True)       # cold winter workday
