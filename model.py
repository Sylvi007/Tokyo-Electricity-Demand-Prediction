"""
Predict hourly electricity demand in the Tokyo (TEPCO) area from weather and calendar features.

Data: tokyo_hourly_clean.csv.gz - 39,744 hourly records, 2022-04-01 to 2026-10-12 (JST)
  * Target: demand_gw (TEPCO Power Grid "でんき予報")
  * Inputs: Japan Meteorological Agency weather + the Japanese calendar

Run:  python model.py
  -> trains 4 models, prints the comparison, saves the best model (demand_model.joblib),
     test-set predictions (test_predictions.csv), metrics (model_comparison.csv) and charts (fig_*.png).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE / "tokyo_hourly_clean.csv.gz"
TEST_START = "2026-01-01"   # train: Apr 2022 - Dec 2025, test: Jan - Oct 2026 (never seen in training)
RANDOM_STATE = 42

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

# Features known in advance of the hour being predicted: the weather (from a forecast) and the calendar.
# Solar output, TEPCO's own forecast and past demand are deliberately NOT used as inputs.
NUMERIC = ["temp_tokyo_c", "temp_kanto_c", "temp_kanto_24h_mean_c", "temp_kanto_max_today_c",
           "cooling_degrees", "heating_degrees",
           "rh_tokyo_pct", "sun_kanto_h", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
CATEGORICAL = ["hour", "dayofweek", "month"]
BINARY = ["is_workday", "is_holiday"]
FEATURES = NUMERIC + CATEGORICAL + BINARY
TARGET = "demand_gw"

FEATURE_LABELS = {
    "temp_tokyo_c": "Tokyo temperature", "temp_kanto_c": "Kanto temperature (pop.-weighted)",
    "temp_kanto_24h_mean_c": "Kanto temperature, last 24 h mean", "temp_kanto_max_today_c": "Today's max Kanto temp.",
    "cooling_degrees": "Cooling degrees (°C above 22)", "heating_degrees": "Heating degrees (°C below 16)",
    "rh_tokyo_pct": "Tokyo humidity", "sun_kanto_h": "Sunshine", "hour_sin": "Hour (sin)", "hour_cos": "Hour (cos)",
    "doy_sin": "Day of year (sin)", "doy_cos": "Day of year (cos)", "hour": "Hour of day", "dayofweek": "Day of week",
    "month": "Month", "is_workday": "Workday", "is_holiday": "Holiday",
}


# --------------------------------------------------------------------------------------
# 1. Prepare the dataset
# --------------------------------------------------------------------------------------
def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["datetime"], index_col="datetime").sort_index()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dayofweek"] = out.index.dayofweek
    out["is_workday"] = out["is_workday"].astype(int)
    out["is_holiday"] = out["is_holiday"].astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    doy = out.index.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # Heat builds up in buildings: the average temperature of the previous 24 hours matters.
    out["temp_kanto_24h_mean_c"] = out["temp_kanto_c"].rolling(24, min_periods=1).mean()
    # Demand rises on BOTH sides of a comfortable temperature (U-shape). These two features let even a
    # linear model capture that: degrees above 22 °C drive air conditioning, degrees below 16 °C drive heating.
    out["cooling_degrees"] = (out["temp_kanto_c"] - 22).clip(lower=0)
    out["heating_degrees"] = (16 - out["temp_kanto_c"]).clip(lower=0)
    # Daily maximum temperature (a standard item in every weather forecast).
    out["temp_kanto_max_today_c"] = out.groupby(out.index.normalize())["temp_kanto_c"].transform("max")
    # Humidity has a few multi-day station outages: fill with the value for the same hour on nearby days.
    out["rh_tokyo_pct"] = out["rh_tokyo_pct"].fillna(out.groupby("hour")["rh_tokyo_pct"].transform(
        lambda s: s.interpolate(limit_direction="both")))
    return out


def split(df: pd.DataFrame):
    train = df[df.index < TEST_START]
    test = df[df.index >= TEST_START]
    return train[FEATURES], test[FEATURES], train[TARGET], test[TARGET], train, test


# --------------------------------------------------------------------------------------
# 2. Models
# --------------------------------------------------------------------------------------
class SeasonalBaseline:
    """Naive benchmark: average demand for the same month, hour and day type in the training data."""

    def fit(self, X, y):
        key = [X["month"], X["hour"], X["is_workday"]]
        self.table_ = y.groupby(key).mean()
        self.fallback_ = y.mean()
        return self

    def predict(self, X):
        idx = pd.MultiIndex.from_arrays([X["month"], X["hour"], X["is_workday"]])
        return self.table_.reindex(idx).fillna(self.fallback_).to_numpy()


def build_models() -> dict:
    linear = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("bin", "passthrough", BINARY)])),
        ("model", LinearRegression())])
    forest = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.5, n_jobs=-1,
                                   random_state=RANDOM_STATE)
    boosting = HistGradientBoostingRegressor(max_iter=800, learning_rate=0.05, max_leaf_nodes=48,
                                             min_samples_leaf=20, l2_regularization=1.0,
                                             random_state=RANDOM_STATE)
    return {"Baseline (seasonal average)": SeasonalBaseline(), "Linear Regression": linear,
            "Random Forest": forest, "Gradient Boosting": boosting}


def metrics(y_true, y_pred) -> dict:
    return {"MAE_GW": mean_absolute_error(y_true, y_pred),
            "RMSE_GW": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAPE_pct": mean_absolute_percentage_error(y_true, y_pred) * 100,
            "R2": r2_score(y_true, y_pred)}


# --------------------------------------------------------------------------------------
# 3. Charts
# --------------------------------------------------------------------------------------
def _style(ax, grid="y"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=8.5)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _title(ax, title, sub):
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=24)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def chart_comparison(comp: pd.DataFrame):
    c = comp.sort_values("MAPE_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    _style(ax, grid="x")
    colors = [ORANGE if n == c.index[0] else BLUE for n in c.index]
    ax.barh(c.index, c["MAPE_pct"], color=colors, height=0.6)
    for i, (n, v) in enumerate(c["MAPE_pct"].items()):
        ax.text(v + 0.08, i, f"{v:.2f} %", va="center", fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute percentage error on the test set (lower is better)", color=INK_2, fontsize=9)
    _title(ax, f"{c.index[0]} is the most accurate model",
           "Test set: Jan – Oct 2026 (hours the models never saw). Orange: best model.")
    _save(fig, "fig1_model_comparison.png")


def chart_weeks(test: pd.DataFrame, pred: pd.Series, name: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), facecolor=SURFACE)
    for ax, (start, label) in zip(axes, [("2026-01-19", "Winter week"), ("2026-07-20", "Summer week")]):
        ax.set_facecolor(SURFACE)
        _style(ax)
        sl = slice(start, (pd.Timestamp(start) + pd.Timedelta(days=7)).strftime("%Y-%m-%d"))
        a, p = test.loc[sl, TARGET], pred.loc[sl]
        ax.plot(a.index, a, color=BLUE, linewidth=2, label="Actual demand")
        ax.plot(p.index, p, color=ORANGE, linewidth=2, linestyle="--", label="Predicted")
        ax.set_ylabel("GW", color=INK_2, fontsize=9)
        lo_, hi_ = min(a.min(), p.min()), max(a.max(), p.max())
        ax.set_ylim(lo_ - 1, hi_ + (hi_ - lo_) * 0.18)
        ax.text(0.005, 0.97, f"{label}: {a.index[0]:%d %b} – {a.index[-1]:%d %b %Y}  ·  "
                f"MAPE {mean_absolute_percentage_error(a, p) * 100:.1f} %",
                transform=ax.transAxes, fontsize=9, color=INK, va="top")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d"))
    axes[0].legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK, ncol=2)
    _title(axes[0], "Predicted vs actual demand, hour by hour",
           f"{name} on two test weeks it never saw during training")
    _save(fig, "fig2_actual_vs_predicted_weeks.png")


def chart_scatter(y_true, y_pred, name, m):
    fig, ax = plt.subplots(figsize=(6.4, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    _style(ax, grid="both")
    ax.scatter(y_true, y_pred, s=6, alpha=0.35, color=BLUE, edgecolors="none")
    lo, hi = float(min(y_true.min(), y_pred.min())) - 1, float(max(y_true.max(), y_pred.max())) + 1
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1.2)
    ax.text(hi - 0.5, hi - 2.2, "perfect prediction", color=INK_2, fontsize=8.5, ha="right", rotation=45,
            rotation_mode="anchor")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual demand (GW)", color=INK_2, fontsize=9)
    ax.set_ylabel("Predicted demand (GW)", color=INK_2, fontsize=9)
    _title(ax, "Predictions follow the actual values closely",
           f"{name}, {len(y_true):,} test hours · R² = {m['R2']:.3f} · MAE = {m['MAE_GW']:.2f} GW")
    _save(fig, "fig3_predicted_vs_actual_scatter.png")


def chart_importance(imp: pd.Series, name: str):
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    _style(ax, grid="x")
    ax.barh([FEATURE_LABELS[f] for f in imp.index], imp.values, color=BLUE, height=0.65)
    for i, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.01, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK_2)
    ax.set_xlabel("Increase in test error (MAE, GW) when the feature is shuffled", color=INK_2, fontsize=9)
    _title(ax, "Hour of day, calendar and temperature drive the predictions",
           f"Permutation importance, {name}, test set")
    _save(fig, "fig4_feature_importance.png")


def chart_error_by_month(test: pd.DataFrame, pred: pd.Series, tepco: pd.Series):
    ours = (pred - test[TARGET]).abs().div(test[TARGET]).groupby(test.index.month).mean() * 100
    tep = (tepco - test[TARGET]).abs().div(test[TARGET]).groupby(test.index.month).mean() * 100
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(len(ours))
    fig, ax = plt.subplots(figsize=(9.5, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    _style(ax)
    ax.bar(x - 0.2, ours.values, width=0.38, color=BLUE, label="This model (weather + calendar only)")
    ax.bar(x + 0.2, tep.values, width=0.38, color=AQUA, label="TEPCO's official forecast (reference)")
    ax.set_xticks(x, [months[m - 1] for m in ours.index])
    ax.set_ylabel("Mean absolute % error", color=INK_2, fontsize=9)
    _title(ax, "Error by month in 2026", "This model vs the forecast published by TEPCO, which also uses "
           "recent demand and operator knowledge")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    _save(fig, "fig5_error_by_month.png")


# --------------------------------------------------------------------------------------
def run(save: bool = True) -> dict:
    df = add_features(load())
    X_train, X_test, y_train, y_test, train, test = split(df)

    rows, preds, fitted = {}, {}, {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        p = pd.Series(model.predict(X_test), index=X_test.index)
        preds[name], fitted[name] = p, model
        rows[name] = metrics(y_test, p)
    tepco = test["tepco_forecast_gw"]
    rows["TEPCO official forecast (reference)"] = metrics(y_test, tepco)
    comp = pd.DataFrame(rows).T.round(3)

    candidates = comp.drop(index="TEPCO official forecast (reference)")
    best = candidates["MAPE_pct"].idxmin()
    best_model, best_pred = fitted[best], preds[best]

    imp = permutation_importance(best_model, X_test, y_test, scoring="neg_mean_absolute_error", n_repeats=5,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False)

    out = {"df": df, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "test": test,
           "comparison": comp, "best_name": best, "best_model": best_model, "best_pred": best_pred,
           "importance": importance, "preds": preds}
    if save:
        comp.to_csv(HERE / "model_comparison.csv")
        pd.DataFrame({"actual_gw": y_test, "predicted_gw": best_pred.round(2),
                      "error_gw": (best_pred - y_test).round(2),
                      "tepco_forecast_gw": tepco, "temp_tokyo_c": test["temp_tokyo_c"]}).to_csv(
            HERE / "test_predictions.csv", index_label="datetime")
        joblib.dump({"model": best_model, "features": FEATURES, "name": best}, HERE / "demand_model.joblib",
                    compress=3)
        (HERE / "model_info.json").write_text(json.dumps({
            "best_model": best, "train_period": f"{X_train.index.min():%Y-%m-%d} to {X_train.index.max():%Y-%m-%d}",
            "test_period": f"{X_test.index.min():%Y-%m-%d} to {X_test.index.max():%Y-%m-%d}",
            "train_rows": len(X_train), "test_rows": len(X_test), "features": FEATURES,
            "test_metrics": comp.loc[best].to_dict()}, indent=2))
        chart_comparison(candidates)
        chart_weeks(test, best_pred, best)
        chart_scatter(y_test, best_pred, best, comp.loc[best])
        chart_importance(importance, best)
        chart_error_by_month(test, best_pred, tepco)
    return out


if __name__ == "__main__":
    r = run()
    print(f"Train: {len(r['X_train']):,} hours | Test: {len(r['X_test']):,} hours")
    print(r["comparison"])
    print("\nBest model:", r["best_name"])
    print(r["importance"].round(3))
