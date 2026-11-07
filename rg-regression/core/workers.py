from statsforecast import StatsForecast
from statsforecast.models import Theta, SeasonalNaive, Holt, ARIMA
import pandas as pd
from sysidentpy.model_structure_selection import AOLS
from sysidentpy.basis_function import Polynomial
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np

def _seasonal_naive_worker(args):
    series_id, full_series, train_len, period, window_size, horizon, stride = args

    full = np.asarray(full_series, dtype=np.float64)
    baseline = np.zeros_like(full)

    if train_len < 2:
        raise ValueError(f"{series_id}: train_len < 2")

    # -------------------------------
    # Train and test separation
    # -------------------------------
    train = full[:train_len]
    h_out = len(full) - train_len  # out-of-sample

    # -------------------------------
    # Rolling in-sample forecast (h=1)
    # -------------------------------
    in_sample = train.copy()
    for t in range(window_size, train_len):
        train_df = pd.DataFrame({
            "unique_id": [series_id] * t,
            "ds": np.arange(t),
            "y": in_sample[:t]
        })
        sf = StatsForecast(
            models=[SeasonalNaive(season_length=period)],
            freq=1
        )
        sf.fit(train_df)
        forecast_df = sf.forecast(df=train_df, h=1)
        in_sample[t] = forecast_df["SeasonalNaive"].values[0]

    # -------------------------------
    # Out-of-sample forecast
    # -------------------------------
    if h_out > 0:
        train_df = pd.DataFrame({
            "unique_id": [series_id] * train_len,
            "ds": np.arange(train_len),
            "y": train
        })
        sf = StatsForecast(
            models=[SeasonalNaive(season_length=period)],
            freq=1
        )
        sf.fit(train_df)
        future_df = sf.forecast(df=train_df, h=h_out)
        out_of_sample = future_df["SeasonalNaive"].values
    else:
        out_of_sample = np.array([], dtype=np.float64)

    # -------------------------------
    # Create baseline
    # -------------------------------
    baseline[:train_len] = in_sample
    baseline[train_len:] = out_of_sample

    return series_id, baseline

def _holt_damped_worker(args):
    series_id, full_series, train_len, period, window_size, horizon, stride = args

    full = np.asarray(full_series, dtype=np.float64)
    baseline = np.zeros_like(full)

    if train_len < 2:
        raise ValueError(f"{series_id}: train_len < 2")

    # -------------------------------
    # Train and test separation
    # -------------------------------
    train = full[:train_len]
    h_out = len(full) - train_len  # out-of-sample

    # -------------------------------
    # Rolling in-sample forecast (h=1)
    # -------------------------------
    in_sample = train.copy()
    for t in range(window_size, train_len):
        train_df = pd.DataFrame({
            "unique_id": [series_id] * t,
            "ds": np.arange(t),
            "y": in_sample[:t]
        })

        sf = StatsForecast(
            models=[Holt(season_length=period, error_type="A", alias="Add")],
            freq=1
        )

        sf.fit(train_df)
        forecast_df = sf.forecast(df=train_df, h=1)

        in_sample[t] = forecast_df["Add"].values[0]

    # -------------------------------
    # Out-of-sample forecast
    # -------------------------------
    if h_out > 0:
        train_df = pd.DataFrame({
            "unique_id": [series_id] * train_len,
            "ds": np.arange(train_len),
            "y": train
        })

        sf = StatsForecast(
            models=[Holt(season_length=period, error_type="A", alias="Add")],
            freq=1
        )

        sf.fit(train_df)
        future_df = sf.forecast(df=train_df, h=h_out)

        out_of_sample = future_df["Add"].values
    else:
        out_of_sample = np.array([], dtype=np.float64)

    # -------------------------------
    # Build baseline
    # -------------------------------
    baseline[:train_len] = in_sample
    baseline[train_len:] = out_of_sample

    return series_id, baseline

def _no_baseline_worker(args):
    series_id, full_series, *_ = args
    return series_id, np.zeros_like(full_series)

def _theta_worker(args):
    series_id, full_series, train_len, period, window_size, horizon, stride = args

    full = np.asarray(full_series, dtype=np.float64)
    baseline = np.zeros_like(full)

    if train_len < 2:
        raise ValueError(f"{series_id}: train_len < 2")

    # -------------------------------
    # Train and test separation
    # -------------------------------
    train = full[:train_len]
    h_out = len(full) - train_len  # out-of-sample

    # -------------------------------
    # Rolling in-sample forecast with stride
    # -------------------------------
    in_sample = train.copy()
    for t in range(window_size, train_len, stride):
        y_hist = in_sample[:t]

        # Adjust Theta with season_length = period
        train_df = pd.DataFrame({
            "unique_id": [series_id] * t,
            "ds": np.arange(t),
            "y": y_hist
        })

        sf = StatsForecast(models=[Theta(season_length=period)], freq=1)
        sf.fit(train_df)

        h = min(period, train_len - t)
        if h <= 0:
            continue

        forecast_df = sf.forecast(df=train_df, h=h)
        in_sample[t:t + h] = forecast_df["theta"].values[:h]

    # -------------------------------
    # Out-of-sample forecast
    # -------------------------------
    if h_out > 0:
        train_df = pd.DataFrame({
            "unique_id": [series_id] * train_len,
            "ds": np.arange(train_len),
            "y": train
        })
        sf = StatsForecast(models=[Theta(season_length=period)], freq=1)
        sf.fit(train_df)

        forecast_df = sf.forecast(df=train_df, h=h_out)
        out_of_sample = forecast_df["theta"].values
    else:
        out_of_sample = np.array([], dtype=np.float64)

    # -------------------------------
    # Build baseline
    # -------------------------------
    baseline[:train_len] = in_sample
    baseline[train_len:] = out_of_sample

    return series_id, baseline

def _arima_worker(args):

    series_id, full_series, train_len, period, window_size, horizon, stride = args

    full = np.asarray(full_series, dtype=np.float64)
    baseline = np.zeros_like(full)

    if train_len < 2:
        raise ValueError(f"{series_id}: train_len < 2")

    train = full[:train_len]
    h_out = len(full) - train_len

    # --------------------------------
    # Model
    # --------------------------------
    model = ARIMA(
        order=(1,0,0),
        seasonal_order=(1,0,0),
        season_length=period
    )

    sf = StatsForecast(models=[model], freq=1)

    # --------------------------------
    # In-sample forecast
    # --------------------------------
    baseline[:window_size] = train[:window_size]

    for t in range(window_size, train_len, stride):

        y = train[:t]

        train_df = pd.DataFrame({
            "unique_id": np.repeat(series_id, t),
            "ds": np.arange(t),
            "y": y
        })

        sf.fit(train_df)

        h = min(period, train_len - t)

        forecast_df = sf.forecast(
            df=train_df,
            h=h
        )

        preds = forecast_df["ARIMA"].values

        baseline[t:t+h] = preds

    # --------------------------------
    # Out-of-sample forecast
    # --------------------------------
    if h_out > 0:

        train_df = pd.DataFrame({
            "unique_id": np.repeat(series_id, train_len),
            "ds": np.arange(train_len),
            "y": train
        })

        sf.fit(train_df)

        future_df = sf.forecast(
            df=train_df,
            h=h_out
        )

        baseline[train_len:] = future_df["ARIMA"].values

    return series_id, baseline

def _narmax_worker(args):

    import numpy as np
    from datetime import datetime
    import traceback

    series_id, full_series, train_len, period, window_size, horizon, stride = args

    full = np.asarray(full_series, dtype=np.float64)
    baseline = np.zeros_like(full)

    if train_len < 2:
        raise ValueError(f"{series_id}: train_len < 2")

    train = full[:train_len]
    h_out = len(full) - train_len

    basis_function = Polynomial(degree=1)

def _narmax_worker(args):

    import numpy as np
    from datetime import datetime
    import traceback

    series_id, full_series, train_len, period, window_size, horizon, stride = args

    full = np.asarray(full_series, dtype=np.float64)
    baseline = np.zeros_like(full)

    if train_len < 2:
        raise ValueError(f"{series_id}: train_len < 2")

    train = full[:train_len]
    h_out = len(full) - train_len

    basis_function = Polynomial(degree=1)

    # --------------------------------
    # seasonal naive fallback
    # --------------------------------
    def seasonal_naive(history, steps):

        hist = np.asarray(history).ravel()
        preds = np.zeros(steps)

        if period is None or period <= 1 or len(hist) < period:
            preds[:] = hist[-1] if len(hist) > 0 else 0.0
            return preds

        for i in range(steps):
            idx = len(hist) - period + i
            if idx < len(hist):
                preds[i] = hist[idx]
            else:
                preds[i] = preds[i - period] if i >= period else hist[-period]

        return preds

    def get_ylag(period):
        return 1

    # --------------------------------
    # filter function
    # --------------------------------
    def should_fallback(y_flat, ylag):

        n = len(y_flat)

        if period <= 1 and n < 40:
            return True

        effective_n = n - (ylag + 1)

        if effective_n < max(ylag * 10, 40):
            return True

        if not np.isfinite(y_flat).all():
            return True

        std = np.std(y_flat)

        if std < 1e-8:
            return True

        diff_std = np.std(np.diff(y_flat))

        if diff_std < 1e-8:
            return True

        if np.unique(y_flat).size < max(3, int(0.02 * n)):
            return True

        if np.max(np.abs(y_flat)) > 1e6:
            return True

        if n <= 2 * ylag + 1:
            return True

        return False

    # --------------------------------
    # rolling forecast
    # --------------------------------
    baseline[:window_size] = train[:window_size]

    for t in range(window_size, train_len, stride):

        h = min(stride, train_len - t)

        y = train[:t].reshape(-1, 1)

        try:

            y_flat = y.ravel()
            n = len(y_flat)

            ylag = get_ylag(period)

            if should_fallback(y_flat, ylag):

                preds = seasonal_naive(y, h)

            else:

                k = 1

                model = AOLS(
                    ylag=ylag,
                    basis_function=basis_function,
                    model_type="NAR",
                    k=k,
                )

                model.fit(y=y)

                y_val = y[-model.max_lag:].reshape(-1, 1)

                y_hat = model.predict(
                    y=y_val,
                    forecast_horizon=h
                )

                preds = y_hat[model.max_lag:].ravel()

                # validation post-predict
                if len(preds) < h:
                    raise RuntimeError("Prediction shorter than horizon")

                if not np.isfinite(preds).all():
                    preds = seasonal_naive(y, h)

                if np.max(np.abs(preds)) > 1e6:
                    preds = seasonal_naive(y, h)

        except Exception as e:

            preds = seasonal_naive(y, h)

            log_msg = (
                "\n================ NARMAX ERROR ================\n"
                f"timestamp : {datetime.utcnow().isoformat()} UTC\n"
                f"series_id : {series_id}\n"
                f"timestep  : {t}\n"
                f"train_len : {train_len}\n"
                f"period    : {period}\n"
                f"stride    : {stride}\n"
                f"horizon   : {h}\n\n"
                f"Exception:\n{str(e)}\n\n"
                f"Traceback:\n{traceback.format_exc()}\n"
            )

            with open("narmax_error_log.txt", "a") as f:
                f.write(log_msg)

        baseline[t:t+h] = preds[:h]

    # --------------------------------
    # Out-of-sample forecast
    # --------------------------------
    if h_out > 0:

        y = train.reshape(-1, 1)

        try:

            y_flat = y.ravel()
            n = len(y_flat)

            ylag = get_ylag(period)

            if should_fallback(y_flat, ylag):

                preds = seasonal_naive(y, h_out)

            else:

                k = 1

                model = AOLS(
                    ylag=ylag,
                    basis_function=basis_function,
                    model_type="NAR",
                    k=k,
                )

                model.fit(y=y)

                y_val = y[-model.max_lag:].reshape(-1, 1)

                y_hat = model.predict(
                    y=y_val,
                    forecast_horizon=h_out
                )

                preds = y_hat[model.max_lag:].ravel()

                # validation post-predict
                if not np.isfinite(preds).all():
                    preds = seasonal_naive(y, h_out)

                if np.max(np.abs(preds)) > 1e6:
                    preds = seasonal_naive(y, h_out)

        except Exception:

            preds = seasonal_naive(y, h_out)

        baseline[train_len:] = preds[:h_out]

    return series_id, baseline

def _fit_nar(y, lag, ridge=1e-6):

    n = len(y)

    if n <= lag:
        lag = max(1, n - 1)

    # Regression matrix
    X = sliding_window_view(y, lag + 1)

    Y = X[:, -1]
    X = X[:, :-1][:, ::-1]

    # ridge
    XtX = X.T @ X
    XtX.flat[:: XtX.shape[0] + 1] += ridge

    beta = np.linalg.solve(XtX, X.T @ Y)

    return beta

def _forecast_nar(history, beta, h):

    lag = len(beta)

    hist = np.empty(len(history) + h, dtype=np.float64)
    hist[:len(history)] = history

    for i in range(h):

        hist[len(history) + i] = (
            beta @ hist[len(history) + i - lag : len(history) + i][::-1]
        )

    return hist[-h:]

# -----------------------------
# Drift forecast
# -----------------------------
def drift_forecast(history, h):
    n = len(history)
    slope = (history[-1] - history[0]) / max(n - 1, 1)
    return history[-1] + slope * np.arange(1, h + 1)

# -----------------------------
# Seasonal naive forecast
# -----------------------------
def seasonal_naive_forecast(history, season, h):
    if season < 1 or len(history) < season:
        return np.repeat(history[-1], h)
    return np.array([history[-season + (i % season)] for i in range(h)])

# -----------------------------
# HAR-like forecast with seasonal horizons
# -----------------------------
def har_forecast_seasonal(history, season, h):
    n = len(history)
    if n == 0:
        return np.zeros(h)

    horizons = [1]
    if season > 1:
        extra = [season // 2, season, 2 * season, 4 * season]
        horizons += sorted(set([w for w in extra if w >= 1]))

    features = np.array([np.mean(history[-w:]) for w in horizons if n >= w])
    if len(features) == 0:
        return np.repeat(history[-1], h)

    coefs = np.ones(len(features)) / len(features)
    preds = np.empty(h, dtype=np.float64)

    for i in range(h):
        pred = np.dot(coefs, features)
        preds[i] = pred
        features[:-1] = features[1:]
        features[-1] = pred

    return preds

# -----------------------------
# Ensemble HAR + Seasonal Naive + Drift worker
# -----------------------------
def _ar_worker(args):
    series_id, full_series, train_len, season, window_size, horizon, stride = args
    full = np.asarray(full_series, dtype=np.float64)
    baseline = np.zeros_like(full)

    train = full[:train_len]
    h_out = len(full) - train_len

    # Initialize initial windows
    baseline[:window_size] = train[:window_size]

    # Rolling forecast in-sample
    for t in range(window_size, train_len, stride):
        y_hist = train[:t]
        h = min(stride, train_len - t)

        preds_har = har_forecast_seasonal(y_hist, season, h)
        preds_seasonal = seasonal_naive_forecast(y_hist, season, h)
        preds_drift = drift_forecast(y_hist, h)

        # Weighted average
        baseline[t:t+h] = 0.5 * preds_har + 0.3 * preds_seasonal + 0.2 * preds_drift

    # Out-of-sample forecast
    if h_out > 0:
        y_hist = train
        preds_har = har_forecast_seasonal(y_hist, season, h_out)
        preds_seasonal = seasonal_naive_forecast(y_hist, season, h_out)
        preds_drift = drift_forecast(y_hist, h_out)
        baseline[train_len:] = 0.5 * preds_har + 0.3 * preds_seasonal + 0.2 * preds_drift

    return series_id, baseline