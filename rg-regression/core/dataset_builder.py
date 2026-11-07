from datasetsforecast.m4 import M4, M4Info
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import requests
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from .workers import _seasonal_naive_worker, _holt_damped_worker, _theta_worker, _arima_worker, _ar_worker, _no_baseline_worker, _narmax_worker

class DatasetBuilder:
    def __init__(self,
                 subset="Hourly", 
                 window_size=240, 
                 val_ratio=0.2,
                 baseline_method="ARIMA", # "theta", "seasonal-naive", "ARIMA", "AR" ou "holt-damped"
                 stride=48): 
        
        self.horizon = M4Info[subset].horizon
        self.seasonality = M4Info[subset].seasonality
        self.stride = stride
        self.subset = subset
        self.window_size = window_size
        self.data_dir = os.path.join("data", "M4")
        self.val_ratio = val_ratio
        self.baseline_method = baseline_method

        self.has_been_built = False

    def reconstruct_full_series_predictions(self, y_pred_model_raw):
        y_pred_series_full = {}

        for i in range(len(self.cached_windows)):
            w = self.cached_windows[i]

            series_id = w["series_id"]
            y_pred_windows = y_pred_model_raw[i]

            scaler = self.scalers[series_id]
            y_pred_series_full[series_id] = scaler.inverse_transform(y_pred_windows.reshape(-1, 1)).flatten()

        return y_pred_series_full

    # -------------------------------------------------
    # Build
    # -------------------------------------------------
    def _save_model_results(self, results):
        baseline_dir = os.path.join("results", "Baselines")
        os.makedirs(baseline_dir, exist_ok=True)

        npz_path = os.path.join(
            baseline_dir,
            f"{self.subset}_{self.baseline_method}_results.npz"
        )

        df, *_ = M4.load(directory=self.data_dir, group=self.subset)
        original_order = df["unique_id"].drop_duplicates().tolist()
        results_dict = {r["series_id"]: r["baseline"] for r in results}

        series_ids = []
        y_preds = []

        for sid in original_order:
            if sid not in results_dict:
                raise ValueError(f"Series ID {sid} not found in the forecasts")
            series_ids.append(sid)
            y_preds.append(np.array(results_dict[sid], dtype=np.float32))

        np.savez_compressed(
            npz_path,
            series_id=np.array(series_ids, dtype=object),
            y_pred=np.array(y_preds, dtype=object)
        )

        print(f"✅ Model results saved to {npz_path}")

    def _generate_model_baseline(self, series_list):
        
        args_list = [
            (
                series_dict["series_id"],
                series_dict["full_series"],
                series_dict["train_len"],
                self.seasonality,
                self.window_size,
                self.horizon,
                self.stride
            )
            for series_dict in series_list
        ]

        if len(args_list) == 0:
            return []

        available_cpus = max(1, cpu_count() - 1)
        n_workers = min(available_cpus, len(args_list))
        results = []

        print(f"\nStarting model rolling forecast with {n_workers} workers")
        print(f"Total series: {len(args_list)}\n")

        if self.baseline_method == "theta":
            worker_to_use = _theta_worker
        elif self.baseline_method == "seasonal-naive":
            worker_to_use = _seasonal_naive_worker
        elif self.baseline_method == "no":
            worker_to_use = _no_baseline_worker
        elif self.baseline_method == "NARMAX":
            worker_to_use = _narmax_worker
        elif self.baseline_method == "holt-damped":
            worker_to_use = _holt_damped_worker
        elif self.baseline_method == "ARIMA":
            worker_to_use = _arima_worker
        elif self.baseline_method == "AR":
            worker_to_use = _ar_worker
        elif self.baseline_method == "mixed":
            raise ValueError(f"Unsupported generation for mixed method")
        else:
            raise ValueError(f"Unsupported baseline method: {self.baseline_method}")
    
        with Pool(n_workers) as pool:
            for series_id, baseline_series in tqdm(
                pool.imap_unordered(worker_to_use, args_list),
                total=len(args_list),
                desc="Model Progress",
                smoothing=0.1
            ):
                results.append({
                    "series_id": series_id,
                    "baseline": baseline_series
                })

        print("\n✅ Model rolling forecast finished.\n")

        self._save_model_results(results)

        return results
    
    # -------------------------------------------------
    # Window generator
    # -------------------------------------------------
    def _create_windows(self, series, baseline_preds, window_size, horizon, stride):

        valid_len = np.count_nonzero(~np.isnan(series))
        series = series[:valid_len]
        baseline_preds = baseline_preds[:valid_len]

        X, y, b = [], [], []

        max_start = len(series) - window_size - horizon + 1

        for i in range(0, max_start, stride):
            X.append(series[i : i + window_size].reshape(window_size, 1))
            y.append(series[i + window_size : i + window_size + horizon])
            b.append(baseline_preds[i + window_size : i + window_size + horizon])

        return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(b, dtype=np.float32),
        )
    
    def build(self, limit=None):
        df, *_ = M4.load(directory=self.data_dir, group=self.subset)

        seen = set()
        unique_ids = []
        for uid in df["unique_id"].tolist():
            if uid not in seen:
                seen.add(uid)
                unique_ids.append(uid)

        df_groups = df.groupby("unique_id", sort=False)
        series_groups = [(uid, df_groups.get_group(uid)) for uid in unique_ids]

        if limit is not None:
            series_groups = series_groups[:limit]

        baseline_dir = os.path.join("results", "Baselines")
        os.makedirs(baseline_dir, exist_ok=True)

        existing_series = {}

        model_npz_path = os.path.join(
            baseline_dir,
            f"{self.subset}_{self.baseline_method}_results.npz"
        )

        if os.path.exists(model_npz_path):
            data = np.load(model_npz_path, allow_pickle=True)
            existing_series = dict(
                zip(data["series_id"].tolist(), data["y_pred"].tolist())
            )

        self.series_data = []
        series_to_compute = []

        for series_id, g in series_groups:
            full_series = g["y"].values.astype(np.float32)
            min_required_len = self.window_size + self.horizon

            if len(full_series) < min_required_len:
                pad_size = min_required_len - len(full_series)

                if len(full_series) >= self.seasonality:
                    first_season = full_series[:self.seasonality]

                    reps = int(np.ceil(pad_size / self.seasonality))
                    pad_values = np.tile(first_season, reps)[:pad_size]
                else:
                    raise RuntimeError("The series is less than one season.")

                full_series = np.concatenate([pad_values, full_series])

            train_series = full_series[:-self.horizon]
            test_series  = full_series[-self.horizon:]

            train_len = len(train_series)
            baseline_full = None

            if series_id in existing_series:
                candidate = existing_series[series_id]
                if len(candidate) == len(full_series):
                    baseline_full = np.array(candidate, dtype=np.float32)

            if baseline_full is not None:
                self.series_data.append({
                    "series_id": series_id,
                    "full_series": full_series,
                    "baseline": baseline_full,
                    "train_len": train_len
                })
            else:
                series_to_compute.append({
                    "series_id": series_id,
                    "full_series": full_series,
                    "train_len": train_len
                })

        if series_to_compute:
            model_results = self._generate_model_baseline(series_to_compute)
            series_map = {s["series_id"]: s for s in series_to_compute}
            for res in model_results:
                sid = res["series_id"]
                s_info = series_map[sid]
                self.series_data.append({
                    "series_id": sid,
                    "full_series": s_info["full_series"],
                    "baseline": res["baseline"].astype(np.float32),
                    "train_len": s_info["train_len"]
                })

        if len(self.series_data) == 0:
            raise RuntimeError("No valid series processed.")

        df, *_ = M4.load(directory=self.data_dir, group=self.subset)

        lib_ids = df["unique_id"].unique()

        mapper = {s["series_id"]: s for s in self.series_data}

        self.series_data = [mapper[sid] for sid in lib_ids]

        self.cached_windows = []
        self.scalers = {}

        for s in tqdm(self.series_data, desc="Precomputing windows"):
            series_id = s["series_id"]
            full_series = s["full_series"]
            baseline_full = s["baseline"]
            train_len = s["train_len"]
            train_series = full_series[:train_len]
            test_series  = full_series[train_len:]
            baseline_train = baseline_full[:train_len]
            baseline_test  = baseline_full[train_len:]

            # -------------------------
            # Scaling
            # -------------------------
            scaler = MinMaxScaler()
            scaler.fit(train_series.reshape(-1, 1))
            self.scalers[series_id] = scaler
            train_scaled = scaler.transform(train_series.reshape(-1, 1)).flatten()
            test_scaled  = scaler.transform(test_series.reshape(-1, 1)).flatten()
            baseline_train_scaled = scaler.transform(baseline_train.reshape(-1, 1)).flatten()
            baseline_test_scaled  = scaler.transform(baseline_test.reshape(-1, 1)).flatten()

            # -------------------------
            # Windows
            # -------------------------
            X_all, y_all, b_all = self._create_windows(
                train_scaled,
                baseline_train_scaled,
                self.window_size,
                self.horizon,
                self.stride
            )

            if len(X_all) >= 2:
                n_windows = len(X_all)
                n_val = max(1, int(n_windows * self.val_ratio))
                n_val = min(n_val, n_windows - 1)
                X_train = X_all[:-n_val]
                y_train = y_all[:-n_val]
                b_train = b_all[:-n_val]
                X_val = X_all[-n_val:]
                y_val = y_all[-n_val:]
                b_val = b_all[-n_val:]
            else:
                X_train = y_train = b_train = None
                X_val = y_val = b_val = None

            X_test = train_scaled[-self.window_size:].reshape(1, -1)
            y_test = test_scaled.reshape(1, -1)
            b_test = baseline_test_scaled.reshape(1, -1)

            self.cached_windows.append({
                "series_id": series_id,
                "full_series": full_series,
                "train_len": train_len,
                "X_train": X_train,
                "y_train": y_train,
                "b_train": b_train,
                "X_val": X_val,
                "y_val": y_val,
                "b_val": b_val,
                "X_test": X_test,
                "y_test": y_test,
                "b_test": b_test,
            })

        self.has_been_built = True
        return self

    def build_loaders(self, batch_size=512):
        if not self.has_been_built:
            raise ValueError("Call build() first")

        X_train_all, y_train_all, b_train_all = [], [], []
        X_val_all, y_val_all, b_val_all = [], [], []
        self.X_test_all, self.y_test_all, self.b_test_all = [], [], []

        for w in self.cached_windows:
            if w["X_train"] is not None and w["X_train"].size > 0:
                X_train_all.append(torch.from_numpy(w["X_train"]))
                y_train_all.append(torch.from_numpy(w["y_train"]))
                b_train_all.append(torch.from_numpy(w["b_train"]))
            if w["X_val"] is not None and w["X_val"].size > 0:
                X_val_all.append(torch.from_numpy(w["X_val"]))
                y_val_all.append(torch.from_numpy(w["y_val"]))
                b_val_all.append(torch.from_numpy(w["b_val"]))

            self.X_test_all.append(torch.from_numpy(w["X_test"]))
            self.y_test_all.append(torch.from_numpy(w["y_test"]))
            self.b_test_all.append(torch.from_numpy(w["b_test"]))

        def stack(lst):
            if not lst:
                return torch.empty(0)

            x = torch.cat(lst, dim=0)

            if x.dim() == 2:          # (batch, window)
                x = x.unsqueeze(-1)   # (batch, window, 1)

            return x

        train_ds = TensorDataset(stack(X_train_all), stack(y_train_all), stack(b_train_all))
        val_ds   = TensorDataset(stack(X_val_all), stack(y_val_all), stack(b_val_all))
        test_ds  = TensorDataset(stack(self.X_test_all), stack(self.y_test_all), stack(self.b_test_all))

        return DataLoader(
            train_ds, batch_size=batch_size, shuffle=False,
            num_workers=min(os.cpu_count() - 1, 4), pin_memory=True, persistent_workers=True
        ), \
        DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=min(os.cpu_count() - 1, 4), pin_memory=True, persistent_workers=True
        ), \
        DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=min(os.cpu_count() - 1, 4), pin_memory=True, persistent_workers=True
        )