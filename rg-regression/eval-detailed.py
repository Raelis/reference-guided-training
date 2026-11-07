import pandas as pd
from core.dataset_builder import DatasetBuilder
import torch
import os
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score
)
import numpy as np
import optuna
from datasetsforecast.m4 import M4Evaluation, M4, M4Info
from scipy.stats import spearmanr, linregress
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import re
import shutil
import subprocess


def save_pdf(path):
    # Export a vector PDF with embedded TrueType fonts (not Type 3).

    # Applies the font settings at save time, even if a plotting theme has
    # changed matplotlib.rcParams earlier in the run. If Poppler's pdffonts
    # is available, verify the saved PDF before reporting success.
    
    with plt.rc_context({
        "pdf.fonttype": 42,           # embed TrueType, rather than Type 3
        "ps.fonttype": 42,
        "pdf.use14corefonts": False, # avoid unembedded core fonts
    }):
        plt.savefig(path, bbox_inches="tight")

    pdffonts = shutil.which("pdffonts")
    if pdffonts is None:
        print(f"[PDF] Saved {path} with pdf.fonttype=42. "
              "Font verification skipped (pdffonts not installed).")
        return

    result = subprocess.run(
        [pdffonts, str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    if re.search(r"\bType\s*3\b", result.stdout, re.IGNORECASE):
        raise RuntimeError(f"{path} contains Type 3 fonts; export failed verification.\n"
                           f"{result.stdout}")

    # The last five whitespace-delimited fields are: emb sub uni object ID.
    font_rows = [line.split() for line in result.stdout.splitlines()[2:]
                 if line.strip()]
    unembedded = [row for row in font_rows if len(row) >= 5
                  and row[-5].lower() != "yes"]
    if unembedded:
        raise RuntimeError(f"{path} contains non-embedded fonts:\n{result.stdout}")

    print(f"[PDF] OK: {path} - no Type 3 fonts; fonts embedded.")


_NAIVE2_METRICS_CACHE = {}


def get_naive2_metrics(subset):
    # Benchmark sMAPE and MASE used as denominators by M4Evaluation.
    if subset not in _NAIVE2_METRICS_CACHE:
        forecasts = M4Evaluation.load_benchmark("data/M4", subset)
        evaluation = M4Evaluation.evaluate(
            directory="data/M4", group=subset, y_hat=forecasts
        )
        _NAIVE2_METRICS_CACHE[subset] = (
            float(evaluation.loc[subset, "SMAPE"]),
            float(evaluation.loc[subset, "MASE"]),
        )
    return _NAIVE2_METRICS_CACHE[subset]


def pooled_owa_runs(sub_df, method, weights):
    # Official M4-style pooled OWA, computed per evaluation run.

    # Aggregate subset sMAPE and MASE using series counts before dividing
    # by their respective pooled Naive2 benchmarks. A weighted average of
    # frequency-level OWA scores is NOT equivalent to pooled OWA.

    smape_runs = np.stack(sub_df[f"{method}_smape_runs"].to_numpy())
    mase_runs = np.stack(sub_df[f"{method}_mase_runs"].to_numpy())

    pooled_smape = np.average(smape_runs, axis=0, weights=weights)
    pooled_mase = np.average(mase_runs, axis=0, weights=weights)
    naive2_smape = np.average(sub_df["naive2_smape"], weights=weights)
    naive2_mase = np.average(sub_df["naive2_mase"], weights=weights)

    return 0.5 * (pooled_smape / naive2_smape + pooled_mase / naive2_mase)


def load(model_class, path, device, window_size, horizon):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_params = checkpoint["model_params"]
    model_params, other_params = model_class.prepare_model_params(model_params)
    model_params["num_features"] = window_size
    model_params["output_dim"] = horizon
    model = model_class(**model_params).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model_class.model_postprocess(model)
    return model, checkpoint

def _get_best_hyperparams(experiment_path_used, experiment_dir):
    db_path = f"{experiment_path_used}/{experiment_dir}/optuna_study.db"
    if not os.path.exists(db_path):
        print(f"[WARNING] Optuna DB not found on {db_path}.")
        return None

    try:
        storage = f"sqlite:///{experiment_path_used}/{experiment_dir}/optuna_study.db"
        study = optuna.load_study(
            study_name="experiment_with_indicators",
            storage=storage
        )
        return study.best_trial.params
    except Exception as e:
        print(f"[ERROR] Failed to load Optuna study: {e}")
        return None
    
def build_m4_forecast_matrix(dataset_builder, y_pred_series_full):
    forecasts = []
    
    for s in dataset_builder.cached_windows:
        series_id = s["series_id"]
        pred = y_pred_series_full[series_id]
        forecasts.append(pred)

    return np.array(forecasts)

def _get_model_metrics(dataset_builder, test_loader, model_class, experiment_dir, model_name, subset, window_size, horizon, experiment_path):
    model, checkpoint = load(model_class, f"{experiment_path}/{experiment_dir}/{model_name}.pt", "cuda", window_size, horizon)
    model.eval()
    y_pred_list, y_true_list = [], []
    with torch.no_grad():
        for X_batch, y_batch, b_batch in test_loader:
            X_batch = X_batch.to("cuda")
            outputs = model(X_batch).cpu()
            y_pred_list.append(outputs.numpy())
            y_true_list.append(y_batch.numpy())
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)

    y_pred_adjusted = dataset_builder.reconstruct_full_series_predictions(y_pred)
    results_matrix = build_m4_forecast_matrix(dataset_builder, y_pred_adjusted)

    evaluation = M4Evaluation.evaluate(
        directory="data/M4",   # folder where M4 dataset is located
        group=subset,
        y_hat=results_matrix
    )

    results = {}
    results["smape"] = evaluation.loc[subset, "SMAPE"]
    results["mase"] = evaluation.loc[subset, "MASE"]
    results["owa"] = evaluation.loc[subset, "OWA"]
    results["y_true"] = y_true
    results["y_pred"] = y_pred
    
    results["train_metrics"] = checkpoint["metrics"]
    return results


def get_model_metrics(model_name = "best-model_r1", experiment_dir = "CNN_n0.40_kd-ResNet20-n0.40_not-loaded_kd"):
    experiment_paths = ["results"]#,"../src-classification/results"]
    results = {}

    experiment_path_used = None

    for experiment_path in experiment_paths:
        if os.path.exists(f"{experiment_path}/{experiment_dir}/experiment-results.csv"):
            experiment_path_used = experiment_path
            break

    if experiment_path_used is None:
        raise FileNotFoundError(f"{experiment_path}/{experiment_dir}/experiment-results.csv not found")

    experiment_info = pd.read_csv(f"{experiment_path_used}/{experiment_dir}/experiment-results.csv")

    row = experiment_info.iloc[0]

    results["name"] = row["name"]

    if "Daily" in row["name"]:
        results["subset"] = "Daily"
    elif "Hourly" in row["name"]:
        results["subset"] = "Hourly"
    elif "Weekly" in row["name"]:
        results["subset"] = "Weekly"
    elif "Monthly" in row["name"]:
        results["subset"] = "Monthly"
    elif "Quarterly" in row["name"]:
        results["subset"] = "Quarterly"
    elif "Yearly" in row["name"]:
        results["subset"] = "Yearly"

    results["model"] = row["model"]
    results["loss"] = row["loss"]
    results["loss_baseline"] = row["loss_baseline"]

    if (results["name"].endswith("_baseline")) and (results["loss_baseline"] == "no"):
        results["type"] = "baseline"
    elif results["loss_baseline"] in ["seasonal-naive","NARMAX"]:
        results["type"] = "guided"

    results["best_hyperparams"] = _get_best_hyperparams(experiment_path_used, experiment_dir)

    print(results)

    if results["model"] == "TCN":
        from models.tcn_model import MyModel
    elif results["model"] == "MLP":
        from models.mlp_model import MyModel
    elif results["model"] == "GRU":
        from models.gru_model import MyModel
    else:
        raise ValueError(f"Unknown model prefix in experiment_dir: {experiment_dir}")
    
    for experiment_path in experiment_paths:
        if os.path.exists(f"{experiment_path}/{experiment_dir}/{model_name}.pt"):
            break

    strides = {"Hourly": 48, "Daily": 198, "Weekly": 13, "Monthly": 112, "Quarterly": 32, "Yearly": 6}
    results["stride"] = strides[results["subset"]]
    window_sizes = {"Hourly": 192, "Daily": 56, "Weekly": 52, "Monthly": 72, "Quarterly": 32, "Yearly": 18}
    results["window_size"] = window_sizes[results["subset"]]

    results["horizon"] = M4Info[results["subset"]].horizon
    results["seasonality"] = M4Info[results["subset"]].seasonality

    dataset_builder = DatasetBuilder(subset=results["subset"], 
                                     baseline_method=results["loss_baseline"], 
                                     window_size=results["window_size"], 
                                     stride=results["stride"])
    dataset_builder.build()
    train_loader, val_loader, test_loader = dataset_builder.build_loaders(batch_size=128)

    if "baseline" != model_name.split("_")[-1]:
        results["y_baseline"] = dataset_builder.b_test_all
        y_baseline_adjusted = dataset_builder.reconstruct_full_series_predictions(results["y_baseline"])
        results_matrix_baseline = build_m4_forecast_matrix(dataset_builder, y_baseline_adjusted)
        evaluation_baseline = M4Evaluation.evaluate(
            directory="data/M4",   # folder where M4 dataset is located
            group=results["subset"],
            y_hat=results_matrix_baseline
        )
        results["baseline_smape"] = evaluation_baseline.loc[results["subset"], "SMAPE"]
        results["baseline_mase"] = evaluation_baseline.loc[results["subset"], "MASE"]
        results["baseline_owa"] = evaluation_baseline.loc[results["subset"], "OWA"]

    results_list = []
    for i in range(min(len(experiment_info),5)):
        results_list.append(_get_model_metrics(dataset_builder, test_loader, MyModel, experiment_dir, f"best-model_r{i+1}", results["subset"], results["window_size"], results["horizon"], experiment_path))

    results["experiments"] = results_list
    return results

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    rows = []

    # storage = {}

    for model in ["MLP", "GRU", "TCN"]:
        for subset in ["Yearly", "Quarterly", "Monthly", "Weekly", "Hourly", "Daily"]:

            folder_rg = f"{model}_{subset}_baseline-mse-NARMAX"
            folder_erm = f"{model}_{subset}_mse"

            # data_rg = storage[f"{model}_{subset}_rg"] 
            # data_erm = storage[f"{model}_{subset}_erm"] 
            data_rg = get_model_metrics(experiment_dir=folder_rg)
            data_erm = get_model_metrics(experiment_dir=folder_erm)

            # storage[f"{model}_{subset}_rg"] = data_rg
            # storage[f"{model}_{subset}_erm"] = data_erm

            # OWA averages
            owa_rg = [exp["owa"] for exp in data_rg["experiments"]]
            owa_erm = [exp["owa"] for exp in data_erm["experiments"]]

            avg_owa_rg = np.mean(owa_rg)
            std_owa_rg = np.std(owa_rg)

            avg_owa_erm = np.mean(owa_erm)
            std_owa_erm = np.std(owa_erm)

            # reshape ERM / RG for (runs, samples, horizon)
            y_pred_rg = np.squeeze(np.array([e["y_pred"] for e in data_rg["experiments"]]))
            y_pred_erm = np.squeeze(np.array([e["y_pred"] for e in data_erm["experiments"]]))
            y_true = np.squeeze(np.array([e["y_true"] for e in data_rg["experiments"]]))

            # guide (sample, horizon)
            y_guide = np.concatenate([t.detach().cpu().numpy() for t in data_rg["y_baseline"]])

            # EXPAND guide for runs
            y_guide = np.repeat(y_guide[None, :, :], repeats=y_pred_rg.shape[0], axis=0)

            # now all aligned
            err_rg = np.abs(y_true - y_pred_rg)
            err_erm = np.abs(y_true - y_pred_erm)
            err_guide = np.abs(y_true - y_guide)

            win_rate = np.mean(err_rg < err_erm)

            spearman_vals_rg = []
            spearman_vals_erm = []

            for r in range(err_rg.shape[0]):

                guide = err_guide[r].reshape(-1)
                rg = err_rg[r].reshape(-1)
                erm = err_erm[r].reshape(-1)

                rho_rg, _ = spearmanr(guide, rg)
                rho_erm, _ = spearmanr(guide, erm)

                spearman_vals_rg.append(rho_rg)
                spearman_vals_erm.append(rho_erm)

            spearman_rg_mean = np.mean(spearman_vals_rg)
            spearman_rg_std = np.std(spearman_vals_rg)

            spearman_erm_mean = np.mean(spearman_vals_erm)
            spearman_erm_std = np.std(spearman_vals_erm)

            delta_owa = avg_owa_rg - avg_owa_erm
            naive2_smape, naive2_mase = get_naive2_metrics(subset)

            rows.append({
                "model": model,
                "subset": subset,
                "erm_mean": avg_owa_erm,
                "erm_std": std_owa_erm,
                "rg_mean": avg_owa_rg,
                "rg_std": std_owa_rg,
                "delta": delta_owa,
                # "p": p,
                "win_rate": win_rate,
                "guide_owa": data_rg["baseline_owa"],
                "erm_smape_runs": np.array([e["smape"] for e in data_erm["experiments"]]),
                "erm_mase_runs": np.array([e["mase"] for e in data_erm["experiments"]]),
                "rg_smape_runs": np.array([e["smape"] for e in data_rg["experiments"]]),
                "rg_mase_runs": np.array([e["mase"] for e in data_rg["experiments"]]),
                "naive2_smape": naive2_smape,
                "naive2_mase": naive2_mase,
                "spearman_erm_mean": spearman_erm_mean,
                "spearman_erm_std": spearman_erm_std,
                "spearman_rg_mean": spearman_rg_mean,
                "spearman_rg_std": spearman_rg_std,
            })

    df = pd.DataFrame(rows)

    def format_value(val, fmt="{:.3f}"):
        return fmt.format(val)

    rows = []

    # header
    rows.append([
        "Architecture",
        "Subset",
        "Delta_Guide",
        "Spearman_ERM",
        "Spearman_RG",
        "Delta_Spearman",
        "Delta_OWA"
    ])

    subsets_order = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]

    for subset in subsets_order:
        sub_df = df[df["subset"] == subset]

        for _, row in sub_df.iterrows():

            erm_s = row["spearman_erm_mean"]
            rg_s = row["spearman_rg_mean"]

            delta_spearman = rg_s - erm_s
            delta_guide = row["guide_owa"] - row["erm_mean"]
            delta_owa = row["delta"]

            rows.append([
                row["model"],
                row["subset"],
                format_value(delta_guide),
                format_value(erm_s),
                format_value(rg_s),
                format_value(delta_spearman),
                format_value(delta_owa),
            ])

    with open("processed_results/guide_alignment.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


    def format_cell(mean, std):
        if std != 0:
            return f"{mean:.3f} ± {std:.3f}"
        else:
            return f"{mean:.3f}"

    def format_value(val, fmt="{:.2f}"):
        return fmt.format(val)


    df_copy = df.copy()

    df_copy["subset_grouped"] = df_copy["subset"].replace({
        "Weekly": "Others",
        "Daily": "Others",
        "Hourly": "Others"
    })

    subset_counts = {
        "Yearly": 23000,
        "Quarterly": 24000,
        "Monthly": 48000,
        "Weekly": 359,
        "Daily": 4227,
        "Hourly": 414
    }

    rows = []

    # header
    rows.append([
        "Architecture",
        "Subset",
        "ERM_OWA",
        "RG_OWA",
        "Relative_Improvement_percent",
        "Win_rate_percent"
    ])

    subset_order = ["Yearly", "Quarterly", "Monthly", "Others"]
    model_order = list(df_copy["model"].unique())

    for subset in subset_order:
        for model in model_order:

            sub_df = df_copy[
                (df_copy["model"] == model) &
                (df_copy["subset_grouped"] == subset)
            ]

            if len(sub_df) == 0:
                continue

            weights = sub_df["subset"].map(subset_counts).to_numpy()

            erm_owa_runs = pooled_owa_runs(sub_df, "erm", weights)
            rg_owa_runs = pooled_owa_runs(sub_df, "rg", weights)
            erm_mean, erm_std = np.mean(erm_owa_runs), np.std(erm_owa_runs)
            rg_mean, rg_std = np.mean(rg_owa_runs), np.std(rg_owa_runs)

            rel_improvement = ((erm_mean - rg_mean) / erm_mean) * 100
            win_rate = np.average(sub_df["win_rate"], weights=weights) * 100

            rows.append([
                model,
                subset,
                format_cell(erm_mean, erm_std),
                format_cell(rg_mean, rg_std),
                format_value(rel_improvement, "{:.2f}"),
                format_value(win_rate, "{:.1f}")
            ])


    with open("processed_results/m4_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


    plt.rcParams.update({
        "pdf.fonttype": 42,  # embed TrueType fonts in the PDF (NeurIPS)
        "ps.fonttype": 42,
        "pdf.use14corefonts": False,
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16
    })

    df_copy = df.copy()

    df_copy["subset_grouped"] = df_copy["subset"].replace({
        "Weekly": "Others",
        "Daily": "Others",
        "Hourly": "Others"
    })

    subset_counts = {
        "Yearly": 23000,
        "Quarterly": 24000,
        "Monthly": 48000,
        "Weekly": 359,
        "Daily": 4227,
        "Hourly": 414
    }

    def weighted_mean(sub_df, col, weights):
        return np.average(sub_df[col].values, weights=weights)

    x_points = []
    y_points = []
    labels = []

    subset_order = ["Yearly", "Quarterly", "Monthly", "Others"]
    model_order = df_copy["model"].unique()

    for model in model_order:
        for subset in subset_order:

            sub_df = df_copy[
                (df_copy["model"] == model) &
                (df_copy["subset_grouped"] == subset)
            ]

            if len(sub_df) == 0:
                continue

            weights = sub_df["subset"].map(subset_counts).to_numpy()

            # =========================
            # x = Spearman (ERM, guide)
            # =========================
            x = weighted_mean(sub_df, "spearman_erm_mean", weights)

            # =========================
            # y = improvement (%)
            # =========================
            erm_mean = np.mean(pooled_owa_runs(sub_df, "erm", weights))
            rg_mean = np.mean(pooled_owa_runs(sub_df, "rg", weights))
            y = (erm_mean - rg_mean) / erm_mean * 100

            x_points.append(x)
            y_points.append(y)
            labels.append(subset)

    print(f"R^2 (Spearman vs OWA improvement): {linregress(x_points, y_points).rvalue ** 2:.6f}")

    # =========================
    # Plot
    # =========================

    plt.figure(figsize=(7,4))

    sns.regplot(
        x=x_points,
        y=y_points,
        scatter_kws={"s": 60},
        line_kws={"linewidth": 2}
    )

    for subset in subset_order:
        idx = [i for i, l in enumerate(labels) if l == subset]

        plt.scatter(
            np.array(x_points)[idx],
            np.array(y_points)[idx],
            label=subset,
            s=60
        )

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Spearman (ERM, guide)")
    plt.ylabel("OWA improvement \nover ERM (%)")

    plt.legend(ncols=2, columnspacing=0.5)
    plt.tight_layout()

    save_pdf("RG-gain-vs-ERM-Guide-alignment.pdf")
    plt.show()