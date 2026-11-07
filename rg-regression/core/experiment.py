import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
import numpy as np
import random
import pandas as pd
import json
import pyarrow as pa
import pyarrow.parquet as pq
import traceback
import optuna
import shutil
import pickle
import copy
import time

from datasetsforecast.m4 import M4Evaluation, M4, M4Info

from .trainer import train_model
from .schedulers import SchedulerFactory
from .optimizers import OptimizerFactory
from .loss_factory import LossFactory
from .early_stopping import LossEarlyStopping
from .callbacks import SaveBestValLossCallback
from .optuna_finetuning import finetuning, finetuning_only_delta_lambda, finetuning_baseline
from .dataset_builder import DatasetBuilder

from core.utils import save_model, load_model, save_parquet_with_config, load_parquet_with_config

def build_m4_forecast_matrix(dataset_builder, y_pred_series_full):
    forecasts = []
    
    for s in dataset_builder.cached_windows:
        series_id = s["series_id"]
        pred = y_pred_series_full[series_id]
        forecasts.append(pred)

    return np.array(forecasts)

def build_results_dataframe(dataset_builder, results_matrix):
    rows = []

    for i, w in enumerate(dataset_builder.cached_windows):

        series_id = w["series_id"]

        # original y_true (no scale)
        full_series = w["full_series"]
        train_len = w["train_len"]
        y_true = full_series[train_len: train_len + dataset_builder.horizon]

        # forecasts are already denormalized
        y_pred = results_matrix[i]

        for h in range(dataset_builder.horizon):
            rows.append({
                "series_id": series_id,
                "horizon": h + 1,
                "y_true": float(y_true[h]),
                "y_pred": float(y_pred[h])
            })

    return pd.DataFrame(rows)

class Experiment:
    def __init__(self, exp_model_name, exp_loss_name, subset, SEED):
        self.exp_model_name = exp_model_name
        self.exp_loss_name = exp_loss_name
        self.subset = subset
        self.SEED = SEED
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        if exp_model_name == "MLP":
            from models.mlp_model import MyModel
        elif exp_model_name == "GRU":
            from models.gru_model import MyModel
        elif exp_model_name == "TCN":
            from models.tcn_model import MyModel
        else:
            raise ValueError(f"Unsupported model: {exp_model_name}")

        self.MyModel = MyModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA available? {torch.cuda.is_available()}")

        # Chosen as the smallest stride satisfying horizon <= stride < the maximum stride yielding 100k windows.
        # strides = {"Hourly": 48, "Daily": 99, "Weekly": 13, "Monthly": 70, "Quarterly": 16, "Yearly": 6}
        strides = {"Hourly": 48, "Daily": 198, "Weekly": 13, "Monthly": 112, "Quarterly": 32, "Yearly": 6}
        self.stride = strides[subset]

        # Chosen such that 3*horizon < window_size < second decile - horizon, ideally 4*horizon (except for Yearly).
        window_sizes = {"Hourly": 192, "Daily": 56, "Weekly": 52, "Monthly": 72, "Quarterly": 32, "Yearly": 18}
        self.window_size = window_sizes[subset]


    def run_pre_train(self):
        
        experiment_name = (
            f"{self.MyModel.model_name}_" + 
            f"{self.subset}_" +
            f"{self.exp_loss_name}"
        )

        experiment_dir = os.path.join("results", experiment_name)

        if not os.path.exists(os.path.join(experiment_dir, "median-model.pt")):
            os.makedirs(experiment_dir, exist_ok=True)
            print("Folder created:", experiment_dir)

            try:
                loss_factory = LossFactory(loss_name=self.exp_loss_name)

                dataset_builder = DatasetBuilder(baseline_method="no", subset=self.subset, window_size=self.window_size, stride=self.stride)
                dataset_builder.build()

                study_path = os.path.join(experiment_dir, "optuna_study.db")
                finetuning(experiment_dir, dataset_builder, self.MyModel, loss_factory, plot_manager = None, max_trials=50, epochs=30, patience=15, window_size=self.window_size, SEED=self.SEED)

                csv_path = os.path.join(experiment_dir, "experiment-results.csv")

                if os.path.exists(csv_path):
                    df_results = pd.read_csv(csv_path)
                else:
                    df_results = pd.DataFrame(
                        columns=[
                            "name", "model",
                            "loss", "loss_baseline",
                            "sMAPE", "MASE","OWA",
                            "train_time", "epochs", "time_per_epoch",
                        ]
                    )

                experiment_data_path = os.path.join(experiment_dir, "experiment_data.parquet")
                if os.path.exists(experiment_data_path):
                    df_experiment, df_experiment_config = load_parquet_with_config(experiment_data_path)
                else:
                    df_experiment = pd.DataFrame()
                    df_experiment_config = {"total_runs": 0}

                # Carrega melhores parâmetros do Optuna
                study = optuna.load_study(
                    study_name='experiment_with_indicators',
                    storage=f"sqlite:///{study_path}",
                )
                model_best_params = study.best_params

                model_params = copy.deepcopy(model_best_params)
                model_params, other_params = self.MyModel.prepare_model_params(model_params)
                model_params["num_features"] = self.window_size
                model_params["output_dim"] = dataset_builder.horizon
                
                train_loader, val_loader, test_loader = dataset_builder.build_loaders(batch_size=other_params["batch_size"])

                max_repeat_training_times = 5
                best_loss = float("inf")
                losses = {}

                # -------------------------
                # Training loop
                # -------------------------
                for run in range(df_experiment_config["total_runs"] + 1,
                            df_experiment_config["total_runs"] + 1 + max_repeat_training_times):
                    
                    # rng = np.random.default_rng(seed=run)
                    # rng_seed = rng.integers(low=0, high=(2**32) - 1, endpoint=True)
                    # custom_init_weights(model,seed=rng_seed)

                    model = self.MyModel(**model_params).to(self.device)
                    self.MyModel.model_postprocess(model)

                    scheduler_factory = SchedulerFactory(scheduler_name="plateau")
                    optimizer_factory = OptimizerFactory(
                        model, optimizer_name="adamw",
                        lr=other_params["lr"], weight_decay=other_params["weight_decay"]
                    )
                    
                    save_best_callback = SaveBestValLossCallback(
                        save_dir=experiment_dir, save_name="best-model.pt", best_init=best_loss
                    )
                    save_callback_temp = SaveBestValLossCallback(
                        save_dir=experiment_dir, save_name=f"best-model_r{run}.pt", best_init=float("inf")
                    )

                    loss = loss_factory.create()

                    start_time = time.time()
                    state = train_model(
                        model, train_loader, val_loader, loss, optimizer_factory, scheduler_factory,
                        params=model_best_params, epochs=3000, device=self.device, stop_criteria=[LossEarlyStopping(200)],
                        callbacks=[save_best_callback, save_callback_temp], save_best=False, verbose=True,
                        seed = run
                    )
                    train_time = time.time() - start_time

                    # -------------------------
                    # Save best model
                    # -------------------------
                    val_loss = np.min(state.metrics.get("val_loss_per_epoch"))
                    if val_loss < best_loss:
                        best_loss = val_loss

                    losses[run] = val_loss
                    # -------------------------
                    # Evaluation on test set
                    # -------------------------
                    model = load_model(self.MyModel, os.path.join(experiment_dir, f"best-model_r{run}.pt"), self.device, self.window_size, dataset_builder.horizon)

                    model.eval()
                    y_pred_list = []
                    y_true_list = []

                    with torch.no_grad():
                        for x_test, y_test, _ in test_loader:
                            x_test = x_test.to(self.device)
                            outputs = model(x_test).cpu().numpy()
                            y_pred_list.append(outputs)
                            y_true_list.append(y_test.numpy())

                    y_pred_model_raw = np.concatenate(y_pred_list, axis=0)
                    y_true_model_raw = np.concatenate(y_true_list, axis=0)

                    # -------------------------
                    # Reconstruct full forecasts by series_id
                    # -------------------------
                    y_pred_series_full = dataset_builder.reconstruct_full_series_predictions(y_pred_model_raw)

                    # ===============================
                    # Build forecast matrix
                    # ===============================
                    results_matrix = build_m4_forecast_matrix(dataset_builder, y_pred_series_full)

                    results_df = build_results_dataframe(dataset_builder, results_matrix)
                    results_df.to_parquet(
                        os.path.join(experiment_dir, "m4_predictions.parquet"),
                        index=False
                    )

                    # ===============================
                    # Evaluate with official M4 code
                    # ===============================
                    evaluation = M4Evaluation.evaluate(
                        directory="data/M4",   # pasta onde dataset M4 está
                        group=self.subset,
                        y_hat=results_matrix
                    )

                    avg_smape = evaluation.loc[self.subset, "SMAPE"]
                    avg_mase = evaluation.loc[self.subset, "MASE"]
                    owa = evaluation.loc[self.subset, "OWA"]

                    print(f"sMAPE={avg_smape:.4f}, MASE={avg_mase:.4f}, OWA={owa:.4f}")

                    # -------------------------
                    # Log in results DataFrame
                    # -------------------------
                    df_results.loc[len(df_results)] = {
                        "name": experiment_name,
                        "model": self.MyModel.model_name,
                        "loss": self.exp_loss_name,
                        "loss_baseline": "no",
                        "sMAPE": avg_smape,
                        "MASE": avg_mase,
                        "OWA": owa,
                        "train_time": train_time,
                        "epochs": len(state.metrics.get("val_loss_per_epoch")),
                        "time_per_epoch": train_time / len(state.metrics.get("val_loss_per_epoch")),
                    }

                    # Salva parciais
                    df_results.to_csv(csv_path, index=False)
                    df_experiment_config["total_runs"] = run
                    save_parquet_with_config(df_experiment, experiment_data_path, df_experiment_config)
                    print("All results have been saved.")
                
                # -------------------------
                # Save median model
                # -------------------------
                # list of (run, loss) ordered by loss
                sorted_runs = sorted(losses.items(), key=lambda x: x[1])
                median_idx = len(sorted_runs) // 2
                median_run = sorted_runs[median_idx][0]
                median_model_path = os.path.join(
                    experiment_dir,
                    f"best-model_r{median_run}.pt"
                )
                shutil.copy(
                    median_model_path,
                    os.path.join(experiment_dir, "median-model.pt")
                )

            except Exception as e:
                # Path to error file
                error_file = os.path.join(experiment_dir, "error.txt")
                
                # Write error details
                with open(error_file, "w") as f:
                    f.write("An error occurred:\n")
                    f.write(str(e) + "\n\n")
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())
                
                print("An error occurred:", e)
                print(f"Error saved to {error_file}")
        else:
            print("Experiment already done.")


    def run_baseline_train(self):
        experiment_name = (
            f"{self.MyModel.model_name}_" + 
            f"{self.subset}_" +
            f"{self.exp_loss_name}_baseline"
        )
        experiment_dir = os.path.join("results", experiment_name)

        if not os.path.exists(os.path.join(experiment_dir, "median-model.pt")):
            os.makedirs(experiment_dir, exist_ok=True)
            print("Diretório criado:", experiment_dir)

            try:
                loss_factory = LossFactory(loss_name=self.exp_loss_name)
        
                # if exp_loss_name.startswith("baseline"):
                #     compare_with_standard = True
                #     load_standard_state = True
                #     use_control = False
                # else:
                #     compare_with_standard = False
                #     load_standard_state = False
                #     use_control = True

                dataset_builder = DatasetBuilder(baseline_method="no", subset=self.subset, window_size=self.window_size, stride=self.stride)
                dataset_builder.build()

                # ==============================================
                # FINE-TUNING WITH OPTUNA
                # ==============================================

                # standard_experiment_name = (
                #     f"{self.MyModel.model_name}_" + 
                #     f"{self.subset}_" +
                #     f"{self.exp_loss_name}_pre-train"
                # )
                standard_experiment_name = (
                    f"{self.MyModel.model_name}_" + 
                    f"{self.subset}_" +
                    f"{self.exp_loss_name}"
                )

                standard_experiment_dir = os.path.join("results", standard_experiment_name)
                standard_study_path = os.path.join(standard_experiment_dir, "optuna_study.db")
                standard_study = optuna.load_study(
                    study_name='experiment_with_indicators',
                    storage=f"sqlite:///{standard_study_path}",
                )
                standard_model_best_params = standard_study.best_params

                study_path = os.path.join(experiment_dir, "optuna_study.db")
                standard_state_path = os.path.join(standard_experiment_dir, "median-model.pt")
                checkpoint = torch.load(standard_state_path, map_location=self.device, weights_only=False)["state_dict"]
                    
                finetuning_baseline(experiment_dir, dataset_builder, self.MyModel, loss_factory, plot_manager = None, best_params=standard_model_best_params, checkpoint=checkpoint, max_trials=25, epochs=1000, patience=100, window_size=self.window_size, SEED=self.SEED)
    
                # -------------------------
                # Setup experiment
                # -------------------------
                csv_path = os.path.join(experiment_dir, "experiment-results.csv")

                if os.path.exists(csv_path):
                    df_results = pd.read_csv(csv_path)
                else:
                    df_results = pd.DataFrame(
                        columns=[
                            "name", "model",
                            "loss", "loss_baseline",
                            "sMAPE", "MASE","OWA",
                            "train_time", "epochs", "time_per_epoch",
                        ]
                    )

                experiment_data_path = os.path.join(experiment_dir, "experiment_data.parquet")
                if os.path.exists(experiment_data_path):
                    df_experiment, df_experiment_config = load_parquet_with_config(experiment_data_path)
                else:
                    df_experiment = pd.DataFrame()
                    df_experiment_config = {"total_runs": 0}

                # Load the best parameters from Optuna
                study = optuna.load_study(
                    study_name='experiment_with_indicators',
                    storage=f"sqlite:///{study_path}",
                )
                model_best_params = study.best_params

                standard_model_best_params.update(study.best_params)
                model_best_params = standard_model_best_params

                model_params = copy.deepcopy(model_best_params)
                model_params, other_params = self.MyModel.prepare_model_params(model_params)
                model_params["num_features"] = self.window_size
                model_params["output_dim"] = dataset_builder.horizon
        
                train_loader, val_loader, test_loader = dataset_builder.build_loaders(batch_size=other_params["batch_size"])

                max_repeat_training_times = 5
                best_loss = float("inf")
                losses = {}

                # -------------------------
                # Training loop
                # -------------------------
                for run in range(df_experiment_config["total_runs"] + 1,
                            df_experiment_config["total_runs"] + 1 + max_repeat_training_times):
                    
                    # rng = np.random.default_rng(seed=run)
                    # rng_seed = rng.integers(low=0, high=(2**32) - 1, endpoint=True)
                    # custom_init_weights(model,seed=rng_seed)

                    model = self.MyModel(**model_params).to(self.device)
                    standard_state_path = os.path.join(standard_experiment_dir, "median-model.pt")
                    checkpoint = torch.load(standard_state_path, map_location=self.device, weights_only=False)["state_dict"]
                    model.load_state_dict(checkpoint)
                    self.MyModel.model_postprocess(model)

                    scheduler_factory = SchedulerFactory(scheduler_name="plateau", patience = 90)
                    optimizer_factory = OptimizerFactory(
                        model, optimizer_name="adamw",
                        lr=other_params["lr"], weight_decay=other_params["weight_decay"]
                    )
                    
                    save_best_callback = SaveBestValLossCallback(
                        save_dir=experiment_dir, save_name="best-model.pt", best_init=best_loss
                    )
                    save_callback_temp = SaveBestValLossCallback(
                        save_dir=experiment_dir, save_name=f"best-model_r{run}.pt", best_init=float("inf")
                    )

                    loss = loss_factory.create()

                    start_time = time.time()
                    state = train_model(
                        model, train_loader, val_loader, loss, optimizer_factory, scheduler_factory,
                        params=model_best_params, epochs=3000, device=self.device, stop_criteria=[LossEarlyStopping(200)],
                        callbacks=[save_best_callback, save_callback_temp], save_best=False, verbose=True,
                        seed = run
                    )
                    train_time = time.time() - start_time

                    # -------------------------
                    # Save best model
                    # -------------------------
                    val_loss = np.min(state.metrics.get("val_loss_per_epoch"))
                    if val_loss < best_loss:
                        best_loss = val_loss

                    losses[run] = val_loss
                    # -------------------------
                    # Evaluation on test set
                    # -------------------------
                    model = load_model(self.MyModel, os.path.join(experiment_dir, f"best-model_r{run}.pt"), self.device, self.window_size, dataset_builder.horizon)

                    model.eval()
                    y_pred_list = []
                    y_true_list = []

                    with torch.no_grad():
                        for x_test, y_test, _ in test_loader:
                            x_test = x_test.to(self.device)
                            outputs = model(x_test).cpu().numpy()
                            y_pred_list.append(outputs)
                            y_true_list.append(y_test.numpy())

                    y_pred_model_raw = np.concatenate(y_pred_list, axis=0)
                    y_true_model_raw = np.concatenate(y_true_list, axis=0)

                    # -------------------------
                    # Reconstruct full forecasts by series_id
                    # -------------------------
                    y_pred_series_full = dataset_builder.reconstruct_full_series_predictions(y_pred_model_raw)

                    # ===============================
                    # Build forecast matrix
                    # ===============================
                    results_matrix = build_m4_forecast_matrix(dataset_builder, y_pred_series_full)

                    results_df = build_results_dataframe(dataset_builder, results_matrix)
                    results_df.to_parquet(
                        os.path.join(experiment_dir, "m4_predictions.parquet"),
                        index=False
                    )

                    # ===============================
                    # Evaluate with official M4 code
                    # ===============================
                    evaluation = M4Evaluation.evaluate(
                        directory="data/M4",   # pasta onde dataset M4 está
                        group=self.subset,
                        y_hat=results_matrix
                    )

                    avg_smape = evaluation.loc[self.subset, "SMAPE"]
                    avg_mase = evaluation.loc[self.subset, "MASE"]
                    owa = evaluation.loc[self.subset, "OWA"]

                    print(f"sMAPE={avg_smape:.4f}, MASE={avg_mase:.4f}, OWA={owa:.4f}")

                    # -------------------------
                    # Log in results DataFrame
                    # -------------------------
                    df_results.loc[len(df_results)] = {
                        "name": experiment_name,
                        "model": self.MyModel.model_name,
                        "loss": self.exp_loss_name,
                        "loss_baseline": "no",
                        "sMAPE": avg_smape,
                        "MASE": avg_mase,
                        "OWA": owa,
                        "train_time": train_time,
                        "epochs": len(state.metrics.get("val_loss_per_epoch")),
                        "time_per_epoch": train_time / len(state.metrics.get("val_loss_per_epoch")),
                    }

                    # Save partial results
                    df_results.to_csv(csv_path, index=False)
                    df_experiment_config["total_runs"] = run
                    save_parquet_with_config(df_experiment, experiment_data_path, df_experiment_config)
                    print("All results have been saved.")
                
                # -------------------------
                # Save median model
                # -------------------------
                # list of (run, loss) ordered by loss
                sorted_runs = sorted(losses.items(), key=lambda x: x[1])
                median_idx = len(sorted_runs) // 2
                median_run = sorted_runs[median_idx][0]
                median_model_path = os.path.join(
                    experiment_dir,
                    f"best-model_r{median_run}.pt"
                )
                shutil.copy(
                    median_model_path,
                    os.path.join(experiment_dir, "median-model.pt")
                )

            except Exception as e:
                # Path to error file
                error_file = os.path.join(experiment_dir, "error.txt")
                
                # Write error details
                with open(error_file, "w") as f:
                    f.write("An error occurred:\n")
                    f.write(str(e) + "\n\n")
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())
                
                print("An error occurred:", e)
                print(f"Error saved to {error_file}")
        else:
            print("Experiment already done.")
        
        



    def run_guided_train(self, guided_loss_name, baseline_method="NARMAX"):
        experiment_name = (
            f"{self.MyModel.model_name}_" + 
            f"{self.subset}_" +
            f"{guided_loss_name}-{baseline_method}"
        )
        experiment_dir = os.path.join("results", experiment_name)

        if not os.path.exists(os.path.join(experiment_dir, "median-model.pt")):
            os.makedirs(experiment_dir, exist_ok=True)
            print("Folder created:", experiment_dir)

            try:
                loss_factory = LossFactory(loss_name=guided_loss_name)

                dataset_builder = DatasetBuilder(baseline_method=baseline_method, subset=self.subset, window_size=self.window_size, stride=self.stride)
                dataset_builder.build()

                # ==============================================
                # FINE-TUNING WITH OPTUNA
                # ==============================================

                standard_experiment_name = (
                    f"{self.MyModel.model_name}_" + 
                    f"{self.subset}_" +
                    f"{self.exp_loss_name}"
                )

                standard_experiment_dir = os.path.join("results", standard_experiment_name)
                standard_study_path = os.path.join(standard_experiment_dir, "optuna_study.db")
                standard_study = optuna.load_study(
                    study_name='experiment_with_indicators',
                    storage=f"sqlite:///{standard_study_path}",
                )
                standard_model_best_params = standard_study.best_params

                study_path = os.path.join(experiment_dir, "optuna_study.db")
                standard_state_path = os.path.join(standard_experiment_dir, "median-model.pt")
                checkpoint = torch.load(standard_state_path, map_location=self.device, weights_only=False)["state_dict"]
                finetuning_only_delta_lambda(experiment_dir, dataset_builder, self.MyModel, loss_factory, plot_manager = None, best_params=standard_model_best_params, checkpoint=checkpoint, max_trials=25, epochs=1000, patience=100, window_size=self.window_size, SEED=self.SEED)

                # -------------------------
                # Setup experiment
                # -------------------------
                csv_path = os.path.join(experiment_dir, "experiment-results.csv")

                if os.path.exists(csv_path):
                    df_results = pd.read_csv(csv_path)
                else:
                    df_results = pd.DataFrame(
                        columns=[
                            "name", "model",
                            "loss", "loss_baseline",
                            "sMAPE", "MASE","OWA",
                            "train_time", "epochs", "time_per_epoch",
                        ]
                    )

                experiment_data_path = os.path.join(experiment_dir, "experiment_data.parquet")
                if os.path.exists(experiment_data_path):
                    df_experiment, df_experiment_config = load_parquet_with_config(experiment_data_path)
                else:
                    df_experiment = pd.DataFrame()
                    df_experiment_config = {"total_runs": 0}

                # Load the best parameters from Optuna
                study = optuna.load_study(
                    study_name='experiment_with_indicators',
                    storage=f"sqlite:///{study_path}",
                )
                model_best_params = study.best_params

                standard_model_best_params.update(study.best_params)
                model_best_params = standard_model_best_params

                model_params = copy.deepcopy(model_best_params)
                model_params, other_params = self.MyModel.prepare_model_params(model_params)
                model_params["num_features"] = self.window_size
                model_params["output_dim"] = dataset_builder.horizon
        
                train_loader, val_loader, test_loader = dataset_builder.build_loaders(batch_size=other_params["batch_size"])

                max_repeat_training_times = 5
                best_loss = float("inf")
                losses = {}

                # -------------------------
                # Training loop
                # -------------------------
                for run in range(df_experiment_config["total_runs"] + 1,
                            df_experiment_config["total_runs"] + 1 + max_repeat_training_times):
                    
                    # rng = np.random.default_rng(seed=run)
                    # rng_seed = rng.integers(low=0, high=(2**32) - 1, endpoint=True)
                    # custom_init_weights(model,seed=rng_seed)

                    model = self.MyModel(**model_params).to(self.device)
                    standard_state_path = os.path.join(standard_experiment_dir, "median-model.pt")
                    checkpoint = torch.load(standard_state_path, map_location=self.device, weights_only=False)["state_dict"]
                    model.load_state_dict(checkpoint)
                    self.MyModel.model_postprocess(model)

                    scheduler_factory = SchedulerFactory(scheduler_name="plateau", patience = 90)
                    optimizer_factory = OptimizerFactory(
                        model, optimizer_name="adamw",
                        lr=other_params["lr"], weight_decay=other_params["weight_decay"]
                    )
                    
                    save_best_callback = SaveBestValLossCallback(
                        save_dir=experiment_dir, save_name="best-model.pt", best_init=best_loss
                    )
                    save_callback_temp = SaveBestValLossCallback(
                        save_dir=experiment_dir, save_name=f"best-model_r{run}.pt", best_init=float("inf")
                    )

                    loss = loss_factory.create(lamb=other_params["lamb"], delta=other_params["delta"], margin=other_params["margin"])

                    start_time = time.time()
                    state = train_model(
                        model, train_loader, val_loader, loss, optimizer_factory, scheduler_factory,
                        params=model_best_params, epochs=3000, device=self.device, stop_criteria=[LossEarlyStopping(200)],
                        callbacks=[save_best_callback, save_callback_temp], save_best=False, verbose=True,
                        seed = run
                    )
                    train_time = time.time() - start_time

                    # -------------------------
                    # Save best model
                    # -------------------------
                    val_loss = np.min(state.metrics.get("val_loss_per_epoch"))
                    if val_loss < best_loss:
                        best_loss = val_loss

                    losses[run] = val_loss
                    # -------------------------
                    # Evaluation on test set
                    # -------------------------
                    model = load_model(self.MyModel, os.path.join(experiment_dir, f"best-model_r{run}.pt"), self.device, self.window_size, dataset_builder.horizon)

                    model.eval()
                    y_pred_list = []
                    y_true_list = []

                    with torch.no_grad():
                        for x_test, y_test, _ in test_loader:
                            x_test = x_test.to(self.device)
                            outputs = model(x_test).cpu().numpy()
                            y_pred_list.append(outputs)
                            y_true_list.append(y_test.numpy())

                    y_pred_model_raw = np.concatenate(y_pred_list, axis=0)
                    y_true_model_raw = np.concatenate(y_true_list, axis=0)

                    # -------------------------
                    # Reconstruct full forecasts by series_id
                    # -------------------------
                    y_pred_series_full = dataset_builder.reconstruct_full_series_predictions(y_pred_model_raw)

                    # ===============================
                    # Build forecast matrix
                    # ===============================
                    results_matrix = build_m4_forecast_matrix(dataset_builder, y_pred_series_full)

                    results_df = build_results_dataframe(dataset_builder, results_matrix)
                    results_df.to_parquet(
                        os.path.join(experiment_dir, "m4_predictions.parquet"),
                        index=False
                    )

                    # ===============================
                    # Evaluate with official M4 code
                    # ===============================
                    evaluation = M4Evaluation.evaluate(
                        directory="data/M4",   # pasta onde dataset M4 está
                        group=self.subset,
                        y_hat=results_matrix
                    )

                    avg_smape = evaluation.loc[self.subset, "SMAPE"]
                    avg_mase = evaluation.loc[self.subset, "MASE"]
                    owa = evaluation.loc[self.subset, "OWA"]

                    print(f"sMAPE={avg_smape:.4f}, MASE={avg_mase:.4f}, OWA={owa:.4f}")

                    # -------------------------
                    # Log in results DataFrame
                    # -------------------------
                    df_results.loc[len(df_results)] = {
                        "name": experiment_name,
                        "model": self.MyModel.model_name,
                        "loss": guided_loss_name,
                        "loss_baseline": baseline_method,
                        "sMAPE": avg_smape,
                        "MASE": avg_mase,
                        "OWA": owa,
                        "train_time": train_time,
                        "epochs": len(state.metrics.get("val_loss_per_epoch")),
                        "time_per_epoch": train_time / len(state.metrics.get("val_loss_per_epoch")),
                    }

                    # Save partial results
                    df_results.to_csv(csv_path, index=False)
                    df_experiment_config["total_runs"] = run
                    save_parquet_with_config(df_experiment, experiment_data_path, df_experiment_config)
                    print("All results have been saved.")
                
                # -------------------------
                # Save median model
                # -------------------------
                # list of (run, loss) ordered by loss
                sorted_runs = sorted(losses.items(), key=lambda x: x[1])
                median_idx = len(sorted_runs) // 2
                median_run = sorted_runs[median_idx][0]
                median_model_path = os.path.join(
                    experiment_dir,
                    f"best-model_r{median_run}.pt"
                )
                shutil.copy(
                    median_model_path,
                    os.path.join(experiment_dir, "median-model.pt")
                )

            except Exception as e:
                # Path to error file
                error_file = os.path.join(experiment_dir, "error.txt")
                
                # Write error details
                with open(error_file, "w") as f:
                    f.write("An error occurred:\n")
                    f.write(str(e) + "\n\n")
                    f.write("Traceback:\n")
                    f.write(traceback.format_exc())
                
                print("An error occurred:", e)
                print(f"Error saved to {error_file}")
        else:
            print("Experiment already done.")