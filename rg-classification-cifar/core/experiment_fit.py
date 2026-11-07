# Final repeated-seed fitting, checkpoint persistence and resume.
import os
import time
import shutil
import hashlib
import json
import random
import copy
import numpy as np
import pandas as pd
import torch
from core.train import train_model, SchedulerFactory, OptimizerFactory, LossFactory, SaveBestValLossCallback
from core.utils import load_model, save_parquet_with_config, load_parquet_with_config
from core.experiment_protocol import (
    MODE_TO_METHOD, GUIDED_METHODS, FIXED_WEIGHT_DECAY, FIXED_BATCH_SIZE,
    PROTOCOL_ID, LR_CHOICES,
)
from core.experiment_plan import verify_dependency_manifest
from core.experiment_paths import method_dir, guide_dir
from core.evaluation_metrics import evaluate_on_test
from core.experiment_method import method_spec
from core.model_factory import model_parameters, instantiate

RESULTS_COLUMNS = [
    "name", "model", "type", "state_loaded", "loss", "noise_type",
    "guide_model", "guide_noise_type", "loss-value",
    "log-loss", "accuracy", "precision", "recall",
    "balanced_accuracy", "mcc", "cohen_kappa",
    "auc_roc", "pr_auc", "f1_score",
    "train_time", "epochs", "time_per_epoch", "seed", "trial_number",
]

def load_progress(experiment_dir):
    # Validate paired CSV/Parquet/checkpoint artifacts before resuming.
    csv_path = os.path.join(experiment_dir, "experiment-results.csv")
    parquet_path = os.path.join(experiment_dir, "experiment_data.parquet")
    if os.path.exists(csv_path):
        rows = pd.read_csv(csv_path)
    else:
        rows = pd.DataFrame(columns=RESULTS_COLUMNS)
    if os.path.exists(parquet_path):
        predictions, config = load_parquet_with_config(parquet_path)
    else:
        predictions, config = pd.DataFrame(), {"total_runs": 0}

    n = config.get("total_runs", 0)
    if not isinstance(n, int) or n < 0 or len(rows) != n:
        raise RuntimeError(
            f"Inconsistent CIFAR resume state in {experiment_dir}: "
            f"CSV rows={len(rows)}, Parquet total_runs={n}. "
            "Restore or repair the paired files before resuming."
        )
    losses = {}
    for run in range(1, n + 1):
        for column in (f"y_true_r{run}", f"y_pred_r{run}"):
            if column not in predictions:
                raise RuntimeError(f"Missing {column} in {parquet_path}")
        if not os.path.isfile(os.path.join(experiment_dir, f"best-model_r{run}.pt")):
            raise RuntimeError(f"Missing checkpoint for completed run {run} in {experiment_dir}")
        val_loss = float(rows.iloc[run - 1]["loss-value"])
        if not np.isfinite(val_loss):
            raise RuntimeError(f"Non-finite validation loss in completed run {run}")
        losses[run] = val_loss
    return rows, predictions, config, losses


def checkpoint_for_rank(experiment_dir, losses, *, rank):
    # Select a run by saved objective validation loss.
    if not losses:
        raise ValueError("No completed runs to rank")
    ordered = sorted(losses.items(), key=lambda item: item[1])
    selected = ordered[rank][0]
    return os.path.join(experiment_dir, f"best-model_r{selected}.pt")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_snapshot(architecture_params, guide_path=None):
    # Freeze the architecture and reference actually used in the final runs.
    result = {"protocol": PROTOCOL_ID, "architecture_params": architecture_params}
    if guide_path is not None:
        if not os.path.isfile(guide_path):
            raise FileNotFoundError(f"Guide checkpoint missing: {guide_path}")
        result["guide_checkpoint_sha256"] = file_sha256(guide_path)
    return result


def lock_dependencies(experiment_dir, snapshot, completed_runs):
    path = os.path.join(experiment_dir, "dependencies.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous != snapshot:
            raise RuntimeError(f"Architecture or guide checkpoint changed since this "
                               f"condition started: {path}")
    elif completed_runs:
        raise RuntimeError(f"Completed runs have no dependency provenance: {path}")
    else:
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(snapshot, stream, indent=2, sort_keys=True)
            stream.write("\n")


def lock_selected_hyperparameters(experiment_dir, params, completed_runs):
    # Reject a resume if Optuna's selected parameters changed after fitting began.
    path = os.path.join(experiment_dir, "selected-hyperparameters.json")
    expected = dict(params)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as stream:
            recorded = json.load(stream)
        if recorded != expected:
            raise RuntimeError(
                f"Selected hyperparameters changed after training began: {path}"
            )
    elif completed_runs:
        raise RuntimeError(f"Completed runs have no hyperparameter provenance: {path}")
    else:
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(expected, stream, indent=2, sort_keys=True)
            stream.write("\n")


class FitOperations:
    def run_training(self, mode, n_runs):
        if mode not in MODE_TO_METHOD:
            raise ValueError(
                f"Unsupported training mode '{mode}'. Valid modes: "
                f"{sorted(MODE_TO_METHOD)} (plus 'state' for the architecture search)."
            )

        method = MODE_TO_METHOD[mode]
        self.requested_runs = n_runs
        uses_guide = method in GUIDED_METHODS

        result_path = (guide_dir(self.noise_type) if self.MyModel.model_name == "ResNet56"
                       else method_dir(self.MyModel.model_name, self.noise_type, method,
                                       self.exp_guide_model if uses_guide else None,
                                       self.guide_noise_type if uses_guide else None,
                                       tag=self.tag))
        experiment_name = result_path.name
        experiment_dir = str(result_path)

        os.makedirs(experiment_dir, exist_ok=True)
        print("Folder:", experiment_dir)
        csv_path = os.path.join(experiment_dir, "experiment-results.csv")
        experiment_data_path = os.path.join(experiment_dir, "experiment_data.parquet")
        df_results, df_experiment, df_experiment_config, losses = load_progress(experiment_dir)
        completed_runs = df_experiment_config["total_runs"]
        if hasattr(self, "load_architecture_params"):
            arch_params = self.load_architecture_params()
            guide_path = None
            if uses_guide and self.exp_guide_model == "ResNet56":
                guide_dir_path = str(guide_dir(self.guide_noise_type))
                verify_dependency_manifest(guide_dir_path, {
                    "protocol": PROTOCOL_ID, "method": "guide", "model": "ResNet56",
                    "noise": self.guide_noise_type, "trials": self.n_trials,
                    "runs": n_runs, "epochs": self.total_epochs,
                    "lr_choices": list(LR_CHOICES),
                })
                guide_path = os.path.join(guide_dir_path, "best-model.pt")
            lock_dependencies(experiment_dir, dependency_snapshot(arch_params, guide_path), completed_runs)
        if completed_runs > n_runs:
            raise RuntimeError(f"Stored {completed_runs} runs exceed the requested {n_runs}")
        if completed_runs == n_runs:
            if losses and not os.path.isfile(os.path.join(experiment_dir, "best-model.pt")):
                shutil.copyfile(checkpoint_for_rank(experiment_dir, losses, rank=0),
                                os.path.join(experiment_dir, "best-model.pt"))
            if not os.path.isfile(os.path.join(experiment_dir, "median-model.pt")):
                shutil.copyfile(checkpoint_for_rank(experiment_dir, losses, rank=len(losses)//2),
                                os.path.join(experiment_dir, "median-model.pt"))
            print(f"Experiment already complete ({completed_runs}/{n_runs} runs).")
            return
        if os.path.isfile(os.path.join(experiment_dir, "median-model.pt")):
            raise RuntimeError("Median checkpoint exists for an incomplete run set")
        best_loss = min(losses.values(), default=float("inf"))
        if losses:


            shutil.copyfile(checkpoint_for_rank(experiment_dir, losses, rank=0),
                            os.path.join(experiment_dir, "best-model.pt"))

        dataset_builder = self.build_dataset(use_guide=uses_guide)
        if not hasattr(self, "load_architecture_params"):
            arch_params = {}
        best_params = self.resolve_hyperparameters(experiment_dir, dataset_builder, arch_params, method)
        lock_selected_hyperparameters(experiment_dir, best_params, completed_runs)

        model_params = model_parameters(self.MyModel, arch_params)


        # Save architecture and tuned parameters to reconstruct the model.
        checkpoint_params = dict(arch_params)
        checkpoint_params.update(best_params)

        train_loader, val_loader, test_loader = dataset_builder.build_loaders(batch_size=FIXED_BATCH_SIZE)


        for run in range(df_experiment_config["total_runs"] + 1, n_runs + 1):
            torch.manual_seed(self.SEED + run)
            np.random.seed(self.SEED + run)
            random.seed(self.SEED + run)
            torch.cuda.manual_seed_all(self.SEED + run)

            model = instantiate(self.MyModel, model_params, self.device)

            loss_factory = LossFactory(loss_name=self.exp_loss_name)
            loss = self.build_loss(loss_factory, best_params, method)

            scheduler_factory = SchedulerFactory(scheduler_name="cosine", T_max=self.total_epochs)
            optimizer_factory = OptimizerFactory(
                model, optimizer_name="sgd",
                lr=self.effective_lr(best_params), weight_decay=FIXED_WEIGHT_DECAY,
            )

            save_best_callback = SaveBestValLossCallback(
                save_dir=experiment_dir, save_name="best-model.pt", best_init=best_loss
            )
            save_callback_temp = SaveBestValLossCallback(
                save_dir=experiment_dir, save_name=f"best-model_r{run}.pt", best_init=float("inf")
            )

            start_time = time.time()
            state = train_model(
                model, train_loader, val_loader, loss,
                optimizer_factory, scheduler_factory,
                params=checkpoint_params, epochs=self.total_epochs, device=self.device,
                callbacks=[save_best_callback, save_callback_temp], save_best=False,
                verbose=True, seed=run,
                val_alt_labels=getattr(dataset_builder, "val_labels_noisy", None),
            )
            train_time = time.time() - start_time

            history = state.metrics.get("val_loss_per_epoch")
            if not history or not np.isfinite(np.asarray(history, dtype=float)).all():
                raise RuntimeError(f"Run {run} produced invalid validation losses")
            val_loss = float(np.min(history))
            checkpoint_path = os.path.join(experiment_dir, f"best-model_r{run}.pt")
            if not os.path.isfile(checkpoint_path):
                raise RuntimeError(f"Run {run} did not save a validation-selected checkpoint")
            if val_loss < best_loss:
                best_loss = val_loss
            losses[run] = val_loss

            model = load_model(
                self.MyModel,
                os.path.join(experiment_dir, f"best-model_r{run}.pt"),
                self.device,
                (3, 32, 32),
                10,
            )

            metrics, y_true, y_pred = evaluate_on_test(model, test_loader, self.device)

            df_results.loc[len(df_results)] = {
                "name": experiment_name,
                "model": self.MyModel.model_name,
                "type": method_spec(method).result_type,
                "state_loaded": False,
                "loss": self.exp_loss_name,
                "noise_type": self.noise_type,
                "guide_model": self.exp_guide_model if uses_guide else None,
                "guide_noise_type": self.guide_noise_type if uses_guide else None,
                "loss-value": val_loss,
                "log-loss": metrics["log-loss"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "mcc": metrics["mcc"],
                "cohen_kappa": metrics["cohen_kappa"],
                "auc_roc": metrics["auc_roc"],
                "pr_auc": metrics["pr_auc"],
                "f1_score": metrics["f1_score"],
                "train_time": train_time,
                "epochs": len(state.metrics.get("val_loss_per_epoch")),
                "time_per_epoch": train_time / len(state.metrics.get("val_loss_per_epoch")),
                "seed": self.SEED + run,
                "trial_number": getattr(self, "selected_trial_number", None),
            }
            df_results.to_csv(csv_path, index=False)

            df_experiment[f"y_true_r{run}"] = y_true
            df_experiment[f"y_pred_r{run}"] = y_pred
            df_experiment_config["total_runs"] = run
            save_parquet_with_config(df_experiment, experiment_data_path, df_experiment_config)
            print("All results saved.")


        if len(losses) == n_runs:
            shutil.copyfile(checkpoint_for_rank(experiment_dir, losses, rank=len(losses)//2),
                            os.path.join(experiment_dir, "median-model.pt"))
