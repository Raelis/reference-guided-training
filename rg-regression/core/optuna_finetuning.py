from .early_stopping import LossEarlyStopping
from .callbacks import PlotCallback
from .trainer import train_model
import torch
import numpy as np
from .schedulers import SchedulerFactory
from .optimizers import OptimizerFactory
import optuna
import os
import pickle
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from copy import deepcopy

def finetuning(experiment_dir, dataset_builder, MyModel, loss_factory, plot_manager = None, max_trials = 40, epochs=20, patience=8, SEED=42, window_size = 240, device = "cuda" if torch.cuda.is_available() else "cpu"):

    def optuna_objetive(trial):
        try:
            batch_size = trial.suggest_categorical("batch_size", [256, 512, 768, 1024])
            train_loader, val_loader, _ = dataset_builder.build_loaders(batch_size=batch_size)

            # Learning rate
            # lr_values = np.logspace(-6, -3, num=16).tolist()
            lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True) # lr = trial.suggest_categorical("lr", lr_values)

            if loss_factory.loss_name in ["baseline_mse", "baseline_mae", "baseline_mse_softplus"]:
                lamb_values = np.linspace(0.25 , 4.0, 16).tolist()
                lamb = trial.suggest_categorical("lamb", lamb_values)

                margin_values = np.linspace(0.9, 1.1, 4).tolist()
                margin = trial.suggest_categorical("margin", margin_values)

                delta_values = np.logspace(-5, -2, num=8).tolist()
                delta = trial.suggest_categorical("delta", delta_values)
                loss = loss_factory.create(lamb=lamb, delta=delta, margin=margin)
            else:
                loss = loss_factory.create()

            wd_values = np.logspace(-6, -2, num=8).tolist()
            weight_decay = trial.suggest_categorical("weight_decay", wd_values)

            model_params = MyModel.prepare_trial(trial)
            model_params["num_features"] = window_size
            model_params["output_dim"] = dataset_builder.horizon
            model = MyModel(**model_params).to(device)
            model.model_postprocess(model)

            scheduler_factory = SchedulerFactory(scheduler_name="plateau")
            optimizer_factory = OptimizerFactory(model, optimizer_name="adamw", lr=lr, weight_decay=weight_decay)

            callback = []
            if plot_manager is not None:
                plot_cb = PlotCallback(trial.number, plot_manager)
                plot_manager.add_trial_params(trial.params)
                callback.append(plot_cb)

            state = train_model(
                model, train_loader, val_loader, loss, optimizer_factory, scheduler_factory, params=model_params,
                epochs=epochs, device=device, stop_criteria=[LossEarlyStopping(patience, pre_warming=0)], callbacks=callback
            )

            if state is None or "val_mse_per_epoch" not in state.metrics.as_dict() or len(state.metrics.get("val_mse_per_epoch")) == 0:
                raise RuntimeError("Invalid training state")

            vals = np.array(state.metrics.get("val_mse_per_epoch"), dtype=float)
            if not np.isfinite(vals).all():
                raise FloatingPointError("Non-finite validation values")
            
            if trial.should_prune():
                raise optuna.TrialPruned()

            best = vals.min()
            print(f"* Best loss in trial was {best}")
            return best

        except torch.cuda.OutOfMemoryError:
            print(f"[Trial {trial.number}] OOM: pruning trial automatically.")
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()


    # ==============================================
    # EXECUÇÃO DO ESTUDO OPTUNA
    # ==============================================
    sampler = TPESampler(
        seed=SEED,
        n_startup_trials=5,   # only explores randomly for the first 5 trials
        multivariate=True,
        group=True
    )

    pruner = MedianPruner(
        n_startup_trials=5,   # only starts pruning after exploration
        n_warmup_steps=5,     # allows the model to learn something
        interval_steps=1
    )

    # Create or load the study
    study_path = os.path.join(experiment_dir, "optuna_study.db")
    study = optuna.create_study(
        study_name="experiment_with_indicators",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{study_path}",  # saves locally
        load_if_exists=True
    )

    # ==============================================
    # COUNT COMPLETED TRIALS
    # ==============================================
    VALID_STATES = [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED
    ]

    completed_trials = [t for t in study.trials if t.state in VALID_STATES]

    n_completed = len(completed_trials)
    n_remaining = max_trials - n_completed

    print(f"Trials completos: {n_completed}")
    print(f"Trials restantes: {n_remaining}")

    if n_remaining <= 0:
        print("Number of trials already reached. Nothing to do.")
    else:
        study.optimize(
            lambda trial: optuna_objetive(trial),
            n_trials=n_remaining,
            catch=(RuntimeError, FloatingPointError)
        )
        # After training, you can save the best result explicitly:
        print("Best trial:", study.best_trial.number)
        print("Best parameters:", study.best_trial.params)
        print(f"Best Loss: {study.best_value:.4f}")

        # Save the best parameters in a pickle file
        params_path = os.path.join(experiment_dir, "optuna_best-params.pkl")
        with open(params_path, "wb") as f:
            pickle.dump(study.best_trial.params, f)

        print(f"Best parameters saved to '{params_path}'")


def finetuning_only_delta_lambda(experiment_dir, dataset_builder, MyModel, loss_factory, best_params, plot_manager = None, checkpoint = None, max_trials = 40, epochs=20, patience=8, SEED=42, window_size = 240, device = "cuda" if torch.cuda.is_available() else "cpu"):

    def optuna_objetive(trial):
        try:
            if loss_factory.loss_name != "baseline-mse":
                raise ValueError(
                    f"Loss '{loss_factory.loss_name}' is not supported by this fine-tuning function."
                )
            
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64]) #best_params["batch_size"] # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128]) # best_params["batch_size"]
            train_loader, val_loader, _ = dataset_builder.build_loaders(batch_size=batch_size)

            # Learning rate
            # lr = best_params["lr"]
            # lr_values = np.logspace(-6, -3, num=16).tolist()
            lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True) # lr = trial.suggest_categorical("lr", lr_values)

            lamb = trial.suggest_float("lamb", 0.25 , 10.0)
            margin = trial.suggest_float("margin", 0.9 , 1.1)
            delta = trial.suggest_float("delta", 1e-5, 1e-2, log=True)

            loss = loss_factory.create(lamb=lamb, delta=delta, margin=margin)

            # weight_decay = best_params["weight_decay"]
            wd_values = np.logspace(-6, -2, num=8).tolist()
            weight_decay = trial.suggest_categorical("weight_decay", wd_values)

            best_params_copy = deepcopy(best_params)
            best_params_copy, other_params = MyModel.prepare_model_params(best_params_copy)
            best_params_copy["num_features"] = window_size
            best_params_copy["output_dim"] = dataset_builder.horizon
            model = MyModel(**best_params_copy).to(device)
            model.model_postprocess(model)
            if checkpoint != None:
                model.load_state_dict(checkpoint)
            
            
            scheduler_factory = SchedulerFactory(scheduler_name="plateau")
            optimizer_factory = OptimizerFactory(model, optimizer_name="adamw", lr=lr, weight_decay=weight_decay)

            callback = []
            if plot_manager is not None:
                plot_cb = PlotCallback(trial.number, plot_manager)
                plot_manager.add_trial_params(trial.params)
                callback.append(plot_cb)

            state = train_model(
                model, train_loader, val_loader, loss, optimizer_factory, scheduler_factory,
                epochs=epochs, device=device, stop_criteria=[LossEarlyStopping(patience, pre_warming=0, min_delta=1e-4)], callbacks=callback
            )

            if state is None or "val_mse_per_epoch" not in state.metrics.as_dict() or len(state.metrics.get("val_mse_per_epoch")) == 0:
                raise RuntimeError("Invalid training state")

            vals = np.array(state.metrics.get("val_mse_per_epoch"), dtype=float)
            if not np.isfinite(vals).all():
                raise FloatingPointError("Non-finite validation values")
            
            if trial.should_prune():
                raise optuna.TrialPruned()

            best = vals.min()
            print(f"* best loss in trial was {best}")
            return best

        except torch.cuda.OutOfMemoryError:
            print(f"[Trial {trial.number}] OOM: pruning trial automatically.")
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

    # ==============================================
    # EXECUTION OF THE OPTUNA STUDY
    # ==============================================
    sampler = TPESampler(
        seed=SEED,
        n_startup_trials=5,
        multivariate=True,
        group=True
    )

    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
        interval_steps=1
    )

    study_path = os.path.join(experiment_dir, "optuna_study.db")

    study = optuna.create_study(
        study_name="experiment_with_indicators",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{study_path}",
        load_if_exists=True
    )

    # ==============================================
    # COUNT COMPLETED TRIALS
    # ==============================================
    VALID_STATES = [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED
    ]

    completed_trials = [t for t in study.trials if t.state in VALID_STATES]

    n_completed = len(completed_trials)
    n_remaining = max_trials - n_completed

    print(f"Completed trials: {n_completed}")
    print(f"Remaining trials: {n_remaining}")

    if n_remaining <= 0:
        print("Number of trials already reached. Nothing to do.")
    else:
        study.optimize(
            lambda trial: optuna_objetive(trial),
            n_trials=n_remaining,
            catch=(RuntimeError, FloatingPointError)
        )
        # After training, you can save the best result explicitly:
        print("Best trial:", study.best_trial.number)
        print("Best parameters:", study.best_trial.params)
        print(f"Best Loss: {study.best_value:.4f}")

        # Save the best parameters in a pickle file
        params_path = os.path.join(experiment_dir, "optuna_best-params.pkl")
        with open(params_path, "wb") as f:
            best_params.update(study.best_trial.params)  # Updates the dictionary with the new values of lambda and delta
            pickle.dump(best_params, f)

        print(f"Best parameters saved to '{params_path}'")
    

def finetuning_baseline(experiment_dir, dataset_builder, MyModel, loss_factory, best_params, plot_manager = None, checkpoint = None, max_trials = 40, epochs=20, patience=8, SEED=42, window_size = 240, device = "cuda" if torch.cuda.is_available() else "cpu"):

    def optuna_objetive(trial):
        try:
            if loss_factory.loss_name in ["baseline_mse", "baseline_mae", "baseline_mse_softplus"]:
                raise ValueError(f"Loss '{loss_factory.loss_name}' is not compatible with this fine-tuning function. Use 'finetuning' to optimize all hyperparameters.")
            
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64]) #best_params["batch_size"] # batch_size = trial.suggest_categorical("batch_size", [32, 64, 128]) # best_params["batch_size"]
            train_loader, val_loader, _ = dataset_builder.build_loaders(batch_size=batch_size)

            # Learning rate
            # lr = best_params["lr"]
            # lr_values = np.logspace(-6, -3, num=16).tolist()
            lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True) # lr = trial.suggest_categorical("lr", lr_values)

            # lamb = trial.suggest_float("lamb", 0.25 , 10.0)
            # margin = trial.suggest_float("margin", 0.9 , 1.1)
            # delta = trial.suggest_float("delta", 1e-5, 1e-2, log=True)

            loss = loss_factory.create()

            # weight_decay = best_params["weight_decay"]
            wd_values = np.logspace(-6, -2, num=8).tolist()
            weight_decay = trial.suggest_categorical("weight_decay", wd_values)

            best_params_copy = deepcopy(best_params)
            best_params_copy, other_params = MyModel.prepare_model_params(best_params_copy)
            best_params_copy["num_features"] = window_size
            best_params_copy["output_dim"] = dataset_builder.horizon
            model = MyModel(**best_params_copy).to(device)
            if checkpoint != None:
                model.load_state_dict(checkpoint)
            model.model_postprocess(model)
            
            scheduler_factory = SchedulerFactory(scheduler_name="plateau")
            optimizer_factory = OptimizerFactory(model, optimizer_name="adamw", lr=lr, weight_decay=weight_decay)

            callback = []
            if plot_manager is not None:
                plot_cb = PlotCallback(trial.number, plot_manager)
                plot_manager.add_trial_params(trial.params)
                callback.append(plot_cb)

            state = train_model(
                model, train_loader, val_loader, loss, optimizer_factory, scheduler_factory,
                epochs=epochs, device=device, stop_criteria=[LossEarlyStopping(patience, pre_warming=0, min_delta=1e-4)], callbacks=callback
            )

            if state is None or "val_mse_per_epoch" not in state.metrics.as_dict() or len(state.metrics.get("val_mse_per_epoch")) == 0:
                raise RuntimeError("Invalid training state")

            vals = np.array(state.metrics.get("val_mse_per_epoch"), dtype=float)
            if not np.isfinite(vals).all():
                raise FloatingPointError("Non-finite validation values")
            
            if trial.should_prune():
                raise optuna.TrialPruned()

            best = vals.min()
            print(f"* best trial loss: {best}")
            return best

        except torch.cuda.OutOfMemoryError:
            print(f"[Trial {trial.number}] OOM: pruning trial automatically.")
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()


    # ==============================================
    # EXECUTION OF THE OPTUNA STUDY
    # ==============================================
    sampler = TPESampler(
        seed=SEED,
        n_startup_trials=5,   # ~10% of pure exploration
        multivariate=True,
        group=True
    )

    pruner = MedianPruner(
        n_startup_trials=5,   # start pruning only after exploration
        n_warmup_steps=5,     # allows the model to learn something
        interval_steps=1
    )

    # Create or load the study
    study_path = os.path.join(experiment_dir, "optuna_study.db")
    study = optuna.create_study(
        study_name="experiment_with_indicators",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{study_path}",  # saves locally
        load_if_exists=True
    )
    
    # ==============================================
    # COUNT HOW MANY COMPLETE TRIALS ALREADY EXISTS
    # ==============================================
    VALID_STATES = [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED
    ]

    completed_trials = [t for t in study.trials if t.state in VALID_STATES]

    n_completed = len(completed_trials)
    n_remaining = max_trials - n_completed

    print(f"Complete trials: {n_completed}")
    print(f"Remaining trials: {n_remaining}")

    if n_remaining <= 0:
        print("Number of trials already reached. Nothing to do.")
    else:
        study.optimize(
            lambda trial: optuna_objetive(trial),
            n_trials=n_remaining,
            catch=(RuntimeError, FloatingPointError)
        )
        # After training, you can save the best result explicitly:
        print("Best trial:", study.best_trial.number)
        print("Best parameters:", study.best_trial.params)
        print(f"Best Loss: {study.best_value:.4f}")

        # Save the best parameters in a pickle file
        params_path = os.path.join(experiment_dir, "optuna_best-params.pkl")
        with open(params_path, "wb") as f:
            best_params.update(study.best_trial.params)  # Updates the dictionary with the new values of lr and weight_decay
            pickle.dump(best_params, f)

        print(f"Best parameters saved in '{params_path}'")