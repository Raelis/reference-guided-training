# Optuna execution: Stage 1 architecture tuning and Stage 2 parameter tuning.

import os
import copy
import pickle
import random
import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from core.train import train_model, SchedulerFactory, OptimizerFactory, LossFactory, LossEarlyStopping
from core.experiment_protocol import (
    STUDY_NAME, STAGE1_EPOCHS, STAGE1_PATIENCE, STAGE1_LR,
    FIXED_WEIGHT_DECAY, FIXED_BATCH_SIZE, SEARCH_SPACES,
)
from core.experiment_paths import architecture_dir
from core.experiment_protocol import PROTOCOL_ID, LR_CHOICES
from core.experiment_plan import verify_dependency_manifest
from core.model_factory import model_parameters, instantiate
from core.study_utils import (
    _NullTrial, assert_study_has_result, print_search_space, warn_on_edges,
)

class StudyOperations:
    def load_architecture_params(self):
        #  Stage 1 architecture hyperparameters, when the model has any to search.
        if not self.MyModel.prepare_trial(_NullTrial()):


            return {}

        state_dir = str(architecture_dir(self.MyModel.model_name, self.noise_type))
        state_study_path = os.path.join(state_dir, "optuna_study.db")

        required = {
            "protocol": PROTOCOL_ID,
            "method": "architecture",
            "model": self.MyModel.model_name,
            "noise": self.noise_type,
            "trials": self.n_trials,
            "stage1_lr": STAGE1_LR,
            "lr_choices": list(LR_CHOICES),
        }
        verify_dependency_manifest(state_dir, required)
        if not os.path.exists(state_study_path):
            raise FileNotFoundError(
                f"Stage 1 architecture study not found at {state_study_path}. "
                f"Run `python train.py --stage architecture --model {self.exp_model_name} "
                f"--scenario {('independent' if self.noise_type == 'random1' else 'matched')}` first."
            )

        study = optuna.load_study(study_name=STUDY_NAME, storage=f"sqlite:///{state_study_path}")
        assert_study_has_result(study, state_dir)
        return dict(study.best_params)

    def run_search(self, experiment_dir, dataset_builder, arch_params, method):
        study_path = os.path.join(experiment_dir, "optuna_study.db")

        space = SEARCH_SPACES[method]
        print_search_space(method.upper(), space)

        sampler = TPESampler(seed=self.SEED, n_startup_trials=5, multivariate=True, group=True)
        study = optuna.create_study(
            study_name=STUDY_NAME,


            direction="minimize",
            sampler=sampler,
            storage=f"sqlite:///{study_path}",
            load_if_exists=True,
        )

        valid_states = [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]
        n_completed = len([t for t in study.trials if t.state in valid_states])
        n_remaining = self.n_trials - n_completed

        print(f"Completed trials: {n_completed} | remaining: {n_remaining}")

        if n_remaining > 0:
            train_loader, val_loader, _ = dataset_builder.build_loaders(batch_size=FIXED_BATCH_SIZE)

            def objective(trial):
                torch.manual_seed(self.SEED + trial.number)
                np.random.seed(self.SEED + trial.number)
                random.seed(self.SEED + trial.number)
                torch.cuda.manual_seed_all(self.SEED + trial.number)

                trial_params = self.suggest_params(trial, method)

                model_params = model_parameters(self.MyModel, arch_params)

                model = instantiate(self.MyModel, model_params, self.device)

                loss_factory = LossFactory(loss_name=self.exp_loss_name)
                loss = self.build_loss(loss_factory, trial_params, method)

                scheduler_factory = SchedulerFactory(scheduler_name="cosine", T_max=self.total_epochs)
                optimizer_factory = OptimizerFactory(
                    model, optimizer_name="sgd",
                    lr=self.effective_lr(trial_params), weight_decay=FIXED_WEIGHT_DECAY,
                )

                print(f"Trial {trial.number} params: {trial_params}")

                state = train_model(
                    model, train_loader, val_loader, loss,
                    optimizer_factory, scheduler_factory,
                    epochs=self.total_epochs, device=self.device,
                    params=trial_params, callbacks=[], save_best=False,
                    verbose=True, trial_id=trial.number,
                    val_alt_labels=getattr(dataset_builder, "val_labels_noisy", None),
                )

                # Select the trial on clean validation cross-entropy.
                vals = state.metrics.get("val_second_loss_per_epoch")
                if not vals:
                    raise RuntimeError("Invalid training state: no validation cross-entropy logged")

                vals = np.array(vals, dtype=float)
                if not np.isfinite(vals).all():
                    raise FloatingPointError("Non-finite validation values")

                best = float(vals.min())
                print(f"* Trial {trial.number} best validation CE (clean labels): {best:.6f}")
                return best

            study.optimize(objective, n_trials=n_remaining, catch=(FloatingPointError,), n_jobs=1)

        assert_study_has_result(study, experiment_dir)

        self.selected_trial_number = study.best_trial.number
        print("Best trial:", study.best_trial.number)
        print("Best params:", study.best_trial.params)
        print(f"Best validation CE (clean labels): {study.best_value:.6f}")

        warn_on_edges(study.best_trial.params, space)

        params_path = os.path.join(experiment_dir, "optuna_best-params.pkl")
        with open(params_path, "wb") as f:
            pickle.dump(study.best_trial.params, f)

        return dict(study.best_trial.params)

    def resolve_hyperparameters(self, experiment_dir, dataset_builder, arch_params, method):
        if not self.run_optuna:
            self.selected_trial_number = None
            print(f"[RUN_OPTUNA=False] using fixed hyperparameters: {self.fixed_params}")
            return dict(self.fixed_params)

        if self.hp_source_dir is not None:
            source_study_path = os.path.join(self.hp_source_dir, "optuna_study.db")

            if not os.path.exists(source_study_path):
                raise FileNotFoundError(
                    f"Hyperparameter source study not found at {source_study_path}. "
                    "Run the source condition before reusing hyperparameters."
                )

            source_study = optuna.load_study(
                study_name=STUDY_NAME, storage=f"sqlite:///{source_study_path}"
            )


            assert_study_has_result(source_study, self.hp_source_dir)
            reused = dict(source_study.best_params)
            self.selected_trial_number = source_study.best_trial.number
            print(f"[hyperparameter reuse] from {source_study_path}: {reused}")
            return reused

        return self.run_search(experiment_dir, dataset_builder, arch_params, method)

    def run_architecture_search(self):
        experiment_dir = str(architecture_dir(self.MyModel.model_name, self.noise_type))
        study_path = os.path.join(experiment_dir, "optuna_study.db")

        os.makedirs(experiment_dir, exist_ok=True)
        print("Folder:", experiment_dir)

        if not self.MyModel.prepare_trial(_NullTrial()):
            print(
                f"{self.MyModel.model_name} exposes no architecture hyperparameters "
                f"(prepare_trial returns an empty dict), so Stage 1 is a no-op."
            )
            return

        dataset_builder = self.build_dataset(use_guide=False)
        train_loader, val_loader, _ = dataset_builder.build_loaders(batch_size=FIXED_BATCH_SIZE)

        sampler = TPESampler(seed=self.SEED, n_startup_trials=5, multivariate=True, group=True)
        study = optuna.create_study(
            study_name=STUDY_NAME,

            direction="minimize",
            sampler=sampler,
            storage=f"sqlite:///{study_path}",
            load_if_exists=True,
        )

        valid_states = [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED]
        n_completed = len([t for t in study.trials if t.state in valid_states])
        n_remaining = self.n_trials - n_completed

        print(f"Completed trials: {n_completed} | remaining: {n_remaining}")

        if n_remaining <= 0:
            assert_study_has_result(study, experiment_dir)
            print("Trial budget already reached. Nothing to do.")
            return

        def objective(trial):
            torch.manual_seed(self.SEED + trial.number)
            np.random.seed(self.SEED + trial.number)
            random.seed(self.SEED + trial.number)
            torch.cuda.manual_seed_all(self.SEED + trial.number)

            model_params = self.MyModel.prepare_trial(trial)
            model_params = model_parameters(self.MyModel, model_params)

            model = instantiate(self.MyModel, model_params, self.device)


            loss = LossFactory(loss_name="ce").create()
            scheduler_factory = SchedulerFactory(scheduler_name="cosine", T_max=STAGE1_EPOCHS)
            optimizer_factory = OptimizerFactory(
                model, optimizer_name="sgd", lr=STAGE1_LR, weight_decay=FIXED_WEIGHT_DECAY
            )

            state = train_model(
                model, train_loader, val_loader, loss,
                optimizer_factory, scheduler_factory,
                epochs=STAGE1_EPOCHS, device=self.device,
                params=model_params, callbacks=[], save_best=False,
                verbose=True, trial_id=trial.number,
                stop_criteria=[LossEarlyStopping(STAGE1_PATIENCE)],
                val_alt_labels=getattr(dataset_builder, "val_labels_noisy", None),
            )

            vals = state.metrics.get("val_second_loss_per_epoch")
            if not vals:
                raise RuntimeError("Invalid training state: no validation cross-entropy logged")

            vals = np.array(vals, dtype=float)
            if not np.isfinite(vals).all():
                raise FloatingPointError("Non-finite validation values")

            return float(vals.min())

        study.optimize(objective, n_trials=n_remaining, catch=(FloatingPointError,), n_jobs=1)

        assert_study_has_result(study, experiment_dir)

        print("Best trial:", study.best_trial.number)
        print("Best architecture params:", study.best_trial.params)
        print(f"Best validation CE (clean labels): {study.best_value:.6f}")

        params_path = os.path.join(experiment_dir, "optuna_best-params.pkl")
        with open(params_path, "wb") as f:
            pickle.dump(study.best_trial.params, f)

