# CIFAR experiment coordinator: configuration, data and method dispatch.
# The numerical training loop is executed in core.train.trainer.

import random
import os
import numpy as np
import torch
from core.data import DatasetBuilder
from core.model_factory import import_model_class
from core.experiment_paths import guide_dir
from core.experiment_plan import verify_dependency_manifest
from core.experiment_protocol import PROTOCOL_ID, LR_CHOICES, STAGE1_LR, STAGE1_EPOCHS
from core.experiment_protocol import ERM_SEARCH_SPACE, RG_SEARCH_SPACE, SEARCH_SPACES, MODE_TO_METHOD, GUIDED_METHODS
from core.study_utils import _NullTrial
from core.experiment_method import method_spec
from core.experiment_studies import StudyOperations
from core.experiment_fit import FitOperations

from core.study_utils import print_search_space, warn_on_edges, assert_study_has_result
from core.evaluation_metrics import evaluate_on_test

class CIFAR10NExperiment(StudyOperations, FitOperations):
    def __init__(self, exp_model_name, exp_loss_name, noise_type, exp_guide_model,
                 guide_noise_type, total_epochs, n_trials, seed=42, run_optuna=True,
                 fixed_params=None, hp_source_dir=None, tag=None):

        self.exp_model_name = exp_model_name
        self.exp_loss_name = exp_loss_name
        self.noise_type = noise_type
        self.exp_guide_model = exp_guide_model
        self.guide_noise_type = guide_noise_type
        self.total_epochs = total_epochs
        self.n_trials = n_trials
        self.SEED = seed
        self.run_optuna = run_optuna
        self.fixed_params = fixed_params or {}
        self.hp_source_dir = hp_source_dir
        self.tag = tag

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.MyModel = import_model_class(exp_model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA available? {torch.cuda.is_available()}")

    def build_dataset(self, use_guide):
        if use_guide and self.exp_guide_model == "ResNet56":
            guide_directory = str(guide_dir(self.guide_noise_type))
            required = {
                "protocol": PROTOCOL_ID, "method": "guide", "model": "ResNet56",
                "noise": self.guide_noise_type, "trials": self.n_trials,
                "epochs": self.total_epochs, "lr_choices": list(LR_CHOICES),
                "stage1_lr": STAGE1_LR, "stage1_epochs": STAGE1_EPOCHS,
            }
            if hasattr(self, "requested_runs"):
                required["runs"] = self.requested_runs
            verify_dependency_manifest(guide_directory, required)
        baseline_method = self.exp_guide_model if use_guide else "no"
        guide_noise_type = self.guide_noise_type if use_guide else "clean"

        dataset_builder = DatasetBuilder(
            noise_type=self.noise_type,
            guide_noise_type=guide_noise_type,
            baseline_method=baseline_method,
            seed=self.SEED,
            device=self.device,
        )
        dataset_builder.build()
        return dataset_builder

    def suggest_params(self, trial, method):
        return method_spec(method).sample_parameters(trial)

    def effective_lr(self, params):
        return method_spec(self.method_from_loss()).learning_rate(params)

    def method_from_loss(self):
        try:
            return {"ce": "erm", "rg": "rg"}[self.exp_loss_name]
        except KeyError:
            raise ValueError(f"Unknown CIFAR loss: {self.exp_loss_name}") from None

    def build_loss(self, loss_factory, params, method):
        return method_spec(method).make_loss(loss_factory, params)
