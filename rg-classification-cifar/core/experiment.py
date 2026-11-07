# Public CIFAR experiment interface.

# Mirrors the Fashion-MNIST public Experiment entry point.

from .experiment_runner import CIFAR10NExperiment
from .experiment_method import MethodSpec, method_spec

__all__ = ["CIFAR10NExperiment", "MethodSpec", "method_spec"]
