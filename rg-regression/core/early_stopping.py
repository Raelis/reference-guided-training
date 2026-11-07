from abc import ABC, abstractmethod
import numpy as np
from typing import Sequence

class StopCriterion(ABC):
    
    # Abstract base class for all stopping criteria.

    # Each criterion:
    # - receives a training state - TrainingState
    # - decides whether training should stop
    # - stores a human-readable stop_reason

    # This abstraction enables composition of multiple independent stopping rules.
    
    def __init__(self):
        self.should_stop = False
        self.stop_reason = None

    @abstractmethod
    def step(self, state) -> bool:
        # Evaluate stopping condition.
        pass


class LossEarlyStopping(StopCriterion):
    # Early stopping based on validation loss stagnation.

    # Stops training when validation loss does not improve by at least `min_delta`
    # for `patience` consecutive epochs.
    def __init__(self, patience, min_delta=0.0, pre_warming=0, best=None):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.pre_warming = pre_warming
        self.best = best if best is not None else float("inf")
        self.bad_epochs = 0

    def step(self, state) -> bool:
        epochs = len(state.metrics.get("val_loss_per_epoch"))
        loss = state.metrics.get("val_loss_per_epoch")[-1] if epochs > self.pre_warming else float("inf")

        if loss < self.best - self.min_delta:
            self.best = loss
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True
            self.stop_reason = "no_improvement"
            return True

        return False


class StdCollapseStopping(StopCriterion):
    # Detects model collapse based on prediction standard deviation.

    # If the standard deviation of validation predictions approaches zero,
    # the model likely collapsed to a constant output.
    def __init__(self, threshold=1e-6, patience=3):
        super().__init__()
        self.threshold = threshold
        self.patience = patience
        self.low_std_epochs = 0

    def step(self, state) -> bool:
        std = np.min(state.metrics.get("val_std_per_epoch"))

        if std <= self.threshold:
            self.low_std_epochs += 1
            if self.low_std_epochs >= self.patience:
                self.should_stop = True
                self.stop_reason = "std_collapse"
                return True
        else:
            self.low_std_epochs = 0

        return False


class StopManager:
    # Aggregates multiple StopCriterion objects.

    # Training stops if ANY criterion triggers.

    def __init__(self, criteria: Sequence):
        self.criteria = criteria
        self.stop_reason = None

    def step(self, state) -> bool:
        # Evaluate all stopping criteria.

        for criterion in self.criteria:
            if criterion.step(state):
                self.stop_reason = criterion.stop_reason
                return True
        return False

