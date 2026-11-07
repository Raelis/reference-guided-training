from abc import ABC, abstractmethod
import numpy as np
from typing import Sequence

class StopCriterion(ABC):
    """Base interface for training stop criteria."""
    def __init__(self):
        self.should_stop = False
        self.stop_reason = None

    @abstractmethod
    def step(self, state) -> bool:
        """
        Evaluate stopping condition.

        Parameters
        ----------
        state : TrainingState

        Returns
        -------
        bool
            True if this criterion triggers stopping.
        """
        pass

class LossEarlyStopping(StopCriterion):
    """Stop after `patience` epochs without sufficient validation improvement."""
    def __init__(self, patience, min_delta=0.0, pre_warming=0, best=None,
                 metric="val_loss_per_epoch"):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.pre_warming = pre_warming
        self.best = best if best is not None else float("inf")
        self.bad_epochs = 0
        self.metric = metric

    def step(self, state) -> bool:
        values = state.metrics.get(self.metric)
        epochs = len(values)
        loss = values[-1] if epochs > self.pre_warming else float("inf")

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
    """Stop when validation prediction variance stays below a threshold."""
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
    """
    Aggregates multiple StopCriterion objects.

    Training stops if ANY criterion triggers.
    """

    def __init__(self, criteria: Sequence):
        self.criteria = criteria
        self.stop_reason = None

    def step(self, state) -> bool:
        """
        Evaluate all stopping criteria.

        Parameters
        ---------- 
        state : TrainingState
        """
        for criterion in self.criteria:
            if criterion.step(state):
                self.stop_reason = criterion.stop_reason
                return True
        return False

