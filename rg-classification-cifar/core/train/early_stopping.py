# Early stopping based on validation loss for architecture search.
from abc import ABC, abstractmethod


class StopCriterion(ABC):
    def __init__(self):
        self.should_stop = False
        self.stop_reason = None

    @abstractmethod
    def step(self, state) -> bool:
        pass


class LossEarlyStopping(StopCriterion):
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


class StopManager:
    def __init__(self, criteria):
        self.criteria = criteria
        self.stop_reason = None

    def step(self, state) -> bool:
        for criterion in self.criteria:
            if criterion.step(state):
                self.stop_reason = criterion.stop_reason
                return True
        return False
