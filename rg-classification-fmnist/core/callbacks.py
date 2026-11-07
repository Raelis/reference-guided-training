from abc import ABC
from .state import TrainingState
import os


from .utils import save_model, end_train_save_model


class Callback(ABC):
    priority = 0

    def on_train_begin(self, state: TrainingState): pass
    def on_train_end(self, state: TrainingState): pass

    def on_epoch_begin(self, state: TrainingState): pass
    def on_epoch_end(self, state: TrainingState): pass

    def on_batch_begin(self, batch_idx: int, state: TrainingState): pass
    def on_batch_end(self, batch_idx: int, loss: float, state: TrainingState): pass


class CallbackManager:
    def __init__(self, callbacks=None):
        self.callbacks = sorted(
            callbacks or [],
            key=lambda cb: getattr(cb, "priority", 0)
        )

    def _call(self, hook, *args):
        for cb in self.callbacks:
            fn = getattr(cb, hook, None)
            if fn is None:
                continue
            try:
                fn(*args)
            except Exception as e:
                raise RuntimeError(
                    f"Callback {cb.__class__.__name__}.{hook} failed"
                ) from e

    def on_train_begin(self, state):
        self._call("on_train_begin", state)

    def on_epoch_begin(self, state):
        self._call("on_epoch_begin", state)

    def on_epoch_end(self, state):
        self._call("on_epoch_end", state)

    def on_batch_begin(self, batch_idx, state):
        self._call("on_batch_begin", batch_idx, state)

    def on_batch_end(self, batch_idx, loss, state):
        self._call("on_batch_end", batch_idx, loss, state)

    def on_train_end(self, state):
        self._call("on_train_end", state)

class SaveBestValLossCallback(Callback):
    priority = 0

    def __init__(
        self,
        save_dir: str,
        save_name: str,
        best_init: float,
        delta: float = 0.0,
    ):
        self.best = best_init
        self.delta = delta

        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, save_name)

    def on_epoch_end(self, state: TrainingState):
        val_losses = state.metrics.get("val_loss_per_epoch")

        if not val_losses:
            return

        current = val_losses[-1]

        if current < self.best - self.delta:
            self.best = current
            save_model(state, self.path)
            print(f"* New best model saved in {self.path} with val_loss {current:.6f} *")
    
    def on_train_end(self, state):
        end_train_save_model(state, self.path)    
        print(f"Model last checkpoint saved in {self.path} *")

