from abc import ABC
from typing import Type
from .state import TrainingState
import random
import colorsys
import matplotlib.colors as mcolors
from .plot_manager import PlotManager

import os
import torch

def save_model(state, path):
    checkpoint = {
        "model_class": state.model.__class__,
        "model_params": state.params,
        "state_dict": state.model.state_dict(),
        "metrics": state.metrics.as_dict()
    }
    torch.save(checkpoint, path)

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

class PlotCallback(Callback):
    priority = 0

    def __init__(self, trial_id: int, plot_manager: PlotManager):
        self.trial_id = trial_id
        self.plot_manager = plot_manager
        self.color = self._get_trial_color(trial_id)
        self._initialized = False

    def _get_trial_color(self, trial_id: int) -> str:
        random.seed(trial_id)
        hue = random.random()
        lightness = 0.5 + random.random() * 0.2
        saturation = 0.6 + random.random() * 0.3
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        return mcolors.to_hex(rgb)

    def on_train_begin(self, state):
        self.plot_manager.init_trial(self.trial_id, self.color)
        self._initialized = True

    def on_epoch_end(self, state: TrainingState):
        if not self._initialized:
            return

        self.plot_manager.update_epoch(
            trial_id=self.trial_id,
            metrics=state.metrics,
        )
        self.plot_manager.show()


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
            print(f"* New best model saved to {self.path} with val_loss {current:.6f} *")

