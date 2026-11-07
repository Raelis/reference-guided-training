import os
import torch
from tqdm import tqdm

from .state import TrainingState
from .callbacks import CallbackManager
from .metrics_logger import MetricsLogger
from typing import Optional, Type
from .early_stopping import StopManager
import numpy as np

def save_model(state, path):
    checkpoint = {
        "model_class": state.model.__class__,
        "model_params": state.params,
        "state_dict": state.model.state_dict(),
        "metrics": state.metrics.as_dict()
    }
    torch.save(checkpoint, path)

def train_model(
    model,
    train_data,
    val_data,
    loss_fn,
    optimizer_factory,
    scheduler_factory,
    epochs=20,
    device="cuda",
    verbose=True,
    params = None,
    callbacks: Optional[CallbackManager] = None,
    save_best: bool = False,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
    init_weights=None,
    stop_criteria: Optional[list] = None,
    seed = None,
):

    
    
    model.to(device)
    if init_weights:
        if seed is None:
            rng = np.random.default_rng()
            SEED = int(rng.integers(low=0, high=(2**32) - 1, endpoint=True))
        else:
            SEED = seed
        init_weights(model, seed=SEED)



    optimizer = optimizer_factory.create()
    scheduler = scheduler_factory.create(optimizer)

    state = TrainingState(
        epoch=0,
        step=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        params=params,
        metrics=MetricsLogger(),
    )

    cb_manager = CallbackManager(callbacks)
    cb_manager.on_train_begin(state)
    best_val_loss = float("inf")

    stop_manager = StopManager(stop_criteria) if stop_criteria else None

    for epoch in range(epochs):
        state.epoch = epoch
        cb_manager.on_epoch_begin(state)

        # -------------------
        # Train
        # -------------------
        model.train()
        train_loss_sum = 0.0
        train_mse_sum = 0.0
        n_samples = 0

        for x, y, y_baseline in tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}"):
            state.step += 1
            x = x.to(device)
            y = y.to(device).float()
            y_baseline = y_baseline.to(device).float()

            optimizer.zero_grad()
            outputs = model(x)

            loss = loss_fn(outputs, y, y_baseline=y_baseline, epoch=epoch, max_epochs=epochs)
            loss.backward()
            optimizer.step()

            bs = y.numel()

            train_loss_sum += loss.item() * bs
            train_mse_sum += torch.mean((outputs - y) ** 2).item() * bs
            n_samples += bs

        # global metrics
        train_loss = train_loss_sum / max(1, n_samples)
        train_mse = train_mse_sum / max(1, n_samples)

        state.metrics.log(
            train_loss_per_epoch=train_loss,
            train_mse_per_epoch=train_mse,
        )

        if verbose:
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Train MSE: {train_mse:.6f}")

        # -------------------
        # Validation
        # -------------------
        model.eval()
        val_loss_sum = 0.0
        val_mse_sum = 0.0
        n_samples = 0

        with torch.no_grad():
            for x_val, y_val, y_val_baseline in val_data:
                x_val = x_val.to(device)
                y_val = y_val.to(device).float()
                y_val_baseline = y_val_baseline.to(device).float()

                preds = model(x_val)
                loss = loss_fn(preds, y_val, y_baseline=y_val_baseline, epoch=epoch, max_epochs=epochs)

                bs = y_val.numel()

                val_loss_sum += loss.item() * bs
                val_mse_sum += torch.mean((preds - y_val) ** 2).item() * bs
                n_samples += bs

        # global val metrics
        val_loss = val_loss_sum / max(1, n_samples)
        val_mse = (val_mse_sum / max(1, n_samples))

        state.metrics.log(
            val_loss_per_epoch=val_loss,
            val_mse_per_epoch=val_mse,
        )

        if verbose:
            print(f"Epoch {epoch+1} | Validation Loss: {val_loss:.6f} | Validation MSE: {val_mse:.6f}")

        # -------------------
        # Scheduler & Save best
        # -------------------
        if scheduler is not None:
            try:
                scheduler.step(val_loss)
            except Exception:
                raise RuntimeError("Error updating the scheduler. Verify that the parameters are correct.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_best and save_dir and save_name:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, save_name)                
                save_model(state, save_path)
                print(f"* New best model saved at {save_path} with val_loss {val_loss:.6f} *")

        cb_manager.on_epoch_end(state)

        if stop_manager and stop_manager.step(state):
            if verbose:
                print(f"[EARLY STOP] reason={stop_manager.stop_reason} | epoch={epoch+1}")
            break

    cb_manager.on_train_end(state)
    return state