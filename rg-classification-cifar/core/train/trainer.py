import os
import time
import torch

from .state import TrainingState
from .callbacks import CallbackManager
from .metrics_logger import MetricsLogger
from typing import Optional
from .early_stopping import StopManager
import numpy as np

import torch.nn.functional as F


def save_model(state, path):
    checkpoint = {
        "model_class": state.model.__class__,
        "model_params": state.params,
        "state_dict": state.model.state_dict(),
        "metrics": state.metrics.as_dict()
    }
    torch.save(checkpoint, path)
    time.sleep(0.01)

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
    trial_id = None,
    val_alt_labels = None,
):
    """Train with method-specific loss; log noisy-label validation diagnostics separately.

    Alternative labels follow validation loader order and never affect selection.
    """

    model.to(device)
    if init_weights:
        init_seed = (seed if seed is not None else
                     int(np.random.default_rng().integers(low=0, high=(2**32) - 1, endpoint=True)))
        init_weights(model, seed=init_seed)

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


        model.train()
        train_loss_sum = 0.0
        train_second_loss_sum = 0.0
        n_samples = 0
        train_correct = 0
        train_total = 0


        for x, y, y_baseline in train_data:
            state.step += 1
            x = x.to(device)
            y = y.to(device).long()
            y_baseline = y_baseline.to(device)

            optimizer.zero_grad()
            outputs = model(x)


            loss = loss_fn(outputs, y, y_baseline=y_baseline, epoch=epoch, max_epochs=epochs)
            loss.backward()
            optimizer.step()


            bs = y.numel()

            train_loss_sum += loss.item() * bs
            train_second_loss_sum += torch.nn.functional.cross_entropy(outputs, y, reduction='none').sum().item()
            n_samples += bs

            train_correct += (outputs.argmax(dim=1) == y).sum().item()
            train_total += bs


        train_loss = train_loss_sum / max(1, n_samples)
        train_second_loss = train_second_loss_sum / max(1, n_samples)
        train_acc = train_correct / max(1, train_total)


        state.metrics.log(
            train_loss_per_epoch=train_loss,
            train_second_loss_per_epoch=train_second_loss,
            train_acc_per_epoch=train_acc,

        )

        if verbose:
            if trial_id is not None:
                print(f"Trial {trial_id} | Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Train CE: {train_second_loss:.6f} | Train Acc: {train_acc:.4f}", flush=True)
            else:
                print(f"Epoch {epoch+1} | Train Loss: {train_loss:.6f} | Train CE: {train_second_loss:.6f} | Train Acc: {train_acc:.4f}", flush=True)


        model.eval()
        val_loss_sum = 0.0
        val_cross_entropy_sum = 0.0
        n_samples = 0
        val_correct = 0
        val_total = 0


        # Alternative-label metrics never affect checkpoint selection.
        alt_logits = [] if val_alt_labels is not None else None

        with torch.no_grad():
            for x_val, y_val, y_val_baseline in val_data:
                x_val = x_val.to(device)
                y_val = y_val.to(device).long()
                y_val_baseline = y_val_baseline.to(device)

                preds = model(x_val)
                loss = loss_fn(preds, y_val, y_baseline=y_val_baseline, epoch=epoch, max_epochs=epochs)


                bs = y_val.numel()


                val_loss_sum += loss.item() * bs
                val_cross_entropy_sum += torch.nn.functional.cross_entropy(preds, y_val, reduction='none').sum().item()
                n_samples += bs

                val_correct += (preds.argmax(dim=1) == y_val).sum().item()
                val_total += bs

                if alt_logits is not None:
                    alt_logits.append(preds.detach().cpu())


        val_loss = val_loss_sum / max(1, n_samples)
        val_second_loss = (val_cross_entropy_sum / max(1, n_samples))
        val_acc = val_correct / max(1, val_total)


        val_alt_ce = None
        val_alt_acc = None
        if alt_logits is not None:
            logits = torch.cat(alt_logits, dim=0)
            alt = val_alt_labels[: logits.shape[0]].to(logits.device).long()
            val_alt_ce = F.cross_entropy(logits, alt, reduction="mean").item()
            val_alt_acc = (logits.argmax(dim=1) == alt).float().mean().item()

        state.metrics.log(
            val_loss_per_epoch=val_loss,
            val_second_loss_per_epoch=val_second_loss,
            val_acc_per_epoch=val_acc,
            val_alt_ce_per_epoch=val_alt_ce,
            val_alt_acc_per_epoch=val_alt_acc,

        )

        if verbose:
            alt_txt = ""
            if val_alt_ce is not None:
                alt_txt = f" | Alt CE: {val_alt_ce:.6f} | Alt Acc: {val_alt_acc:.4f}"
            if trial_id is not None:
                print(f"Trial {trial_id} | Epoch {epoch+1} | Validation Loss: {val_loss:.6f} | Validation CE: {val_second_loss:.6f} | Validation Acc: {val_acc:.4f}{alt_txt}", flush=True)
            else:
                print(f"Epoch {epoch+1} | Validation Loss: {val_loss:.6f} | Validation CE: {val_second_loss:.6f} | Validation Acc: {val_acc:.4f}{alt_txt}", flush=True)


        if scheduler is not None:
            try:
                if scheduler_factory.scheduler_name == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            except Exception:
                raise RuntimeError("Failed to update the learning-rate scheduler.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_best and save_dir and save_name:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, save_name)
                save_model(state, save_path)
                print(f"* Novo melhor modelo salvo em {save_path} com val_loss {val_loss:.6f} *", flush=True)


        cb_manager.on_epoch_end(state)

        if stop_manager and stop_manager.step(state):
            if verbose:
                print(f"[EARLY STOP] reason={stop_manager.stop_reason} | epoch={epoch+1}", flush=True)
            break

    cb_manager.on_train_end(state)
    return state


