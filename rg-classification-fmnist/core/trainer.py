import os
import torch
from typing import Optional
import numpy as np

from .early_stopping import StopManager
from .state import TrainingState
from .callbacks import CallbackManager
from .metrics_logger import MetricsLogger
from .utils import save_model


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
    state = None,
    callbacks: Optional[CallbackManager] = None,
    save_best: bool = False,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
    init_weights=None,
    stop_criteria: Optional[list] = None,
    seed = None,
    trial_id = None,
):

    model.to(device)
    if init_weights:
        init_seed = (seed if seed is not None else
                     int(np.random.default_rng().integers(low=0, high=(2**32) - 1, endpoint=True)))
        init_weights(model, seed=init_seed)

    optimizer = optimizer_factory.create()
    scheduler = scheduler_factory.create(optimizer)

    if state is None:
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

        correct = 0
        total = 0 
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

            predicted = outputs.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += bs


        train_loss = train_loss_sum / max(1, n_samples)
        train_second_loss = train_second_loss_sum / max(1, n_samples)
        train_acc = correct / max(1, total)

        state.metrics.log(
            train_loss_per_epoch=train_loss,
            train_second_loss_per_epoch=train_second_loss,
            train_acc_per_epoch=train_acc,
        )

        if verbose:
            if trial_id is not None:
                print(f"Trial {trial_id} | Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Train CE: {train_second_loss:.6f} | Train Acc: {train_acc:.4f}", flush=True)
            else:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Train CE: {train_second_loss:.6f} | Train Acc: {train_acc:.4f}", flush=True)




        model.eval()
        val_loss_sum = 0.0
        val_cross_entropy_sum = 0.0
        n_samples = 0

        correct = 0
        total = 0
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

                predicted = preds.argmax(dim=1)
                correct += (predicted == y_val).sum().item()
                total += bs


        val_loss = val_loss_sum / max(1, n_samples)
        val_second_loss = (val_cross_entropy_sum / max(1, n_samples))
        val_acc = correct / max(1, total)

        state.metrics.log(
            val_loss_per_epoch=val_loss,
            val_second_loss_per_epoch=val_second_loss,
            val_acc_per_epoch=val_acc,
        )

        if verbose:
            if trial_id is not None:
                print(f"Trial {trial_id} | Epoch {epoch+1}/{epochs} | Validation Loss: {val_loss:.6f} | Validation CE: {val_second_loss:.6f} | Validation Acc: {val_acc:.4f}", flush=True)
            else:
                print(f"Epoch {epoch+1}/{epochs} | Validation Loss: {val_loss:.6f} | Validation CE: {val_second_loss:.6f} | Validation Acc: {val_acc:.4f}", flush=True)




        if scheduler is not None:
            try:
                if scheduler_factory.scheduler_name == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            except Exception:
                raise RuntimeError("Scheduler error.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_best and save_dir and save_name:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, save_name)                
                save_model(state, save_path)
                print(f"* New best model saved in {save_path} with val_loss {val_loss:.6f} *", flush=True)


        cb_manager.on_epoch_end(state)

        if stop_manager and stop_manager.step(state):
            if verbose:
                print(f"[EARLY STOP] reason={stop_manager.stop_reason} | epoch={epoch+1}/{epochs}", flush=True)
            break

    cb_manager.on_train_end(state)
    state.last_scheduler_lr = scheduler.get_last_lr()[0]
    return state

    
def train_model_spl(
    model,
    train_data,
    val_data,
    loss_fn,
    optimizer_factory,
    scheduler_factory,
    epochs=20,
    device="cuda",
    verbose=True,
    params=None,
    callbacks: Optional[CallbackManager] = None,
    save_best: bool = False,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
    init_weights=None,
    stop_criteria: Optional[list] = None,
    seed=None,
    trial_id=None,
):
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




    spl_lambda0 = params.get("spl_lambda0", 0.5) if params else 0.5
    spl_k = params.get("spl_k", 1.5) if params else 1.5

    def spl_weights(losses, lamb):
        return (losses < lamb).float()




    for epoch in range(epochs):
        state.epoch = epoch
        cb_manager.on_epoch_begin(state)

        model.train()

        train_loss_sum = 0.0
        train_second_loss_sum = 0.0
        n_samples = 0

        lamb = spl_lambda0 * (spl_k ** epoch)

        correct = 0
        total = 0
        for x, y, y_baseline in train_data:
            state.step += 1

            x = x.to(device)
            y = y.to(device).long()
            y_baseline = y_baseline.to(device)

            optimizer.zero_grad()
            outputs = model(x)


            per_sample_loss = loss_fn(outputs, y, y_baseline=y_baseline, epoch=epoch, max_epochs=epochs, reduction="none")


            weights = spl_weights(per_sample_loss.detach(), lamb)

            loss = (weights * per_sample_loss)
            train_loss_sum += loss.sum().item()
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            bs = y.numel()

            train_second_loss_sum += torch.nn.functional.cross_entropy(outputs, y, reduction='none').sum().item()
            n_samples += bs

            predicted = outputs.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += bs

        train_loss = train_loss_sum / max(1, n_samples)
        train_second_loss = train_second_loss_sum / max(1, n_samples)
        train_acc = correct / max(1, total)

        state.metrics.log(
            train_loss_per_epoch=train_loss,
            train_second_loss_per_epoch=train_second_loss,
            train_acc_per_epoch=train_acc,
        )

        if verbose:
            if trial_id is not None:
                print(f"Trial {trial_id} | Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Train CE: {train_second_loss:.6f} | Train Acc: {train_acc:.4f}", flush=True)
            else:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Train CE: {train_second_loss:.6f} | Train Acc: {train_acc:.4f}", flush=True)





        model.eval()

        val_loss_sum = 0.0
        val_cross_entropy_sum = 0.0
        n_samples = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val, y_val_baseline in val_data:

                x_val = x_val.to(device)
                y_val = y_val.to(device).long()
                y_val_baseline = y_val_baseline.to(device)

                preds = model(x_val)

                loss = loss_fn(preds, y_val, y_baseline=y_val_baseline,
                                epoch=epoch, max_epochs=epochs, reduction="none")
                bs = y_val.numel()

                val_loss_sum += loss.sum().item()
                val_cross_entropy_sum += torch.nn.functional.cross_entropy(preds, y_val, reduction='none').sum().item()
                n_samples += bs
                predicted = preds.argmax(dim=1)
                correct += (predicted == y_val).sum().item()
                total += bs

        val_loss = val_loss_sum / max(1, n_samples)
        val_second_loss = (val_cross_entropy_sum / max(1, n_samples))
        val_acc = correct / max(1, total)

        state.metrics.log(
            val_loss_per_epoch=val_loss,
            val_second_loss_per_epoch=val_second_loss,
            val_acc_per_epoch=val_acc,
        )

        if verbose:
            if trial_id is not None:
                print(f"Trial {trial_id} | Epoch {epoch+1}/{epochs} | Validation Loss: {val_loss:.6f} | Validation CE: {val_second_loss:.6f} | Validation Acc: {val_acc:.4f}", flush=True)
            else:
                print(f"Epoch {epoch+1}/{epochs} | Validation Loss: {val_loss:.6f} | Validation CE: {val_second_loss:.6f} | Validation Acc: {val_acc:.4f}", flush=True)





        if scheduler is not None:
            try:
                if scheduler_factory.scheduler_name == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            except Exception:
                raise RuntimeError("Scheduler error.")





        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_best and save_dir and save_name:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, save_name)                
                save_model(state, save_path)
                print(f"* New best model saved in {save_path} with val_loss {val_loss:.6f} *", flush=True)

        cb_manager.on_epoch_end(state)

        if stop_manager and stop_manager.step(state):
            if verbose:
                print(f"[EARLY STOP] reason={stop_manager.stop_reason} | epoch={epoch+1}/{epochs}", flush=True)
            break

    cb_manager.on_train_end(state)

    return state




from models.mentornet_pd import (load_mentornet_pd, student_pd_weights,
                           PD_MOVING_AVERAGE_DECAY)


def train_model_mentornet_pd(
    model,
    train_data,
    val_data,
    loss_fn,
    optimizer_factory,
    scheduler_factory,
    epochs=20,
    device="cuda",
    verbose=True,
    params=None,
    callbacks: Optional[CallbackManager] = None,
    save_best: bool = False,
    save_dir: Optional[str] = None,
    save_name: Optional[str] = None,
    init_weights=None,
    stop_criteria: Optional[list] = None,
    seed=None,
    trial_id=None,
):
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




    config = params or {}
    mnpd_loss_p_percentile = float(config.get('mnpd_loss_p_percentile', 0.75))
    mnpd_burn_in_pct = int(config.get('mnpd_burn_in_pct', 0))
    mnpd_moving_average_decay = float(config.get(
        'mnpd_moving_average_decay', PD_MOVING_AVERAGE_DECAY))
    if not 0.0 <= mnpd_loss_p_percentile <= 1.0:
        raise ValueError('mnpd_loss_p_percentile must be in [0, 1]')
    if not 0 <= mnpd_burn_in_pct <= 100:
        raise ValueError('mnpd_burn_in_pct must be in [0, 100]')
    if not 0.0 <= mnpd_moving_average_decay <= 1.0:
        raise ValueError('mnpd_moving_average_decay must be in [0, 1]')
    mentor = load_mentornet_pd(device=device)
    loss_moving_avg = 0.0
    total_steps = epochs * len(train_data)
    original_weight_decay = [group['weight_decay'] for group in optimizer.param_groups]

    for epoch in range(epochs):
        state.epoch = epoch
        cb_manager.on_epoch_begin(state)

        model.train()

        train_loss_sum = 0.0
        train_second_loss_sum = 0.0
        n_samples = 0

        correct = 0
        total = 0
        for x, y, y_baseline in train_data:
            state.step += 1

            x = x.to(device)
            y = y.to(device).long()
            y_baseline = y_baseline.to(device)

            optimizer.zero_grad()
            outputs = model(x)


            per_sample_loss = loss_fn(outputs, y, y_baseline=y_baseline, epoch=epoch, max_epochs=epochs, reduction="none")


            progress = min(99, int((state.step - 1) * 100 / max(1, total_steps)))
            dropout_rate = 0.5 if progress < 17 else 0.05
            if progress < mnpd_burn_in_pct:
                weights = torch.ones_like(per_sample_loss)
            else:
                weights, loss_moving_avg = student_pd_weights(
                    mentor, per_sample_loss, progress, loss_moving_avg,
                    mnpd_loss_p_percentile, dropout_rate,
                    decay=mnpd_moving_average_decay)
            for group, wd in zip(optimizer.param_groups, original_weight_decay):
                group['weight_decay'] = wd * weights.mean().item()


            loss = (weights * per_sample_loss)
            train_loss_sum += loss.sum().item()
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            bs = y.numel()

            train_second_loss_sum += torch.nn.functional.cross_entropy(outputs, y, reduction='none').sum().item()
            n_samples += bs

            predicted = outputs.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += bs

        train_loss = train_loss_sum / max(1, n_samples)
        train_second_loss = train_second_loss_sum / max(1, n_samples)
        train_acc = correct / max(1, total)

        state.metrics.log(
            train_loss_per_epoch=train_loss,
            train_second_loss_per_epoch=train_second_loss,
            train_acc_per_epoch=train_acc,
        )

        if verbose:
            if trial_id is not None:
                print(f"Trial {trial_id} | Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Train CE: {train_second_loss:.6f} | Train Acc: {train_acc:.4f}", flush=True)
            else:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Train CE: {train_second_loss:.6f} | Train Acc: {train_acc:.4f}", flush=True)





        model.eval()

        val_loss_sum = 0.0
        val_cross_entropy_sum = 0.0
        n_samples = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val, y_val_baseline in val_data:

                x_val = x_val.to(device)
                y_val = y_val.to(device).long()
                y_val_baseline = y_val_baseline.to(device)

                preds = model(x_val)

                loss = loss_fn(preds, y_val, y_baseline=y_val_baseline,
                                epoch=epoch, max_epochs=epochs, reduction="none")
                bs = y_val.numel()

                val_loss_sum += loss.sum().item()
                val_cross_entropy_sum += torch.nn.functional.cross_entropy(preds, y_val, reduction='none').sum().item()
                n_samples += bs
                predicted = preds.argmax(dim=1)
                correct += (predicted == y_val).sum().item()
                total += bs

        val_loss = val_loss_sum / max(1, n_samples)
        val_second_loss = (val_cross_entropy_sum / max(1, n_samples))
        val_acc = correct / max(1, total)

        state.metrics.log(
            val_loss_per_epoch=val_loss,
            val_second_loss_per_epoch=val_second_loss,
            val_acc_per_epoch=val_acc,
        )

        if verbose:
            if trial_id is not None:
                print(f"Trial {trial_id} | Epoch {epoch+1}/{epochs} | Validation Loss: {val_loss:.6f} | Validation CE: {val_second_loss:.6f} | Validation Acc: {val_acc:.4f}", flush=True)
            else:
                print(f"Epoch {epoch+1}/{epochs} | Validation Loss: {val_loss:.6f} | Validation CE: {val_second_loss:.6f} | Validation Acc: {val_acc:.4f}", flush=True)





        if scheduler is not None:
            try:
                if scheduler_factory.scheduler_name == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            except Exception:
                raise RuntimeError("Scheduler error.")





        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_best and save_dir and save_name:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, save_name)
                save_model(state, save_path)
                print(f"* New best model saved in {save_path} with val_loss {val_loss:.6f} *", flush=True)

        cb_manager.on_epoch_end(state)

        if stop_manager and stop_manager.step(state):
            if verbose:
                print(f"[EARLY STOP] reason={stop_manager.stop_reason} | epoch={epoch+1}/{epochs}", flush=True)
            break

    cb_manager.on_train_end(state)

    return state
