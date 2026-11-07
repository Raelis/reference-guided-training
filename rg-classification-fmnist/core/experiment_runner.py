# Run Optuna searches, final fits, checkpoint selection and result storage.
import csv
import json
import os
import re
import shutil
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.samplers import TPESampler

from core.experiment_paths import architecture_dir, method_dir, crossfit_guide_dir
from core.dataset_builder import DatasetBuilder
from core.loss_factory import LossFactory
from core.optimizers import OptimizerFactory
from core.schedulers import SchedulerFactory
from core.trainer import train_model, train_model_spl, train_model_mentornet_pd
from core.experiment import (METHOD_SPECS, method_spec, BASE_SEED, BASE_LR, LR_GRID,
    FIXED_WEIGHT_DECAY, FIXED_BATCH_SIZE, FIXED_MOMENTUM)

STUDY_NAME = 'experiment_with_indicators'
MENTOR_PD_STUDY_NAME = 'experiment_with_indicators_mentornet_pd_tuned'


def _study_name(method):
    return MENTOR_PD_STUDY_NAME if method == 'mentornet-pd' else STUDY_NAME
METHODS = tuple(METHOD_SPECS)
TRIAL_BUDGET = 20
_INTERRUPTED_ATTR = '_interrupted_by_user'
_RESTART_SOURCE_ATTR = '_restarted_from_trial'
_RESTART_SEED_ATTR = '_original_trial_number'
_REQUEUED_ATTR = '_interrupted_trials_requeued'
_FINITE_LR_ATTR = '_finite_lr_value'


def _trial_seed_number(trial):
    # Use each trial's own seed, or the original number for a restarted attempt.
    return int(trial.user_attrs.get(_RESTART_SEED_ATTR, trial.number))


def _mark_interrupted(trial):
    # Tag Ctrl+C failures so the next execution can restart their parameters.
    trial.set_user_attr(_INTERRUPTED_ATTR, True)


def _remaining_optuna_trials(study, budget):
    # Restart abandoned/failed trials from epoch one without changing their seeds.
    budget = min(budget, TRIAL_BUDGET)
    state = optuna.trial.TrialState
    requeued = set(study.user_attrs.get(_REQUEUED_ATTR, []))
    for old in study.get_trials(deepcopy=False):
        if old.number in requeued:
            continue
        if old.state not in (state.RUNNING, state.FAIL):
            continue
        if old.state == state.RUNNING:
            study.tell(old.number, state=state.FAIL)
        study.enqueue_trial(dict(old.params), user_attrs={
            _RESTART_SOURCE_ATTR: old.number,
            _RESTART_SEED_ATTR: int(old.user_attrs.get(_RESTART_SEED_ATTR, old.number)),
        })
        requeued.add(old.number)
        study.set_user_attr(_REQUEUED_ATTR, sorted(requeued))
        print(f'Restarting interrupted Optuna trial {old.number} from epoch 1.', flush=True)

    # An interrupted or failed attempt never consumes another trial slot.
    consumed = sum(t.state not in (state.WAITING, state.RUNNING)
                   and t.number not in requeued
                   for t in study.get_trials(deepcopy=False))
    return max(0, budget - consumed)



def _finite_lr_trials(study, options, budget):
    # Queue each distinct LR at most once and preserve failed trials' seeds on retry.
    # Support existing TPE studies. Already completed values count as covered,
    # even when the earlier study contains repeated configurations.

    options = tuple(options)[:min(int(budget), TRIAL_BUDGET)]
    state = optuna.trial.TrialState
    trials = study.get_trials(deepcopy=False)
    complete = {t.params['lr'] for t in trials
                if t.state == state.COMPLETE and t.params.get('lr') in options}
    queued = set()
    requeued = set(study.user_attrs.get(_REQUEUED_ATTR, []))
    for trial in trials:
        if trial.state != state.WAITING:
            continue
        fixed = trial.system_attrs.get('fixed_params', {})
        lr = fixed.get('lr', trial.user_attrs.get(_FINITE_LR_ATTR))
        if lr in options and lr not in complete:
            queued.add(lr)

    # A running trial in the persisted study belongs to an interrupted process.
    # Restart it from epoch 1 using its original parameters and random seed.
    for trial in trials:
        if trial.state == state.RUNNING:
            study.tell(trial.number, state=state.FAIL)
        if trial.state not in (state.RUNNING, state.FAIL):
            continue
        lr = trial.params.get('lr')
        if (lr not in options or lr in complete or lr in queued
                or trial.number in requeued):
            continue
        study.enqueue_trial(dict(trial.params), user_attrs={
            _RESTART_SOURCE_ATTR: trial.number,
            _RESTART_SEED_ATTR: int(trial.user_attrs.get(_RESTART_SEED_ATTR,
                                                         trial.number)),
            _FINITE_LR_ATTR: lr,
        })
        requeued.add(trial.number)
        study.set_user_attr(_REQUEUED_ATTR, sorted(requeued))
        queued.add(lr)
        print(f'Restarting interrupted Optuna trial {trial.number} '
              f'(lr={lr}) from epoch 1.', flush=True)

    for lr in options:
        if lr not in complete and lr not in queued:
            study.enqueue_trial({'lr': lr}, user_attrs={_FINITE_LR_ATTR: lr})
            queued.add(lr)
    # Include WAITING trials inherited from an older runner; no training will
    # occur for a duplicate LR whose result is already COMPLETE.
    pending = sum(t.state == state.WAITING for t in study.get_trials(deepcopy=False))
    return options, pending


def _optimize_finite_lr(study, objective, options, max_attempts, *, catch):
    """Run queued finite-grid trials without training duplicate LR values."""
    state = optuna.trial.TrialState
    for _ in range(max_attempts):
        completed = {t.params['lr'] for t in study.get_trials(deepcopy=False)
                     if t.state == state.COMPLETE and t.params.get('lr') in options}
        if len(completed) == len(options):
            break

        def skip_completed(trial):
            # Only one worker is used, so the coverage check is stable.
            params = trial.system_attrs.get('fixed_params', {})
            lr = params.get('lr')
            if lr in completed:
                raise optuna.TrialPruned(f'LR {lr} already evaluated')
            return objective(trial)

        study.optimize(skip_completed, n_trials=1, catch=catch, n_jobs=1)

    completed = {t.params['lr'] for t in study.get_trials(deepcopy=False)
                 if t.state == state.COMPLETE and t.params.get('lr') in options}
    missing = [lr for lr in options if lr not in completed]
    if missing:
        raise RuntimeError(f'LR configurations failed or were interrupted: {missing}. '
                           'Use --resume to restart these trials with their original seeds.')


def ensure_noise_cache(noise):
    """Create deterministic noise-label tensors when their cache is absent."""
    name = 'clean' if noise == 0 else f'fmnist-sym_{int(noise*100)}'
    path = Path('noise_cache') / f'{name}_seed{BASE_SEED}_fashionmnist.pt'
    if path.is_file():
        return path
    from torchvision.datasets import FashionMNIST
    original = FashionMNIST(root='./data', train=True, download=True)
    ds = DatasetBuilder(noise_ratio=noise, baseline_noise_ratio=noise,
                        baseline_method='no', seed=BASE_SEED, device='cpu')
    ds.load_or_create_noise(original.targets.detach().clone())
    if not path.is_file():
        raise RuntimeError(f'Unable to build fixed noise cache: {path}')
    print(f'Generated deterministic noise cache: {path}')
    return path


from core.model_factory import model_class, construct, architecture_from_study
from core.utils import seed_everything, minimum_finite as _minimum_finite


def read_architecture(model):
    cls = model_class(model)
    if model in ('ResNet20', 'MLP-guide'):
        return {}
    study_path = architecture_dir(model) / 'optuna_study.db'
    if not study_path.is_file():
        raise FileNotFoundError(
            f'Missing frozen stage-1 ERM architecture study: {study_path}. '
            f'Run `python train.py --stage architecture --model {model} --noise 0` first, '
            f'or provide the clean-data architecture study. Do not choose architecture on test.')
    s = optuna.load_study(study_name=STUDY_NAME, storage=f'sqlite:///{study_path.resolve()}')
    return architecture_from_study(model, s.best_params)


def output_dir(model, noise, method, guide_noise=None):
    return method_spec(method).output_dir(model, noise, guide_noise)


def selected_study_path(model, noise, method, guide_noise):
    return method_spec(method).study_path(model, noise, guide_noise)


def make_data(noise, method, guide_noise, device):
    guide_name = 'ResNet20' if method in ('rg', 'kd', 'rg-mean-one') else 'no'
    ds = DatasetBuilder(noise_ratio=noise, baseline_noise_ratio=guide_noise,
                        baseline_method=guide_name, seed=BASE_SEED, device=device)
    ds.build()
    return ds


def sample_params(trial, method, args=None):
    return method_spec(method).sample_parameters(
        trial, lr_grid=bool(getattr(args, 'lr_grid', False)))


def sample_params_from_trial(best_trial, method):
    return method_spec(method).selected_parameters(best_trial)




def run_architecture_stage(model_name, *, loss_name='ce', noise_ratio=0.,
                           seed=BASE_SEED, device=None):
    if model_name in ('ResNet20', 'MLP-guide'):
        print(f'{model_name} has a fixed architecture; no Stage-1 study is needed.')
        return

    seed_everything(seed)
    Model = model_class(model_name)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    folder = architecture_dir(Model.model_name, noise_ratio)
    folder.mkdir(parents=True, exist_ok=True)
    print('Folder created:', str(folder))

    try:
        dataset = DatasetBuilder(
            noise_ratio=noise_ratio, seed=seed,
            baseline_method='no', device=device,
        )
        dataset.build()
        run_architecture_search(
            str(folder), dataset, Model, LossFactory(loss_name=loss_name),
            max_trials=20, epochs=30, patience=15,
            seed=seed, device=device,
        )
        db = folder / 'optuna_study.db'
        study = optuna.load_study(
            study_name=STUDY_NAME, storage=f'sqlite:///{db.resolve()}',
        )
        if not any(t.state == optuna.trial.TrialState.COMPLETE
                   for t in study.trials):
            raise RuntimeError(f'No successful architecture trial in {db}')
    except Exception as error:
        error_file = folder / 'error.txt'
        with error_file.open('w') as handle:
            handle.write('An error occurred:\n')
            handle.write(str(error) + '\n\n')
            handle.write('Traceback:\n')
            handle.write(traceback.format_exc())
        print('An error occurred:', error)
        print(f'Error saved to {error_file}')
        raise


def run_architecture_search(experiment_dir, dataset_builder, model_class, loss_factory,
                            *, max_trials=20, epochs=30, patience=15, seed=42, device='cpu'):
    # Select a Stage-1 architecture using validation cross-entropy.
    import pickle
    from .early_stopping import LossEarlyStopping

    def objective(trial):
        try:
            seed_everything(seed + _trial_seed_number(trial))
            batch_size = trial.suggest_categorical('batch_size', [256])
            train_loader, val_loader, _ = dataset_builder.build_loaders(batch_size=batch_size)
            lr = trial.suggest_float('lr', 5e-3, 5e-3)
            weight_decay = trial.suggest_float('weight_decay', 5e-4, 5e-4)
            model_params = model_class.prepare_trial(trial)
            model = model_class(**model_params).to(device)
            model.model_postprocess(model)
            scheduler_factory = SchedulerFactory(scheduler_name='cosine')
            optimizer_factory = OptimizerFactory(model, optimizer_name='sgd',
                                                 lr=lr, weight_decay=weight_decay)
            state = train_model(
                model, train_loader, val_loader, loss_factory.create(),
                optimizer_factory, scheduler_factory, epochs=epochs, device=device,
                stop_criteria=[LossEarlyStopping(
                    patience, metric='val_second_loss_per_epoch')], callbacks=[],
                trial_id=trial.number,
            )
            values = state.metrics.get('val_second_loss_per_epoch') if state is not None else None
            if values is None or len(values) == 0:
                raise RuntimeError('Invalid training state')
            validation_ce = np.asarray(values, dtype=float)
            if not np.isfinite(validation_ce).all():
                raise FloatingPointError('Non-finite validation values')
            if trial.should_prune():
                raise optuna.TrialPruned()
            best = validation_ce.min()
            print(f'* The best trial loss was {best}')
            return best
        except torch.cuda.OutOfMemoryError:
            print(f'[Trial {trial.number}] Out of Memory.')
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        except KeyboardInterrupt:
            _mark_interrupted(trial)
            raise

    experiment_dir = Path(experiment_dir)
    study_path = experiment_dir / 'optuna_study.db'
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction='minimize',
        sampler=TPESampler(seed=seed, n_startup_trials=5, multivariate=True, group=True),
        storage=f'sqlite:///{study_path}',
        load_if_exists=True,
    )
    remaining = _remaining_optuna_trials(study, max_trials)
    if remaining <= 0:
        print('Trial budget already reached.')
        return
    study.optimize(objective, n_trials=remaining,
                   catch=(RuntimeError, FloatingPointError), n_jobs=1)
    print('Best trial number:', study.best_trial.number)
    print('Best HP found:', study.best_trial.params)
    print(f'Best CE validation loss: {study.best_value:.4f}')
    with (experiment_dir / 'optuna_best-params.pkl').open('wb') as file:
        pickle.dump(study.best_trial.params, file)


def run_single_experiment(ds, model_name, arch, method, params, args, seed, *, mode, trial_number=None,
               save=None, checkpoint_prefix=None, checkpoint_filename=None, return_details=False):
    if mode not in ('search', 'fit'):
        raise ValueError(f'Unknown single-experiment mode: {mode}')
    if (mode == 'search' and save is not None) or (mode == 'fit' and save is None):
        raise ValueError('Optuna search must not save fit checkpoints; final fit requires a destination')
    method_config = method_spec(method)
    seed_everything(seed)
    device = args.device
    train_loader, val_loader, test_loader = ds.build_loaders(batch_size=params['batch_size'])
    model = construct(model_name, arch, device)
    checkpoint_params = dict(arch)
    checkpoint_params.update(params)
    if method == 'mentornet-pd':
        from models.mentornet_pd import MENTOR_CHECKPOINT
        checkpoint_params['mentornet_checkpoint'] = str(MENTOR_CHECKPOINT)



    opt = OptimizerFactory(model, optimizer_name='sgd', lr=params['lr'],
                           weight_decay=params['weight_decay'])
    sched = SchedulerFactory(scheduler_name='cosine', T_max=args.epochs)

    trainers = {'standard': train_model, 'spl': train_model_spl,
                'mentornet-pd': train_model_mentornet_pd}
    trainer = trainers[method_config.trainer]



    trial_stop_criteria = None
    if mode == 'search':
        from .early_stopping import LossEarlyStopping
        trial_stop_criteria = [LossEarlyStopping(
            30, metric='val_second_loss_per_epoch')]
    state = trainer(model, train_loader, val_loader, method_config.make_loss(params),
                        opt, sched, epochs=args.epochs, device=device,
                        verbose=not args.quiet, params=checkpoint_params,
                        save_best=mode == 'fit', save_dir=str(save) if mode == 'fit' else None,
                        save_name=(checkpoint_filename if checkpoint_filename else f'{checkpoint_prefix + "_" if checkpoint_prefix else ""}best-model_seed{seed}.pt') if mode == 'fit' else None,
                        trial_id=trial_number, stop_criteria=trial_stop_criteria)


    if mode == 'fit':
        epoch_losses = state.metrics.get('val_loss_per_epoch')
        completed_epochs = len(epoch_losses) if epoch_losses is not None else 0
        if completed_epochs != args.epochs:
            raise RuntimeError(
                f'Incomplete final fit for {method}: {completed_epochs}/{args.epochs} epochs. '
                'Final runs must not use early stopping.')



    val_ce = state.metrics.get('val_second_loss_per_epoch')
    if mode == 'search':
        if val_ce is None or not np.isfinite(np.asarray(val_ce, dtype=float)).all():
            raise FloatingPointError(
                'Training produced missing/nonfinite validation cross-entropy')
        result = {'validation_ce': float(np.min(val_ce))}
        if method == 'mentornet-pd':


            result['validation_ce_per_epoch'] = list(val_ce)
        return result





    best_val_objective = _minimum_finite(
        state.metrics.get('val_loss_per_epoch'), 'validation objective loss')

    ckpt_file = save / (checkpoint_filename if checkpoint_filename else f'{checkpoint_prefix + "_" if checkpoint_prefix else ""}best-model_seed{seed}.pt')
    if not ckpt_file.is_file():
        raise FileNotFoundError(f'Best-validation checkpoint not written for seed {seed}: {ckpt_file}')
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)

    weights = ckpt['state_dict']
    if any(torch.is_tensor(t) and not torch.isfinite(t).all().item()
           for t in weights.values()):
        raise FloatingPointError(f'Nonfinite weights in best checkpoint: {ckpt_file}')
    model.load_state_dict(weights)
    model.eval()
    n = correct = 0
    truths, preds, probs = [], [], []
    with torch.inference_mode():
        for x, y, _ in test_loader:
            scores = model(x.to(device)).cpu()
            if not torch.isfinite(scores).all().item():
                raise FloatingPointError(f'Nonfinite predictions from checkpoint: {ckpt_file}')
            pred = scores.argmax(1)
            correct += int((pred == y).sum().item())
            n += int(y.numel())
            if return_details:
                truths.append(y.numpy())
                preds.append(pred.numpy())
                probs.append(torch.softmax(scores, dim=1).numpy())
    checkpoint_metrics = ckpt.get('metrics', {})
    best_epoch = len(checkpoint_metrics.get('val_loss_per_epoch', []))
    if not best_epoch:
        raise RuntimeError(f'Checkpoint does not record a selected epoch: {ckpt_file}')
    result = {'validation_ce': _minimum_finite(val_ce, 'validation cross-entropy'),
              'accuracy': correct/n, 'best_epoch': best_epoch}
    if return_details:
        result['y_true'] = np.concatenate(truths)
        result['y_pred'] = np.concatenate(preds)
        result['y_probs'] = np.concatenate(probs)
        result['val_objective'] = best_val_objective
        result['epochs'] = len(state.metrics.get('val_loss_per_epoch'))
    return result




def _existing_run_ids(folder):
    """Report checkpoint IDs from the experiment directory, not from metadata."""
    found = set()
    for path in folder.glob('best-model_r*.pt'):
        match = re.fullmatch(r'best-model_r([1-9][0-9]*)\.pt', path.name)
        if match and path.is_file():
            found.add(int(match.group(1)))
    return found


def _prediction_pairs(predictions):
    """Find complete run-indexed prediction pairs, independently of total_runs."""
    true_ids, pred_ids = set(), set()
    for col in predictions.columns:
        m = re.fullmatch(r'y_(true|pred)_r([1-9][0-9]*)', str(col))
        if m:
            (true_ids if m.group(1) == 'true' else pred_ids).add(int(m.group(2)))
    return true_ids & pred_ids, true_ids ^ pred_ids


def _align_existing_csv(csv_df, predictions, selected):
    """Match selected run IDs to CSV rows and preserve unrelated records."""
    from sklearn.metrics import accuracy_score, f1_score

    original = csv_df.to_dict('records')
    count = len(original)
    pairs, incomplete = _prediction_pairs(predictions)
    mapping = {}
    used = set()

    if not count:
        return original, mapping, set()

    if 'run_id' in csv_df.columns and csv_df['run_id'].notna().all():
        try:
            explicit = [int(v) for v in csv_df['run_id']]
            if (len(set(explicit)) != count or any(i < 1 for i in explicit)
                    or any(float(v) != i for v, i in zip(csv_df['run_id'], explicit))):
                raise ValueError('Invalid run identifiers')
        except (TypeError, ValueError) as exc:
            raise RuntimeError('Existing CSV has invalid or duplicate run_id values') from exc
        mapping = {run_id: offset for offset, run_id in enumerate(explicit)}
        return original, mapping, set()

    if 'accuracy' not in csv_df.columns or 'f1_score' not in csv_df.columns:
        if any(i <= count for i in selected):
            raise RuntimeError('Cannot identify selected CSV runs: accuracy/F1 missing.')
        return original, mapping, set()

    def candidates_for(run_id):
        if run_id not in pairs:
            return []
        truth = predictions[f'y_true_r{run_id}'].to_numpy()
        pred = predictions[f'y_pred_r{run_id}'].to_numpy()
        if not len(truth) or len(truth) != len(pred):
            return []
        try:
            acc = accuracy_score(truth, pred)
            f1 = f1_score(truth, pred, average='macro', zero_division=0)
        except (ValueError, TypeError):
            return []
        result = []
        for offset, row in enumerate(original):
            try:
                row_acc = float(row['accuracy'])
                row_f1 = float(row['f1_score'])
            except (ValueError, TypeError, KeyError):
                continue
            if (np.isfinite(row_acc) and np.isfinite(row_f1)
                    and abs(row_acc - acc) <= 2e-6
                    and abs(row_f1 - f1) <= 2e-6):
                result.append(offset)
        return result



    for run_id in sorted(selected):
        if run_id > count:
            continue
        hits = [j for j in candidates_for(run_id) if j not in used]
        if run_id - 1 in hits:
            chosen = run_id - 1
        elif len(hits) == 1:
            chosen = hits[0]
        else:
            raise RuntimeError(
                f'Cannot identify the existing CSV record for SELECTED r{run_id} '
                f'(matching offsets, 1-based: {[j+1 for j in hits]}). '
                'Other runs were not inspected or modified. The existing '
                'CSV has no run_id; overwriting a guessed row could corrupt '
                'another seed. Inspect rN predictions or provide an explicit '
                'run-to-row mapping before replacing this run.')
        mapping[run_id] = chosen
        used.add(chosen)




    unverified = set()
    for run_id in sorted(i for i in pairs if i <= count and i not in selected):
        hits = [j for j in candidates_for(run_id) if j not in used]
        if run_id - 1 in hits:
            chosen = run_id - 1
        elif len(hits) == 1:
            chosen = hits[0]
        else:
            unverified.add(run_id)
            continue
        mapping[run_id] = chosen
        used.add(chosen)
    unverified.update(i for i in range(1, count + 1)
                      if i not in mapping and i not in selected)
    return original, mapping, unverified


def _refresh_checkpoint_aliases(folder, rows, csv_row_by_run):
    # Select the best and median validation-objective runs from saved checkpoints.
    candidates = []
    current_ckpts = _existing_run_ids(folder)
    for run_id, offset in sorted(csv_row_by_run.items()):
        if run_id not in current_ckpts:
            continue
        try:
            val = float(rows[offset]['loss-value'])
        except (TypeError, ValueError, KeyError):
            continue
        if np.isfinite(val):
            candidates.append((run_id, val))


    unmapped_checkpoints = current_ckpts - set(csv_row_by_run)
    if unmapped_checkpoints:
        print(f'WARNING: cannot safely recalculate best-model.pt and '
              f'median-model.pt: checkpoint run IDs without verified CSV '
              f'losses = {sorted(unmapped_checkpoints)}. Existing aliases '
              'are preserved; no checkpoints are deleted.')
        return
    if not candidates:
        print('WARNING: No finite validation objectives with verified '
              'checkpoints; best-model.pt and median-model.pt unchanged.')
        return
    best_run = min(candidates, key=lambda x: x[1])[0]
    median_run = sorted(candidates, key=lambda x: x[1])[len(candidates) // 2][0]
    for target, chosen in [('best-model.pt', best_run),
                           ('median-model.pt', median_run)]:
        temp = folder / ('.' + target + '.tmp')
        shutil.copy2(folder / f'best-model_r{chosen}.pt', temp)
        os.replace(temp, folder / target)
    print(f'Final: {folder / "experiment-results.csv"}; best run={best_run}, median run={median_run}')


def run_final_fit(args, method, condition=None):
    # Run final training seeds and write condition-level artifacts.
    import time
    import pandas as pd
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, balanced_accuracy_score,
                                 matthews_corrcoef, cohen_kappa_score,
                                 roc_auc_score, average_precision_score, log_loss)
    from core.utils import save_parquet_with_config, load_parquet_with_config

    if method not in METHODS:
        raise ValueError(f'No final fit implemented for {method}')
    arch = (read_stage_one_parameters(args.model) if condition is not None and
            condition.parameter_source == 'architecture' else read_architecture(args.model))
    ensure_noise_cache(args.noise)
    folder = Path(condition.directory) if condition is not None else output_dir(
        args.model, args.noise, method, args.guide_noise)


    folder.mkdir(parents=True, exist_ok=True)

    if condition is not None:
        selected_trial_number = None
        params, selected_trial_number = scenario_parameters(
            condition, return_selected_trial=True)
    elif method == 'erm' and (getattr(args, 'guide_fit', False) or not getattr(args, 'lr_grid', False)):

        selected_trial_number = None
        params = dict(lr=BASE_LR,
                      weight_decay=FIXED_WEIGHT_DECAY,
                      batch_size=FIXED_BATCH_SIZE, momentum=FIXED_MOMENTUM)
    else:
        study_db = selected_study_path(args.model, args.noise, method, args.guide_noise)
        if not study_db.is_file():
            raise FileNotFoundError(
                f'Missing selected Optuna study for {method}: {study_db}. '
                'Run --stage search or restore its optuna_study.db; --stage fit does not tune.')
        study = optuna.load_study(study_name=_study_name(method),
                                  storage=f'sqlite:///{study_db.resolve()}')
        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            raise RuntimeError(f'No COMPLETE Optuna trials in {study_db}')
        selected_trial_number = study.best_trial.number
        selection_method = 'rg' if method in ('rg-mean-one', 'erm-lr-matched') else method
        params = sample_params_from_trial(study.best_trial, selection_method)

        if getattr(args, 'lr_grid', False) and method == 'erm-lr-matched':
            params['lr'] = study.best_trial.params['lr']

    args.epochs = args.fit_epochs
    folder_csv = folder / 'experiment-results.csv'
    folder_parquet = folder / 'experiment_data.parquet'





    config = {'total_runs': 0}
    rows = []
    predictions = pd.DataFrame()
    run_ids_arg = getattr(args, 'run_ids', None)
    start_run_arg = getattr(args, 'start_run', None)
    resume = bool(getattr(args, 'resume', False))
    manual_selection = run_ids_arg is not None or start_run_arg is not None
    if sum((run_ids_arg is not None, start_run_arg is not None, resume)) > 1:
        raise ValueError('--run-ids, --start-run and --resume cannot be combined.')
    if run_ids_arg is not None:
        try:
            selected = sorted(set(int(part.strip()) for part in run_ids_arg.split(',')))
        except (TypeError, ValueError):
            raise ValueError('Use --run-ids 2,3 or --run-ids 1,3,5 (comma-separated integers).') from None
        if not selected or any(i < 1 for i in selected):
            raise ValueError('--run-ids must contain positive run numbers.')
    elif start_run_arg is not None:
        if not 1 <= start_run_arg <= args.runs:
            raise ValueError(f'--start-run must be between 1 and {args.runs}.')
        selected = list(range(start_run_arg, args.runs + 1))
    else:
        selected = None




    keep_previous = (manual_selection or resume)
    has_csv = folder_csv.is_file()
    has_parquet = folder_parquet.is_file()
    if keep_previous and has_csv != has_parquet:
        if not resume:
            raise RuntimeError('Cannot preserve runs: CSV and Parquet must both exist '
                               '(or both be absent); checkpoints alone cannot '
                               'reconstruct per-example predictions.')
        print('Incomplete CSV/Parquet pair: rebuilding all runs for this condition.')
        keep_previous = False
        resume = False
    ckpt_ids = _existing_run_ids(folder)
    if keep_previous and has_csv:
        old_predictions, old_config = load_parquet_with_config(folder_parquet)
        existing_csv = pd.read_csv(folder_csv)
        pred_ids, incomplete_pairs = _prediction_pairs(old_predictions)
        if incomplete_pairs:
            print(f'WARNING: Parquet has incomplete prediction pairs for '
                  f'runs {sorted(incomplete_pairs)}; selected runs may repair them.')
        print(f'Existing folder: {len(existing_csv)} CSV records, '
              f'checkpoints={sorted(ckpt_ids)}, '
              f'Parquet prediction pairs={sorted(pred_ids)}, '
              f'metadata total_runs={old_config.get("total_runs", "absent")} '
              '(informational only).')
        if not existing_csv.empty:
            expected_loss = {'erm': 'ce', 'erm-lr-matched': 'ce', 'rg': 'rg',
                'rg-mean-one': 'rg-mean-one', 'kd': 'kd',
                'spl': 'ce', 'mentornet-pd': 'ce'}[method]
            if (not (existing_csv['name'] == folder.name).all()
                    or not (existing_csv['model'] == args.model).all()
                    or not (existing_csv['loss'] == expected_loss).all()
                    or not np.allclose(existing_csv['noise_ratio'], args.noise)):
                raise RuntimeError('Existing CSV does not match this condition.')
            if method in ('rg', 'rg-mean-one', 'kd'):
                if not np.allclose(existing_csv['guide_noise_ratio'], args.guide_noise):
                    raise RuntimeError('Existing CSV guide noise differs.')
            if {'run_id', 'seed', 'lr', 'selected_trial'} <= set(existing_csv.columns):
                if not existing_csv['run_id'].notna().all():
                    raise RuntimeError('Existing CSV has incomplete run IDs.')
                if not np.allclose(existing_csv['seed'], BASE_SEED + existing_csv['run_id']):
                    raise RuntimeError('Existing CSV run IDs and seeds disagree.')
                saved_trials = existing_csv['selected_trial']
                if selected_trial_number is None:
                    same_trial = saved_trials.isna().all()
                else:
                    same_trial = (saved_trials.notna().all() and
                                  np.allclose(saved_trials, selected_trial_number))
                if not same_trial:
                    raise RuntimeError('Selected Optuna trial changed; cannot reuse existing runs.')
                if method == 'mentornet-pd':
                    for key in ('mnpd_loss_p_percentile', 'mnpd_burn_in_pct',
                                'mnpd_moving_average_decay'):
                        if (key not in existing_csv or
                                not np.allclose(existing_csv[key], params[key])):
                            raise RuntimeError(f'Existing MentorNet-PD runs differ in {key}.')
                actual_lr = params['lr']
                if not np.allclose(existing_csv['lr'], actual_lr):
                    raise RuntimeError('Existing runs used a different optimizer LR.')
        if selected is None:


            existing_ids = (set(existing_csv['run_id'].astype(int))
                            if 'run_id' in existing_csv.columns
                            and existing_csv['run_id'].notna().all()
                            else set(range(1, len(existing_csv) + 1)))
            completed = {i for i in range(1, args.runs + 1)
                         if i in ckpt_ids and i in pred_ids and i in existing_ids}
            selected = sorted(set(range(1, args.runs + 1)) - completed) if resume else list(range(1, args.runs + 1))
        rows, csv_row_by_run, unverified_runs = _align_existing_csv(
            existing_csv, old_predictions, selected)
        predictions = old_predictions.copy()
        config = {'total_runs': max(len(rows), *(pred_ids or {0}), *(ckpt_ids or {0}))}
        if unverified_runs:
            print(f'WARNING: unrelated existing run IDs could not be matched '
                  f'to CSV rows: {sorted(unverified_runs)}. Their records, '
                  'predictions and checkpoints will be left untouched.')
        for run_id in selected:
            if run_id in csv_row_by_run and csv_row_by_run[run_id] != run_id - 1:
                print(f'INFO: selected r{run_id} matches existing CSV row '
                      f'{csv_row_by_run[run_id] + 1} (not row {run_id}).')
        if len(rows) > args.runs:
            print(f'INFO: preserving all {len(rows)} existing CSV runs '
                  f'(requested target --runs={args.runs}).')
        if old_config.get('total_runs') != len(rows):
            print('WARNING: Parquet total_runs disagrees with observed CSV '
                  'records; metadata will be refreshed after successful rerun.')

    cols = ['name', 'model', 'type', 'state_loaded', 'loss', 'noise_ratio',
            'guide_model', 'guide_noise_ratio', 'loss-value', 'log-loss',
            'accuracy', 'precision', 'recall', 'balanced_accuracy', 'mcc',
            'cohen_kappa', 'auc_roc', 'pr_auc', 'f1_score']
    metadata_cols = ['run_id', 'seed', 'lr', 'lambda', 'delta', 'gamma',
                     'selected_trial', 'best_epoch']
    if method == 'mentornet-pd':
        metadata_cols.extend(('mnpd_loss_p_percentile', 'mnpd_burn_in_pct',
                              'mnpd_moving_average_decay'))
    standard_cols = list(cols)
    if rows:

        cols = list(dict.fromkeys([*existing_csv.columns.tolist(), *cols]))
        rows = [{k: r.get(k) for k in cols} for r in rows]

    cols = list(dict.fromkeys([*cols, *metadata_cols]))

    if selected is None:
        selected = list(range(len(rows) + 1, args.runs + 1)) if resume else list(range(1, args.runs + 1))
    if not (keep_previous and has_csv):
        csv_row_by_run = {}
        unverified_runs = set()
    if not selected:
        _refresh_checkpoint_aliases(folder, rows, csv_row_by_run)
        print(f'No pending runs among requested r1..r{args.runs}; '
              f'{len(rows)} existing CSV records preserved: {folder_csv}')
        return




    existing_count = len(rows)
    available = set(range(1, existing_count + 1)) | set(selected)
    if any(i not in available for i in range(1, max(max(selected), existing_count) + 1)):
        missing = [i for i in range(1, max(max(selected), existing_count) + 1) if i not in available]
        raise RuntimeError(f'Run selection would leave CSV/Parquet gaps at {missing}. '
                           'Include missing runs in --run-ids or use --start-run.')



    retained = set(range(1, existing_count + 1)) - set(selected)
    missing_ckpts = [i for i in sorted(retained) if i not in ckpt_ids]
    pairs_now, _ = _prediction_pairs(predictions)
    missing_prediction_pairs = sorted(retained - pairs_now)
    if missing_ckpts or missing_prediction_pairs:
        print('WARNING: non-selected runs have incomplete artifacts; '
              f'missing checkpoints={missing_ckpts}, '
              f'missing predictions={missing_prediction_pairs}. '
              'Their existing CSV rows and available files will be preserved.')
    print(f'Final-fit destination for {method}: {folder.resolve()}')
    print(f'Runs to execute: {selected} (seeds {[BASE_SEED + i for i in selected]})')
    replace_existing = not keep_previous
    if replace_existing:
        print('New final fit will replace previous CSV/Parquet/checkpoints '
              'after the first run succeeds. optuna_study.db is preserved.')
    ds = make_scenario_data(condition, args.device) if condition is not None else make_data(
        args.noise, method, args.guide_noise, args.device)
    for run_number in selected:
        seed = BASE_SEED + run_number
        tmp_name = f'.best-model_r{run_number}.running.pt'
        tmp_path = folder / tmp_name
        if tmp_path.exists():
            tmp_path.unlink()
        final_ckpt = folder / f'best-model_r{run_number}.pt'
        started = time.monotonic()
        print(f'{method} run {run_number}/{max(args.runs, len(rows), max(selected))}, seed {seed}')
        try:
            result = run_single_experiment(ds, args.model, arch, method, params, args, seed, mode='fit',
                                save=folder, checkpoint_filename=tmp_name,
                                return_details=True)
            seconds = time.monotonic() - started
            yt, yp, proba = result['y_true'], result['y_pred'], result['y_probs']
            val_objective = result['val_objective']
            try:
                auc = roc_auc_score(yt, proba, multi_class='ovr', average='macro')
            except ValueError:
                auc = float('nan')
            try:
                pr = average_precision_score(np.eye(10)[yt], proba,
                                             average='macro')
            except ValueError:
                pr = float('nan')
            try:
                ll = log_loss(yt, proba)
            except ValueError:
                ll = float('nan')
            label = {'erm': 'ce', 'erm-lr-matched': 'ce', 'rg': 'rg',
                     'rg-mean-one': 'rg-mean-one', 'kd': 'kd',
                     'spl': 'ce', 'mentornet-pd': 'ce'}[method]
            is_guided = method_spec(method).guide
            row = dict(zip(standard_cols, [
                folder.name, args.model,
                (condition.metadata_type if condition is not None and condition.metadata_type
                 else method_spec(method).result_type()),
                False, label, args.noise,
                (condition.metadata_guide if condition is not None and condition.metadata_guide
                 else ('ResNet20' if is_guided else None)),
                (args.guide_noise if is_guided else None), val_objective, ll,
                accuracy_score(yt, yp),
                precision_score(yt, yp, average='macro', zero_division=0),
                recall_score(yt, yp, average='macro', zero_division=0),
                balanced_accuracy_score(yt, yp), matthews_corrcoef(yt, yp),
                cohen_kappa_score(yt, yp), auc, pr,
                f1_score(yt, yp, average='macro', zero_division=0),
            ]))
            row.update({
                'run_id': run_number, 'seed': seed,
                'lr': params['lr'],
                'lambda': (params.get('lamb') if method in
                           ('rg', 'rg-mean-one')
                           else None),
                'delta': (params.get('delta') if method in
                          ('rg', 'rg-mean-one') else None),
                'gamma': (params.get('margin') if method in
                          ('rg', 'rg-mean-one') else None),
                'selected_trial': selected_trial_number,
                'best_epoch': int(result['best_epoch']),
                'mnpd_loss_p_percentile': params.get('mnpd_loss_p_percentile') if method == 'mentornet-pd' else None,
                'mnpd_burn_in_pct': params.get('mnpd_burn_in_pct') if method == 'mentornet-pd' else None,
                'mnpd_moving_average_decay': params.get('mnpd_moving_average_decay') if method == 'mentornet-pd' else None,
            })
            if 'train_time' in cols:
                row['train_time'] = seconds
            if 'epochs' in cols:
                row['epochs'] = result['epochs']
            if 'time_per_epoch' in cols:
                row['time_per_epoch'] = seconds / max(1, result['epochs'])
            if replace_existing:


                for old_file in [folder_csv, folder_parquet, folder/'best-model.pt',
                                 folder/'median-model.pt',
                                 *folder.glob('best-model_r[0-9]*.pt')]:
                    if old_file.is_file():
                        old_file.unlink()
                replace_existing = False
            os.replace(tmp_path, final_ckpt)
            if run_number in csv_row_by_run:
                rows[csv_row_by_run[run_number]] = row
            elif run_number == len(rows) + 1:
                csv_row_by_run[run_number] = len(rows)
                rows.append(row)
            elif run_number <= len(rows):

                raise RuntimeError(f'No verified CSV record for selected r{run_number}.')
            else:
                raise RuntimeError('Internal error: non-contiguous run selection.')
            predictions[f'y_true_r{run_number}'] = yt
            predictions[f'y_pred_r{run_number}'] = yp


            complete_pairs, _ = _prediction_pairs(predictions)
            config = {'total_runs': max(len(rows), *(complete_pairs or {0}),
                                        *(_existing_run_ids(folder) or {0}))}
            temp_csv = folder / '.experiment-results.rerun.tmp.csv'
            temp_parquet = folder / '.experiment_data.rerun.tmp.parquet'
            pd.DataFrame(rows, columns=cols).to_csv(temp_csv, index=False)
            save_parquet_with_config(predictions, temp_parquet, config)
            os.replace(temp_csv, folder_csv)
            os.replace(temp_parquet, folder_parquet)
            if condition is not None and condition.guide_kind == 'crossfit':
                accuracy = getattr(ds, 'fold_accuracies', None)
                if accuracy is not None:
                    metrics_file = folder / 'guide_accuracy.json'
                    temp_metrics = folder / '.guide_accuracy.tmp.json'
                    temp_metrics.write_text(json.dumps(accuracy, indent=2), encoding='utf-8')
                    os.replace(temp_metrics, metrics_file)
            print(f'Seed {seed} completed. Accuracy {row["accuracy"]:.4%}, '
                  f'best validation objective {val_objective:.6f}')
        except Exception:
            (folder / 'error.txt').write_text(traceback.format_exc())
            print(f'FAILED seed {seed}. No failed checkpoint promoted.')
            raise




    _refresh_checkpoint_aliases(folder, rows, csv_row_by_run)


def run_search(args, method):
    # Search method hyperparameters with a resumable Optuna study.
    if not method_spec(method).tune and not method_spec(method).study_from_rg \
            and not (getattr(args, 'lr_grid', False) and method == 'erm'):
        read_architecture(args.model)
        print('ERM uses the Stage-1 architecture study; no Stage-2 method search.')
        return
    arch = read_architecture(args.model)
    ensure_noise_cache(args.noise)
    study_db = selected_study_path(args.model, args.noise, method, args.guide_noise)
    if method_spec(method).study_from_rg:
        if not study_db.is_file():
            raise FileNotFoundError(f'Missing selected RG study for optimization control {method}: {study_db}')
        study = optuna.load_study(study_name=STUDY_NAME,
                                  storage=f'sqlite:///{study_db.resolve()}')
        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            raise RuntimeError(f'No COMPLETE trials in RG study: {study_db}')
        print(f'{method} reuses selected RG hyperparameters (no LR tuning): {study_db}')
        return
    folder = output_dir(args.model, args.noise, method, args.guide_noise)
    folder.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=_study_name(method), direction='minimize',
        sampler=TPESampler(seed=BASE_SEED, n_startup_trials=5,
                           multivariate=True, group=True),
        storage=f'sqlite:///{study_db.resolve()}', load_if_exists=True)
    # ERM has only three distinct Stage-2 LR settings.
    # MentorNet-PD now tunes the PD curriculum as well and keeps the TPE search.
    finite_lr_only = bool(getattr(args, 'lr_grid', False)) and method == 'erm'
    if finite_lr_only:
        lr_options, remaining = _finite_lr_trials(study, LR_GRID, args.trials)
    else:
        remaining = _remaining_optuna_trials(study, args.trials)
    if not remaining:
        print(f'Optuna search already complete: {study_db}')
    else:
        ds = make_data(args.noise, method, args.guide_noise, args.device)
        args.epochs = args.search_epochs

        def objective(trial):
            params = sample_params(trial, method, args)
            try:


                seed = BASE_SEED + _trial_seed_number(trial)
                outcome = run_single_experiment(ds, args.model, arch, method, params,
                                     args, seed, mode='search', trial_number=trial.number)
                if method == 'mentornet-pd':
                    for epoch, value in enumerate(outcome.get('validation_ce_per_epoch',
                                                              [outcome['validation_ce']])):
                        trial.report(float(value), epoch)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                return outcome['validation_ce']
            except KeyboardInterrupt:
                _mark_interrupted(trial)
                raise
            except (RuntimeError, FloatingPointError) as exc:
                trial.set_user_attr('failure', repr(exc))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise

        if finite_lr_only:
            _optimize_finite_lr(study, objective, lr_options, remaining,
                                catch=(RuntimeError, FloatingPointError))
        else:
            study.optimize(objective, n_trials=remaining,
                           catch=(RuntimeError, FloatingPointError), n_jobs=1)
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise RuntimeError(f'No COMPLETE Optuna trials in {study_db}')
    print(f'Search complete: {len(study.trials)} trials, best trial {study.best_trial.number}')




def read_stage_one_parameters(model):
    # Return the selected Stage-1 parameters also for ResNet20 guide folds.
    if model in ('ResNet20', 'MLP-guide'):
        return {}
    study_path = architecture_dir(model) / 'optuna_study.db'
    if not study_path.is_file():
        raise FileNotFoundError(f'Missing architecture study: {study_path}')
    study = optuna.load_study(study_name=STUDY_NAME,
                              storage=f'sqlite:///{study_path.resolve()}')
    return architecture_from_study(model, study.best_params)


def scenario_parameters(condition, *, return_selected_trial=False):
    # Resolve parameters and the source Optuna trial, when applicable.
    selected_trial = None
    spec = method_spec(condition.method)
    if condition.parameter_source in ('fixed', 'architecture'):
        params = {**spec.fixed_parameters(), **dict(condition.fixed_parameters)}
        if condition.lr_grid and condition.scenario == 'ablations':


            rg_db = method_spec('rg').study_path(
                condition.model, condition.noise, condition.noise)
            if not rg_db.is_file():
                raise FileNotFoundError(f'Missing matching RG study: {rg_db}')
            rg_study = optuna.load_study(study_name=STUDY_NAME,
                storage=f'sqlite:///{rg_db.resolve()}')
            params['lr'] = rg_study.best_trial.params['lr']
            selected_trial = rg_study.best_trial.number
        elif condition.lr_grid:
            path = Path(condition.directory) / 'optuna_study.db'
            study = optuna.load_study(study_name=STUDY_NAME,
                storage=f'sqlite:///{path.resolve()}')
            params['lr'] = study.best_trial.params['lr']
            selected_trial = study.best_trial.number
        return (params, selected_trial) if return_selected_trial else params
    if condition.parameter_source != 'rg-study':
        raise ValueError(f'Unknown parameter source: {condition.parameter_source}')
    study_path = method_spec('rg').study_path(
        condition.model, condition.noise,
        condition.guide_noise if condition.parameter_guide_noise is None
        else condition.parameter_guide_noise)
    if not study_path.is_file():
        raise FileNotFoundError(f'Missing RG Optuna study: {study_path}')
    study = optuna.load_study(study_name=STUDY_NAME,
                              storage=f'sqlite:///{study_path.resolve()}')
    chosen = dict(study.best_params)
    from math import isclose
    for key, expected in condition.expected_parameters:
        if not isclose(chosen[key], expected, rel_tol=1e-9):
            raise ValueError(f'{condition.scenario} {condition.model}/{condition.noise}: '
                             f'{key}={chosen[key]} differs from {expected}')
    params = {**spec.fixed_parameters(), **chosen}
    return (params, study.best_trial.number) if return_selected_trial else params


def scenario_guide_paths(noise):
    return [crossfit_guide_dir(noise, fold) / 'best-model.pt' for fold in (0, 1)]


def make_scenario_data(condition, device):
    # Dataset adaptations only; common runner handles training and persistence.
    if condition.guide_kind == 'crossfit':
        from core.crossfit_data import CrossFitDatasetBuilder
        builder = CrossFitDatasetBuilder(
            [str(path) for path in scenario_guide_paths(condition.guide_noise)],
            noise_ratio=condition.noise, seed=condition.seed,
            baseline_method='ResNet20', baseline_noise_ratio=condition.guide_noise,
            device=device)
    else:
        guide = {'resnet': 'ResNet20', 'mlp': 'MLP-guide'}.get(condition.guide_kind, 'no')
        builder = DatasetBuilder(noise_ratio=condition.noise,
            baseline_noise_ratio=condition.guide_noise, baseline_method=guide,
            seed=condition.seed, device=device)
    builder.build()
    if condition.guide_kind == 'fold':
        from torch.utils.data import Subset
        from core.crossfit_data import crossfit_folds
        folds = crossfit_folds(len(builder.train_dataset), seed=condition.seed, k=2)
        builder.train_dataset = Subset(builder.train_dataset,
                                       np.where(folds == condition.fold)[0])
    return builder


def run_scenario_lr_search(condition, *, device, quiet, trials, search_epochs):
    # Select LR for a fixed scenario using validation loss.
    from types import SimpleNamespace
    if condition.parameter_source not in ('fixed', 'architecture'):
        return
    folder = Path(condition.directory)
    folder.mkdir(parents=True, exist_ok=True)
    db = folder / 'optuna_study.db'
    if db.is_file():
        study = optuna.load_study(study_name=STUDY_NAME,
                                  storage=f'sqlite:///{db.resolve()}')
    else:
        study = optuna.create_study(study_name=STUDY_NAME, direction='minimize',
            sampler=TPESampler(seed=BASE_SEED, n_startup_trials=5,
                               multivariate=True, group=True),
            storage=f'sqlite:///{db.resolve()}', load_if_exists=True)
    lr_options, remaining = _finite_lr_trials(study, LR_GRID, trials)
    if remaining:
        arch = (read_stage_one_parameters(condition.model)
                if condition.parameter_source == 'architecture'
                else read_architecture(condition.model))
        ds = make_scenario_data(condition, device)
        args = SimpleNamespace(device=device, quiet=quiet, epochs=search_epochs,
            lr_grid=True)
        fixed = {**method_spec(condition.method).fixed_parameters(),
                 **dict(condition.fixed_parameters)}
        def objective(trial):
            try:
                params = dict(fixed, lr=trial.suggest_categorical('lr', list(LR_GRID)))
                return run_single_experiment(ds, condition.model, arch, condition.method, params,
                                  args, BASE_SEED + _trial_seed_number(trial),
                                  mode='search', trial_number=trial.number)['validation_ce']
            except KeyboardInterrupt:
                _mark_interrupted(trial)
                raise
        _optimize_finite_lr(study, objective, lr_options, remaining,
                            catch=(RuntimeError, FloatingPointError))
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise RuntimeError(f'No successful LR search trial for {condition.directory}')
    print(f'Scenario LR selected: {condition.directory}: {study.best_trial.params["lr"]}')


def run_scenario(conditions, *, device=None, quiet=False, lr_grid=False,
                 trials=TRIAL_BUDGET, search_epochs=300, resume=False):
    # Execute all fixed conditions via exactly the same run_final_fit engine.
    from types import SimpleNamespace
    conditions = tuple(conditions)
    if not conditions:
        raise ValueError('No scenario conditions supplied')
    if lr_grid:



        conditions = tuple(replace(c, expected_parameters=(),
            lr_grid=(c.scenario not in ('erm-lr-sweep', 'crossfit-guides', 'ablations')))
                           for c in conditions)
        conditions = tuple(replace(c, lr_grid=True) if c.scenario == 'ablations'
                           else c for c in conditions)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    for condition in conditions:
        if lr_grid and condition.lr_grid and condition.scenario not in ('ablations',):
            run_scenario_lr_search(condition, device=device, quiet=quiet,
                trials=trials, search_epochs=search_epochs)
        args = SimpleNamespace(model=condition.model, noise=condition.noise,
            guide_noise=condition.guide_noise, method=condition.method,
            stage='fit', runs=condition.runs, fit_epochs=300, device=device,
            quiet=quiet, resume=resume,
            start_run=None, run_ids=None, lr_grid=lr_grid,
            )
        print(f'Scenario {condition.scenario}: {condition.directory} '
              f'(seed {condition.seed}+run_id, no backup)')
        run_final_fit(args, condition.method, condition=condition)

def run(args, spec=None):
    # Run configured search and/or fit through a single execution engine.
    selected = spec or method_spec(args.method)
    if selected.name != args.method:
        raise ValueError('Method specification must match CLI arguments.')
    if args.stage in ('search', 'all'):
        run_search(args, selected.name)
    if args.stage in ('fit', 'all'):
        return run_final_fit(args, selected.name)
    if args.stage not in ('search', 'fit', 'all'):
        raise ValueError(f'Unsupported stage: {args.stage}')
