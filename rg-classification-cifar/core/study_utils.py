# Optuna study diagnostics and architecture-space inspection.
import numpy as np
import optuna
from core.experiment_protocol import EDGE_WARNING_FRACTION

def print_search_space(label, space):
    print(f"[search space] {label}")
    for name, spec in space.items():
        if spec[0] == "categorical":
            print(f"    {name}: categorical in {list(spec[1])}")
        else:
            print(f"    {name}: {spec[0]} in [{spec[1]}, {spec[2]}]")

def warn_on_edges(best_params, space):
    for name, spec in space.items():
        if name not in best_params:
            continue
        kind = spec[0]
        if kind == "categorical":
            continue
        low, high = spec[1], spec[2]
        value = best_params[name]
        if kind == "log-uniform":
            span = np.log(high) - np.log(low)
            lower_gap = (np.log(value) - np.log(low)) / span
            upper_gap = (np.log(high) - np.log(value)) / span
        else:
            span = high - low
            lower_gap = (value - low) / span
            upper_gap = (high - value) / span
        if lower_gap < EDGE_WARNING_FRACTION:
            print(
                f"[WARNING] selected {name}={value:.6g} sits within "
                f"{EDGE_WARNING_FRACTION:.0%} of the lower bound {low:.6g}. "
                f"The optimum may lie outside the searched range."
            )
        if upper_gap < EDGE_WARNING_FRACTION:
            print(
                f"[WARNING] selected {name}={value:.6g} sits within "
                f"{EDGE_WARNING_FRACTION:.0%} of the upper bound {high:.6g}. "
                f"The optimum may lie outside the searched range."
            )

def assert_study_has_result(study, experiment_dir):
    # Fail when every trial died, instead of on Optuna's "Record does not exist".

    # study.optimize() is called with catch=(RuntimeError, FloatingPointError) so a
    # single bad trial does not abort a long search. CUDA out-of-memory is a
    # RuntimeError subclass, so running several training scripts against the same
    # GPU makes every trial fail silently and the study finishes with nothing.
    
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None and np.isfinite(t.value)]
    if completed:
        return

    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    raise RuntimeError(
        f"No completed finite-valued trial in {experiment_dir} "
        f"({n_failed} FAILED, {n_pruned} PRUNED, {len(study.trials)} total). "
        "Inspect the study's trial history and failure logs before rerunning."
    )


class _NullTrial:
    # Minimal stand-in used to detect whether a model has an architecture space.

    # prepare_trial() is called with this object purely to see whether it returns
    # an empty dict; every suggestion method returns the lower bound so that the
    # call is side-effect free.

    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_int(self, name, low, high, step=1, log=False):
        return low

    def suggest_float(self, name, low, high, step=None, log=False):
        return low
