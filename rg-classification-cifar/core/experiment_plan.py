# Experiment protocol matrix and output protocol enforcement.
import csv
import json
from pathlib import Path

from core.experiment_protocol import (
    FIXED_BATCH_SIZE, FIXED_WEIGHT_DECAY, LR_CHOICES, PROTOCOL_ID, SCENARIOS,
    STAGE1_EPOCHS, STAGE1_LR, STAGE1_PATIENCE,
)
from core.experiment_paths import architecture_dir, guide_dir, method_dir


def build_plan(stage, scenarios, model, method):
    # Return the unique CIFAR tasks required by the requested stage.
    plan = []
    if stage == "architecture":
        for scenario in scenarios:
            noise, _ = SCENARIOS[scenario]
            for learner in (["CNN", "MLP", "Transformer"] if model == "ResNet56" else [model]):
                plan.append(("architecture", learner, noise, "none"))
    elif stage == "guide":
        for scenario in scenarios:
            _, guide_noise = SCENARIOS[scenario]
            plan.append(("guide", "ResNet56", guide_noise, "none"))
    elif stage in ("run", "matrix"):
        for scenario in scenarios:
            learner_noise, guide_noise = SCENARIOS[scenario]
            learners = ["CNN", "MLP", "Transformer"] if model == "ResNet56" else [model]
            for learner in learners:
                methods = ("erm", "rg") if stage == "matrix" else (method,)
                for current in methods:
                    plan.append((current, learner, learner_noise,
                                 guide_noise if current == "rg" else "clean"))
    return list(dict.fromkeys(plan))


def protocol_declaration(action, model, noise, guide_noise, args):
    # Describe the settings that must match when resuming a result directory.
    return {
        "protocol": PROTOCOL_ID,
        "model": model,
        "noise": noise,
        "guide_noise": guide_noise if action == "rg" else None,
        "method": action,
        "epochs": args.epochs,
        "trials": args.trials,
        "runs": args.runs,
        "learning_rate_policy": "optuna_direct_no_scaling",
        "lr_choices": list(LR_CHOICES),
        "stage1_lr": STAGE1_LR,
        "stage1_epochs": STAGE1_EPOCHS,
        "stage1_patience": STAGE1_PATIENCE,
        "batch_size": FIXED_BATCH_SIZE,
        "weight_decay": FIXED_WEIGHT_DECAY,
        "optimizer": "sgd_momentum_0.9",
        "scheduler": "cosine_eta_min_1e-5",
        "data_split": "cifar10_first_45000_train_last_5000_clean_validation",
    }


def prepare_output(action, model, noise, guide_noise, args, *, root):
    # Reject incompatible outputs before launching Optuna or final fitting.
    losses = {"architecture": "ce", "guide": "ce", "erm": "ce", "rg": "rg"}
    mode = "state" if action == "architecture" else ("guided" if action == "rg" else "baseline")
    if action == "architecture":
        out = Path(root) / architecture_dir(model, noise)
    elif action == "guide":
        out = Path(root) / guide_dir(noise)
    elif action == "erm":
        out = Path(root) / method_dir(model, noise, "erm")
    else:
        out = Path(root) / method_dir(model, noise, "rg", "ResNet56", guide_noise)
    expected = protocol_declaration(action, model, noise, guide_noise, args)
    manifest = out / "protocol.json"
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding="utf-8")) != expected:
            raise RuntimeError(f"Existing protocol differs: {out}. Use a clean output directory.")
    elif out.exists() and any(out.iterdir()):
        raise RuntimeError(
            f"Existing CIFAR outputs without a matching protocol manifest: {out}. "
            "Move/archive this directory before running the new categorical-LR protocol."
        )
    out.mkdir(parents=True, exist_ok=True)
    if not manifest.is_file():
        manifest.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    if action != "architecture" and (out / "median-model.pt").is_file():
        result_file = out / "experiment-results.csv"
        if not result_file.is_file():
            raise RuntimeError(f"Median checkpoint exists but run CSV is missing: {out}")
        with result_file.open(newline="", encoding="utf-8") as stream:
            recorded = sum(1 for _ in csv.DictReader(stream))
        if recorded < args.runs:
            raise RuntimeError(
                f"Incomplete run: {recorded}/{args.runs} CSV entries but median-model.pt exists: {out}"
            )
    return mode, losses[action]


def verify_dependency_manifest(directory, required):
    # Reject input artifacts from older CIFAR protocols or different runs.
    directory = Path(directory)
    path = directory / "protocol.json"
    if not path.is_file():
        raise RuntimeError(f"Missing dependency protocol manifest: {path}")
    with path.open(encoding="utf-8") as stream:
        declared = json.load(stream)
    for key, value in required.items():
        if declared.get(key) != value:
            raise RuntimeError(f"Incompatible {key} in {path}: "
                               f"expected {value!r}, found {declared.get(key)!r}")
    return declared
