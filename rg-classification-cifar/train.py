"""CIFAR-10N categorical-LR protocol: 3 scenarios, ResNet56 guides, 5 runs."""
import argparse
import os
from pathlib import Path

from core.experiment_protocol import LR_CHOICES, PROTOCOL_ID, SCENARIOS, STAGE1_LR

ROOT = Path(__file__).resolve().parent


from core.experiment_plan import build_plan, protocol_declaration
from core.experiment_plan import prepare_output as _prepare_output


def prepare_output(action, model, noise, guide_noise, args):
    return _prepare_output(action, model, noise, guide_noise, args, root=ROOT)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["architecture", "guide", "run", "matrix"], default="run")
    p.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="independent")
    p.add_argument("--model", choices=["CNN", "MLP", "Transformer", "ResNet56"], default="CNN")
    p.add_argument("--method", choices=["erm", "rg"], default="rg")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.scenario == "all" and args.stage == "run":
        p.error("--scenario all must use --stage matrix, guide, or architecture")
    if min(args.trials, args.runs, args.epochs) < 1:
        p.error("Trial, run and epoch counts must be positive")
    if args.model == "ResNet56" and args.stage == "run":
        p.error("ResNet56 is a reference guide, not a learner")
    os.chdir(ROOT)
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    plan = build_plan(args.stage, scenarios, args.model, args.method)
    print(f"Protocol: {PROTOCOL_ID}; LR choices: {LR_CHOICES}; Stage-1 fixed LR: {STAGE1_LR}")
    print("Planned:", *plan, sep="\n  ")
    if args.dry_run:
        return 0
    from core.experiment import CIFAR10NExperiment
    for action, model, noise, guide_noise in plan:
        mode, loss = prepare_output(action, model, noise, guide_noise, args)
        exp = CIFAR10NExperiment(
            exp_model_name=model,
            exp_loss_name=loss,
            noise_type=noise,
            exp_guide_model="ResNet56" if action == "rg" else "no",
            guide_noise_type=guide_noise if action == "rg" else "clean",
            total_epochs=args.epochs,
            n_trials=args.trials,
            seed=42,
            run_optuna=True,
        )
        if action == "architecture":
            exp.run_architecture_search()
        else:
            exp.run_training(mode=mode, n_runs=args.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
