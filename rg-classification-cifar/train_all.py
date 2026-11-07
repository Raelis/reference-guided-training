# Run the full CIFAR-10N ERM/RG matrix with fresh, isolated results.

# Re-run this script after an interruption to resume from existing studies/CSV

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = (
    ("architecture: learner random1", ["--stage", "architecture", "--scenario", "independent", "--model", "ResNet56"]),
    ("architecture: learner worst", ["--stage", "architecture", "--scenario", "matched", "--model", "ResNet56"]),
    ("ResNet56 reference guides", ["--stage", "guide", "--scenario", "all"]),
    ("ERM / RG: complete matrix", ["--stage", "matrix", "--scenario", "all", "--model", "ResNet56"]),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Print every planned experiment without GPU")
    parser.add_argument("--resume", action="store_true", help="Resume without removing existing valid compatible outputs")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args(argv)
    results = ROOT / "results"
    if not args.plan and not args.resume and results.exists() and any(p.is_dir() for p in results.iterdir()):
        parser.error("Existing result directories found. Use --resume to continue the current protocol; "
                     "incompatible protocols are rejected by train.py.")
    for label, task in STEPS:
        print(f"\n=== {label} ===", flush=True)
        command = [sys.executable, str(ROOT / "train.py"), *task,
                   "--trials", str(args.trials), "--runs", str(args.runs), "--epochs", str(args.epochs)]
        if args.plan:
            command.append("--dry-run")
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            print(f"Failed step: {label}; use --resume after fixing it.", file=sys.stderr)
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
