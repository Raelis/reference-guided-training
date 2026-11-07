# Evaluate CSV results. Default: original raw CSV concatenation.

# --detailed: per-run detailed analysis (checkpoint-dependent).
# --stats: per-experiment CSV statistics.

import argparse
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


from core.evaluation_summary import find_csv_files, load_and_concat, save_condition_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', default='results')
    parser.add_argument('--out', default='results/processed_results/experiments_results_summary.csv')
    parser.add_argument('--detailed', action='store_true')
    parser.add_argument('--stats', action='store_true', help='Write an additional per-condition statistics CSV')
    parser.add_argument('--stats-out', default='results/processed_results/experiments_detailed_summary.csv')
    args = parser.parse_args()

    csv_paths = find_csv_files(args.results_root)
    print(f'Found {len(csv_paths)} CSV files')
    df = load_and_concat(csv_paths)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f'Saved in: {out}')
    print(f'Final shape: {df.shape}')

    if args.stats or args.detailed:
        save_condition_stats(args.results_root, args.stats_out)


if __name__ == '__main__':
    main()
