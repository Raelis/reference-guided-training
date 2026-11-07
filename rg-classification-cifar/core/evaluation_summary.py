# Read and export per-condition experimental summary tables.
import os
from pathlib import Path
import pandas as pd

def find_csv_files(root_dir):
    csv_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in ('experiment-results.csv', 'experiments-results.csv'):
                csv_paths.append(os.path.join(dirpath, filename))
    return sorted(csv_paths)


def load_and_concat(csv_paths):
    dfs = []
    for path in csv_paths:
        try:
            dfs.append(pd.read_csv(path))
        except Exception as exc:
            print(f'[WARN] erro ao ler {path}: {exc}')
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def save_condition_stats(root, output):
    """Additional CSV report, no changes to the original concatenated records."""
    from core.evaluation import aggregate
    entries = aggregate(root, detailed=True)
    rows = []
    for x in entries:
        row = {}
        for key, value in x.items():
            if key == 'values':
                continue

            if key.endswith('_percent'):
                row[key.removesuffix('_percent')] = value / 100.0 if value is not None else None
            else:
                row[key] = value
        rows.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print('Detailed CSV:', output)


