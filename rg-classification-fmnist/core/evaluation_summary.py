# Aggregate experiment CSVs and export validation-selected hyperparameters.
import csv
import json
import statistics
import os
import pandas as pd
from pathlib import Path


def aggregate(results_root, *, detailed=False, collapse_threshold=15.0, group_by='condition'):
    root=Path(results_root)
    files=sorted(root.rglob('experiment-results.csv'))
    if not files:
        print('No experiment-results.csv found in',root);return []
    summaries=[]
    for p in files:
        try:
            with p.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        except (OSError, UnicodeError, csv.Error) as e:
            print('[ERROR] cannot read',p,e);continue
        items=[];failures=0
        for index,r in enumerate(rows):
            s=r.get('accuracy')
            try:
                a=float(s)
                if not (float('-inf')<a<float('inf')):raise ValueError('nonfinite')
                if not (0.0 <= a <= 1.0):
                    raise ValueError('accuracy must be a fraction between 0 and 1')
                a *= 100.0
                items.append((r.get('seed',str(index+1)),a,r))
            except (ValueError,TypeError):failures+=1
        vals=[a for _,a,_ in items]
        c=sum(a<collapse_threshold for a in vals)
        row={'source':str(p),'condition':p.parent.name,'n':len(vals),
             'failed_or_missing':failures,'collapsed_below_threshold':c,
             'mean_accuracy_percent':round(statistics.mean(vals),5) if vals else None,
             'sd_sample_percent':round(statistics.stdev(vals),5) if len(vals)>1 else None,
             'sd_population_percent':round(statistics.pstdev(vals),5) if len(vals)>1 else None,
             'median_accuracy_percent':round(statistics.median(vals),5) if vals else None,
             'min_accuracy_percent':min(vals) if vals else None,
             'max_accuracy_percent':max(vals) if vals else None,
             'values':[{'seed':s,'accuracy_percent':a,'status':r.get('status','not-recorded'),
                         'trial_number':r.get('trial_number','not-recorded')} for s,a,r in items]}
        summaries.append(row)
        print(f'{row["condition"]:70} n={len(vals):2} failure={failures} collapse={c} '
              f'mean={row["mean_accuracy_percent"]} sd={row["sd_sample_percent"]}')
        if detailed:
            for value in row['values']:
                print(f'  seed={value["seed"]}  accuracy={value["accuracy_percent"]:.3f} '
                      f'status={value["status"]} trial={value["trial_number"]}')
    return summaries


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
            print(f'[WARN] Unable to read {path}: {exc}')
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def save_condition_stats(root, output):
    """Write per-condition accuracy statistics in CSV format."""
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




def export_selected_hyperparameters(results_root='results'):
    # Export best Optuna trials.
    import csv
    import json
    import optuna
    from optuna.trial import TrialState

    rows = []
    for path in sorted(Path(results_root).rglob('optuna_study.db')):
        from .experiment_runner import STUDY_NAME, MENTOR_PD_STUDY_NAME
        is_pd = path.parent.name.endswith('_mentornet-pd')
        name = MENTOR_PD_STUDY_NAME if is_pd else STUDY_NAME
        try:
            study = optuna.load_study(study_name=name,
                                      storage=f'sqlite:///{path.resolve()}')
        except KeyError:
            # A LR-only MentorNet study is not a tuned PD curriculum.
            continue
        complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not complete:
            continue
        row = {'condition': str(path.parent), 'n_trials': len(study.trials),
               'n_complete': len(complete), 'best_trial': study.best_trial.number,
               'best_validation_ce': study.best_value,
               'best_lr': study.best_trial.params.get('lr'),
               'best_params_json': json.dumps(study.best_trial.params,sort_keys=True)}
        rows.append(row)
    output = Path('results/processed_results/selected_hyperparameters.csv')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8') as file:
        writer = csv.DictWriter(file,fieldnames=['condition','n_trials','n_complete',
            'best_trial','best_validation_ce','best_lr','best_params_json'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Exported {len(rows)} Optuna studies to {output}')


