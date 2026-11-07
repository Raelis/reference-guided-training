# Evaluation of run CSVs
import csv
import json
import statistics
from collections import defaultdict
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
             'values':[{'seed':s,'accuracy_percent':a,'status':r.get('status','historical-unknown'),
                         'trial_number':r.get('trial_number','not-recorded')} for s,a,r in items]}
        summaries.append(row)
        print(f'{row["condition"]:70} n={len(vals):2} failure={failures} collapse={c} '
              f'mean={row["mean_accuracy_percent"]} sd={row["sd_sample_percent"]}')
        if detailed:
            for value in row['values']:
                print(f'  seed={value["seed"]}  accuracy={value["accuracy_percent"]:.3f} '
                      f'status={value["status"]} trial={value["trial_number"]}')
    return summaries


def save_summary(rows,out):
    p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved audit (no exclusion of collapsed seeds):',p)
