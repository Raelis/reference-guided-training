# Execute the full Fashion-MNIST experiment plan with resumable steps.
from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Tuple

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MANIFEST = RESULTS / 'experiment_manifest.json'
LOG_DIR = RESULTS / 'execution_logs'
MODELS = ('CNN', 'MLP', 'Transformer')
NOISES = ('0.0', '0.4', '0.8')
METHODS = ('erm', 'rg', 'kd', 'spl', 'mentornet-pd')
CONTROLS = ('erm-lr-matched', 'rg-mean-one')
PROTOCOL = 'fashion-mnist-default'


@dataclass(frozen=True)
class Step:
    name: str
    command: Tuple[str, ...]
    depends_on: Tuple[str, ...] = ()
    refresh_after: Tuple[str, ...] = ()
    artifacts: Tuple[str, ...] = ()
    expected_runs: int = 0


def step_outputs(name, runs):
    """Return the files that establish successful completion of a workflow step."""
    from core.experiment_paths import (
        architecture_dir, method_dir, crossfit_guide_dir,
        crossfit_learner_dir, ablation_dir,
    )
    parts = name.split(':')
    group = parts[0]
    if group == 'mentor-pretrain':
        return ('results/mentornet-pd-pretraining/mentornet.pt',
                'results/mentornet-pd-pretraining/training_metadata.json'), 0
    if group == 'architecture':
        if parts[1] in ('ResNet20', 'MLP-guide'):
            return (), 0
        folder = architecture_dir(parts[1])
        return (str(folder / 'optuna_study.db'),), 0
    if group == 'guide':
        folder = method_dir(parts[1], float(parts[2]), 'erm')
        study = False
        count = runs
    elif group == 'main':
        _, model, noise, method = parts
        folder = method_dir(model, float(noise), method,
                            'ResNet20' if method in ('rg', 'kd') else None,
                            float(noise))
        study, count = True, runs
    elif group == 'control':
        _, model, noise, method = parts
        folder = method_dir(model, float(noise), method,
                            'ResNet20' if method == 'rg-mean-one' else None,
                            float(noise))
        study, count = False, runs
    elif group == 'sweep':
        _, model, noise, guide = parts
        folder = method_dir(model, float(noise), 'rg', 'ResNet20',
                            float(guide.removeprefix('guide')))
        study = not (model == 'CNN' and noise == '0.4')
        count = runs
    elif group == 'crossfit-guide':
        folder = crossfit_guide_dir(float(parts[1]), int(parts[2].removeprefix('fold')))
        study, count = False, 1
    elif group == 'crossfit':
        folder = crossfit_learner_dir(parts[1], float(parts[2]))
        study, count = False, runs
    elif group == 'ablation':
        index = int(parts[1])
        folder = ablation_dir(index, 'MLP-guide' if index == 7 else 'ResNet20')
        study, count = False, runs
    else:
        outputs = {
            'summary': ('results/processed_results/experiments_detailed_summary.csv',),
            'selected-parameters': ('results/processed_results/selected_hyperparameters.csv',),
            'diagnostic:guide-memorization': ('results/diagnostics/guide_memorization.csv',),
            'diagnostic:rg-weight-allocation': ('results/diagnostics/rg_training_weight_allocation.csv',),
        }
        if group == 'error-distributions':
            file_tag = {'CNN': 'cnn', 'Transformer': 'vit'}[parts[1]]
            return (f'figures/fig_global_error_dist_{file_tag}.pdf',
                    f'figures/fig_guide_miss_error_dist_{file_tag}.pdf',
                    f'figures/{file_tag}_40noise_alignment_per_run.csv',
                    f'figures/{file_tag}_40noise_alignment_summary.csv'), 0
        return outputs.get(name, ()), 0
    outputs = [folder / 'experiment-results.csv',
               folder / 'experiment_data.parquet',
               folder / 'best-model.pt', folder / 'median-model.pt']
    outputs.extend(folder / f'best-model_r{i}.pt' for i in range(1, count + 1))
    if study:
        outputs.append(folder / 'optuna_study.db')
    return tuple(map(str, outputs)), count


def artifacts_valid(step):
    """Verify expected outputs and their run identifiers before skipping a step."""
    if step.name == 'mentor-pretrain':
        try:
            metadata = json.loads((ROOT / 'results/mentornet-pd-pretraining/training_metadata.json')
                                  .read_text(encoding='utf-8'))
            if (metadata['training_steps'] != 30000
                    or metadata['sample_size'] != 100000
                    or metadata['seed'] != 42):
                return False
        except (FileNotFoundError, OSError, ValueError, KeyError, TypeError):
            return False
    for file in step.artifacts:
        path = ROOT / file
        if not path.is_file() or path.stat().st_size == 0:
            return False
    if step.expected_runs:
        csv_paths = [ROOT / f for f in step.artifacts
                     if f.endswith('/experiment-results.csv')]
        if len(csv_paths) != 1:
            return False
        try:
            with csv_paths[0].open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            ids = [int(row['run_id']) for row in rows]
            seeds = [int(row['seed']) for row in rows]
        except (OSError, KeyError, TypeError, ValueError, csv.Error):
            return False
        if sorted(ids) != list(range(1, step.expected_runs + 1)):
            return False
        if any(seed != 42 + run_id for seed, run_id in zip(seeds, ids)):
            return False
        if step.name.startswith('main:') and step.name.endswith(':mentornet-pd'):
            for row in rows: 
                mnpd_ranges = {
                    'mnpd_loss_p_percentile': (0.50, 0.95),
                    'mnpd_burn_in_pct': (0.0, 40.0),
                    'mnpd_moving_average_decay': (0.10, 0.90),
                }
                for key, (minimum, maximum) in mnpd_ranges.items():
                    try:
                        value = float(row[key])
                    except (KeyError, ValueError, TypeError):
                        return False
                    if not (minimum <= value <= maximum):
                        return False

    return True


def artifact_signature(step):
    """Fingerprint output metadata to invalidate reports after source changes."""
    if not step.artifacts:
        return None
    records = []
    for file in step.artifacts:
        path = ROOT / file
        if path.is_file():
            stat = path.stat()
            records.append((file, stat.st_size, stat.st_mtime_ns))
        else:
            records.append((file, None, None))
    return hashlib.sha256(json.dumps(records).encode('utf-8')).hexdigest()


def plan(args):
    """A dependency-ordered execution DAG; each scenario condition is isolated."""


    common = ('--trials', str(args.trials),
              '--runs', str(args.runs), '--search-epochs', str(args.search_epochs),
              '--fit-epochs', str(args.fit_epochs), '--device', args.device)
    guide_flags = ('--runs', str(args.runs),
                   '--fit-epochs', str(args.fit_epochs), '--device', args.device)
    if args.quiet:
        common += ('--quiet',)
        guide_flags += ('--quiet',)
    steps = []
    def add(name, cmd, deps=(), *, program='train.py', stage1=False, refresh_after=()):
        path = str(ROOT / program)
        flags = (() if stage1 or program != 'train.py'
                 else guide_flags if name.startswith('guide:') else common)
        command = (sys.executable, path, *cmd, *flags)
        artifacts, expected_runs = step_outputs(name, args.runs)
        steps.append(Step(name, command, tuple(deps), tuple(refresh_after),
                          artifacts, expected_runs))

    for model in MODELS:
        add(f'architecture:{model}', ('--stage', 'architecture', '--model', model,
            '--noise', '0.0', '--device', args.device), stage1=True)

    for noise in NOISES:
        add(f'guide:ResNet20:{noise}', ('--stage', 'guide', '--model', 'ResNet20',
            '--noise', noise))
    add('guide:MLP-guide:0.0', ('--stage', 'guide', '--model', 'MLP-guide',
        '--noise', '0.0'))

    add('mentor-pretrain', ('--stage', 'mentornet-pretrain'),
        program='train.py')

    for model in MODELS:
        for noise in NOISES:
            for method in METHODS:
                deps = [f'architecture:{model}']
                if method in ('rg', 'kd'):
                    deps.append(f'guide:ResNet20:{noise}')
                if method == 'mentornet-pd':
                    deps.append('mentor-pretrain')
                add(f'main:{model}:{noise}:{method}', ('--stage', 'all',
                    '--model', model, '--noise', noise, '--guide-noise', noise,
                    '--method', method), deps)



    for model in MODELS:
        for noise in NOISES:
            for method in CONTROLS:
                add(f'control:{model}:{noise}:{method}', ('--stage', 'fit',
                    '--model', model, '--noise', noise, '--guide-noise', noise,
                    '--method', method), (f'main:{model}:{noise}:rg',))




    for model in MODELS:
        for noise, gnoise in (('0.4', '0.0'), ('0.8', '0.0'), ('0.4', '0.8')):
            if model == 'CNN' and noise == '0.4':
                condition_index = 1 if gnoise == '0.0' else 2
                add(f'sweep:{model}:{noise}:guide{gnoise}',
                    ('--stage', 'scenario', '--scenario', 'guide-quality',
                     '--scenario-index', str(condition_index)),
                    (f'main:CNN:0.4:rg', f'guide:ResNet20:{gnoise}'))
            else:
                add(f'sweep:{model}:{noise}:guide{gnoise}', ('--stage', 'all',
                    '--model', model, '--noise', noise, '--guide-noise', gnoise,
                    '--method', 'rg'), (f'architecture:{model}',
                                     f'guide:ResNet20:{gnoise}'))



    for index, (noise, fold) in enumerate(((n, f) for n in ('0.4', '0.8')
                                            for f in (0, 1)), start=1):
        add(f'crossfit-guide:{noise}:fold{fold}', ('--stage', 'scenario',
            '--scenario', 'crossfit-guides', '--scenario-index', str(index)),
            (f'guide:ResNet20:{noise}',))



    crossfit_conditions = (('Transformer', '0.4'), ('Transformer', '0.8'),
                           ('CNN', '0.4'), ('CNN', '0.8'),
                           ('MLP', '0.4'), ('MLP', '0.8'))
    for index, (model, noise) in enumerate(crossfit_conditions, start=1):
        deps = (f'main:{model}:{noise}:rg',
                f'crossfit-guide:{noise}:fold0', f'crossfit-guide:{noise}:fold1')
        add(f'crossfit:{model}:{noise}', ('--stage', 'scenario', '--scenario',
            'crossfit', '--scenario-index', str(index)), deps)



    for idx in range(1, 8):
        deps = ['main:CNN:0.4:rg']
        deps.append('guide:MLP-guide:0.0' if idx == 7 else 'guide:ResNet20:0.0')
        add(f'ablation:{idx}', ('--stage', 'scenario', '--scenario',
            'ablations', '--scenario-index', str(idx)), deps)

    guide_deps = tuple(f'guide:ResNet20:{n}' for n in NOISES)
    rg_deps = tuple(f'main:{m}:{n}:rg' for m in MODELS for n in NOISES)
    add('diagnostic:guide-memorization', ('--guide-memorization',), guide_deps,
        program='eval.py')
    add('diagnostic:rg-weight-allocation', ('--rg-weight-allocation',), rg_deps + guide_deps,
        program='eval.py')


    producers = tuple(step.name for step in steps
                      if step.name.startswith(('guide:', 'main:', 'control:',
                                               'sweep:', 'crossfit:', 'ablation:',
                                               'crossfit-guide:')))
    studies = tuple(step.name for step in steps
                    if step.name.startswith(('architecture:', 'guide:', 'main:',
                                             'sweep:', 'ablation:')))
    add('summary', ('--stats',), program='eval.py', refresh_after=producers)
    add('selected-parameters', ('--selected-hyperparameters',),
        program='eval.py', refresh_after=studies)
    for model in ('CNN', 'Transformer'):
        deps = tuple(f'main:{model}:0.4:{method}' for method in ('erm', 'rg', 'kd'))
        add(f'error-distributions:{model}', ('--detailed', '--architecture', model,
            '--noise-ratio', '0.4', '--guide-noise-ratio', '0.4'),
            deps + ('guide:ResNet20:0.4',), program='eval.py')
    assert len({s.name for s in steps}) == len(steps)
    names = set()
    for step in steps:
        if any(dep not in names for dep in (*step.depends_on, *step.refresh_after)):
            raise AssertionError(f'Unordered or missing dependency in {step.name}')
        names.add(step.name)
    return steps


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf8')
    os.replace(tmp, path)


def run_logged(step, log_file):
    """Stream output to both the console and a step-specific persistent log."""
    with log_file.open('w', encoding='utf8', errors='replace') as log:
        log.write(f'COMMAND: {subprocess.list2cmdline(step.command)}\n')
        log.flush()
        with subprocess.Popen(step.command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors='replace',
                              bufsize=1, env={**os.environ, 'PYTHONUNBUFFERED': '1'}) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            return process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--trials', type=int, default=20,
                        help='Stage-2 Optuna trials per condition (default: 20; maximum: 20)')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--search-epochs', type=int, default=300)
    parser.add_argument('--fit-epochs', type=int, default=300)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--plan', action='store_true', help='Show DAG and commands; do not execute')
    parser.add_argument('--resume', action='store_true',
                        help='Keep successful conditions and retry failed/blocked ones')
    args = parser.parse_args(argv)
    if args.trials > 20:
        parser.error('--trials must not exceed 20')
    if min(args.trials, args.runs, args.search_epochs, args.fit_epochs) < 1:
        parser.error('All budgets must be positive')
    if args.fit_epochs != 300:
        parser.error('Final fits must complete 300 epochs')
    if args.runs != 5:
        parser.error('This experiment uses exactly 5 final seeds (43..47)')
    steps = plan(args)
    print(f'Fashion-MNIST: {len(steps)} steps; Stage-1 LR 0.005 / 20 trials; '
          f'Stage-2 20 trials (default), LR [0.01,0.005,0.001] sampled by Optuna; '
          'RG scaling off; 300 final epochs.', flush=True)
    if args.plan:
        for i, step in enumerate(steps, 1):
            print(f'{i:03d} {step.name} | depends={list(step.depends_on)}\n'
                  f'    {subprocess.list2cmdline(step.command)}')
        return 0
    os.chdir(ROOT)
    configuration = {k: v for k, v in vars(args).items() if k not in ('resume', 'plan')}
    if args.resume:
        if MANIFEST.is_file():
            manifest = json.loads(MANIFEST.read_text(encoding='utf8'))
        else:
            manifest = {'status': {}, 'started_at': stamp()}
        manifest['protocol'] = PROTOCOL
        manifest['configuration'] = configuration
        manifest['steps'] = [s.name for s in steps]
        manifest['status'] = {name: manifest.get('status', {}).get(name, {})
                              for name in manifest['steps']
                              if name in manifest.get('status', {})}
        RESULTS.mkdir(parents=True, exist_ok=True)
        atomic_json(MANIFEST, manifest)
    else:
        if RESULTS.exists() and any(RESULTS.iterdir()):
            parser.error('results/ must be empty for a fresh experiment. No files deleted or backed up.')
        manifest = {'protocol': PROTOCOL, 'configuration': configuration,
                    'steps': [s.name for s in steps], 'status': {},
                    'started_at': stamp()}
        RESULTS.mkdir(parents=True, exist_ok=True)
        atomic_json(MANIFEST, manifest)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    status = manifest['status']
    steps_by_name = {step.name: step for step in steps}
    for i, step in enumerate(steps, start=1):
        upstream = {name: [status.get(name, {}).get('state'),
                           status.get(name, {}).get('at'),
                           artifact_signature(steps_by_name[name])]
                    for name in step.refresh_after}
        dependencies = {name: [status.get(name, {}).get('state'),
                               status.get(name, {}).get('at')]
                        for name in step.depends_on}
        old = status.get(step.name, {})
        if (old.get('state') == 'COMPLETED'
                and artifacts_valid(step)
                and (not step.refresh_after or old.get('upstream') == upstream)):
            print(f'[{i}/{len(steps)}] COMPLETED (resume): {step.name}', flush=True)
            continue
        blocked = [d for d in step.depends_on
                   if status.get(d, {}).get('state') != 'COMPLETED']
        if blocked:
            if (old.get('state') != 'BLOCKED'
                    or old.get('missing_dependencies') != blocked):
                status[step.name] = {'state': 'BLOCKED', 'at': stamp(),
                                     'missing_dependencies': blocked}
                atomic_json(MANIFEST, manifest)
            print(f'[{i}/{len(steps)}] BLOCKED {step.name}: {blocked}', flush=True)
            continue
        log_file = LOG_DIR / f'{i:03d}_{step.name.replace(":", "_")}.log'
        print(f'[{i}/{len(steps)}] RUN {step.name}', flush=True)
        cmd = step.command
        if args.resume and '--stage' in cmd:
            stage = cmd[cmd.index('--stage') + 1]
            if stage in ('all', 'fit', 'guide', 'scenario'):
                cmd = (*cmd, '--resume')
        try:
            exit_code = run_logged(Step(step.name, tuple(cmd), step.depends_on), log_file)
        except KeyboardInterrupt:
            status[step.name] = {'state': 'INTERRUPTED', 'at': stamp(),
                                 'log': str(log_file.relative_to(ROOT))}
            atomic_json(MANIFEST, manifest)
            raise
        except Exception as exc:
            exit_code = -1
            with log_file.open('a', encoding='utf8') as log:
                log.write(f'\nEXTERNAL ERROR: {exc!r}\n')
        if exit_code == 0 and not artifacts_valid(step):
            exit_code = -2
            with log_file.open('a', encoding='utf8') as log:
                log.write('Expected experiment outputs are missing or incomplete.\n')
        status[step.name] = {'state': 'COMPLETED' if exit_code == 0 else 'FAILED',
                             'at': stamp(), 'exit_code': exit_code,
                             'log': str(log_file.relative_to(ROOT))}
        if exit_code == 0:
            if step.refresh_after:
                status[step.name]['upstream'] = upstream
            if step.depends_on:
                status[step.name]['dependencies'] = dependencies
        atomic_json(MANIFEST, manifest)
        if exit_code != 0:
            print(f'[{i}/{len(steps)}] FAILED {step.name} (code {exit_code}); '
                  'continuing independent experiments', flush=True)
    counts = {state: sum(1 for entry in status.values()
                         if entry['state'] == state)
              for state in ('COMPLETED', 'FAILED', 'BLOCKED', 'INTERRUPTED')}
    manifest['finished_at'] = stamp()
    manifest['counts'] = counts
    atomic_json(MANIFEST, manifest)
    print('Final execution summary:', counts)
    print(f'Manifest: {MANIFEST}\nLogs: {LOG_DIR}')
    return 0 if counts['FAILED'] == counts['BLOCKED'] == counts['INTERRUPTED'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
