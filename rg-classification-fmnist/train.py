"""Fashion-MNIST experiment CLI: Experiment orchestration + shared trainers.

All commands can run from any working directory. Scenarios specify fixed conditions
processed by Experiment and the common runner; no parallel training scripts.
"""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


from core.experiment import SCENARIOS
SCENARIO_NAMES = tuple(SCENARIOS)


def parser():
    p = argparse.ArgumentParser(description='Reference-Guided Training experiments on Fashion-MNIST')
    p.add_argument('--stage', choices=['architecture', 'guide', 'search', 'fit', 'all',
                                      'scenario', 'mentornet-pretrain'], required=True)
    p.add_argument('--model', choices=['CNN', 'MLP', 'Transformer', 'ResNet20', 'MLP-guide'], default='CNN')
    p.add_argument('--noise', type=float, choices=[0., .4, .8], default=.4)
    p.add_argument('--guide-noise', type=float, choices=[0., .4, .8], default=None)
    p.add_argument('--method', choices=['erm', 'erm-lr-matched', 'rg', 'kd', 'rg-mean-one', 'spl', 'mentornet-pd'], default='rg')
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--mentor-pretrain-steps', type=int, default=30000)
    p.add_argument('--mentor-pretrain-samples', type=int, default=100000)
    p.add_argument('--search-epochs', type=int, default=300)
    p.add_argument('--fit-epochs', type=int, default=300)
    p.add_argument('--device', default=None, choices=['cpu', 'cuda'])
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--lr-grid', action='store_true', default=True,
                   help='Use the standard categorical LR search [0.01, 0.005, 0.001] (enabled by default)')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--start-run', type=int, default=None, metavar='N')
    p.add_argument('--run-ids', default=None, metavar='LIST')
    p.add_argument('--scenario', choices=SCENARIO_NAMES, default=None,
                   help='Fixed experimental scenario: predefined conditions and seeds')
    p.add_argument('--scenario-index', type=int, default=None, metavar='N',
                   help='Run exactly one 1-based condition within --scenario (for independent failure handling)')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.guide_noise = args.noise if args.guide_noise is None else args.guide_noise
    reference_stage = args.stage == 'guide'
    scenarios = args.stage == 'scenario'
    scenario = args.scenario
    if scenarios and scenario is None:
        p.error('Choose --scenario for --stage scenario')
    if scenario and not scenarios:
        p.error('--scenario requires --stage scenario')
    if sum((args.resume, args.start_run is not None, args.run_ids is not None)) > 1:
        p.error('--resume, --start-run and --run-ids are mutually exclusive')
    if (args.start_run is not None or args.run_ids is not None) and args.stage not in ('fit', 'all'):
        p.error('--start-run and --run-ids apply only to --stage fit or all')
    if args.scenario_index is not None and (not scenarios or args.scenario_index < 1):
        p.error('--scenario-index requires --stage scenario and a positive index')
    if args.trials > 20:
        p.error('--trials must not exceed 20')
    if args.trials < 1 or args.runs < 1 or args.search_epochs < 1 or args.fit_epochs < 1:
        p.error('positive trial/run/epoch counts required')
    if args.stage == 'guide':
        if args.model not in ('ResNet20', 'MLP-guide'):
            p.error('--stage guide requires --model ResNet20 or MLP-guide')
        args.method = 'erm'
        args.stage = 'fit'
        args.guide_fit = True
    if args.stage in ('search', 'fit', 'all') and args.model in ('ResNet20', 'MLP-guide') and not reference_stage:
        p.error('Reference models cannot be selected as learners by this CLI')
    import torch
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    from core.experiment import Experiment
    experiment = Experiment.from_cli(args)
    if args.stage == 'mentornet-pretrain':
        experiment.pretrain_mentornet_pd(device=args.device,
            steps=args.mentor_pretrain_steps,
            sample_size=args.mentor_pretrain_samples)
        return 0
    if scenarios:


        print('Running experimental scenario:', scenario)
        print('This scenario uses fixed model/noise configurations and seeds.')
        return experiment.run_scenario(scenario, condition_index=args.scenario_index)
    if args.stage == 'architecture':

        experiment.noise_ratio = 0.
        experiment.guide_noise_ratio = 0.
        experiment.exp_guide_model = 'no'
        return experiment.run_arch_hp_tuning()
    return experiment.run()


if __name__ == '__main__':
    raise SystemExit(main())
