"""Fashion-MNIST result aggregation, checkpoint diagnostics and plots."""
import argparse
import os
import sys
from pathlib import Path

from core.evaluation_summary import (
    find_csv_files, load_and_concat, save_condition_stats,
    export_selected_hyperparameters,
)
from core.evaluation_diagnostics import (
    evaluate_guide_memorization, evaluate_rg_weight_allocation,
)

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', default='results')
    parser.add_argument('--out', default='results/processed_results/experiments_results_summary.csv')
    parser.add_argument('--detailed', action='store_true', help='Generate checkpoint-based error distributions')
    parser.add_argument('--paper-figures', action='store_true', help='Generate paper CNN noise/guide sweeps from final CSV results')
    parser.add_argument('--verify-figure-fonts', action='store_true', help='Audit existing figure PDFs without loading dataset, CSVs or checkpoints')
    parser.add_argument('--paper-r-distribution', action='store_true', help='Generate eight-panel guide_memorization_r_dist.pdf from saved checkpoints')
    parser.add_argument('--stats', action='store_true', help='Write per-condition CSV statistics')
    parser.add_argument('--stats-out', default='results/processed_results/experiments_detailed_summary.csv')
    parser.add_argument('--guide-memorization', action='store_true')
    parser.add_argument('--rg-weight-allocation', action='store_true')
    parser.add_argument('--selected-hyperparameters', action='store_true')
    parser.add_argument('--architecture', choices=('CNN', 'Transformer'), default='CNN')
    parser.add_argument('--methods', nargs='+',
                        type=lambda value: value.strip().lower(),
                        choices=('reference', 'erm', 'rg', 'kd', 'spl', 'mentornet-pd'),
                        default=['reference', 'erm', 'rg', 'kd', 'spl'],
                        metavar='METHOD',
                        help='Curves to plot: reference, erm, rg, kd, spl, mentornet-pd. '
                             'Use e.g. --methods erm rg kd to draw exactly three curves.')
    parser.add_argument('--noise-ratio', type=float, default=.4)
    parser.add_argument('--guide-noise-ratio', type=float, default=None)
    parser.add_argument('--results-dir', type=Path, default=None)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('figures'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--guide-dir', default=None)
    parser.add_argument('--guide-checkpoint', type=Path, default=None)
    for method in ('erm', 'rg', 'kd', 'spl', 'mentornet-pd'):
        parser.add_argument(f'--{method}-dir', default=None)
        parser.add_argument(f'--{method}-runs', nargs='+', type=int,
                            default=[1, 2, 3, 4, 5], metavar='RUN_ID')
    parser.add_argument('--grid-points', type=int, default=1200)
    parser.add_argument('--x-max', type=float, default=3.0,
                        help='Right limit of the displayed CE axis (default 3.0, '
                             'matching the paper figure axis). Set 0 to use '
                             '--grid-source/--grid-percentile instead.')
    parser.add_argument('--grid-percentile', type=float, default=99.5)
    parser.add_argument('--grid-source', choices=('reference', 'all'), default='reference')
    parser.add_argument('--global-bandwidth', type=float, default=.06)
    parser.add_argument('--subset-bandwidth', type=float, default=.10)
    parser.add_argument('--show-run-variation', action='store_true')
    parser.add_argument('--png', action='store_true')
    parser.add_argument('--kde-mode', choices=('auto','original','robust'), default='auto', help='Gaussian KDE evaluated with a portable NumPy kernel')
    parser.add_argument('--no-download', action='store_true')
    raw_args = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(raw_args)
    if args.verify_figure_fonts:
        from core.pdf_font_validation import verify_figure_directory
        verify_figure_directory(args.output_dir)
        return
    if len(set(args.methods)) != len(args.methods):
        parser.error('--methods contains duplicate choices')
    if args.detailed and not any(m != 'reference' for m in args.methods):
        parser.error('--methods must include at least one learner method')
    if not args.detailed:
        detailed_flags = {'--architecture', '--methods', '--x-max', '--noise-ratio', '--guide-noise-ratio',
            '--results-dir', '--data-dir', '--output-dir', '--device', '--batch-size',
            '--guide-dir', '--guide-checkpoint', '--erm-dir', '--rg-dir', '--kd-dir',
            '--erm-runs', '--rg-runs', '--kd-runs', '--spl-runs', '--spl-dir',
            '--mentornet-pd-dir', '--mentornet-pd-runs', '--grid-points', '--grid-percentile',
            '--grid-source', '--global-bandwidth', '--subset-bandwidth',
            '--show-run-variation', '--png', '--kde-mode', '--no-download'}
        if args.paper_figures or args.paper_r_distribution:
            detailed_flags -= {'--output-dir', '--results-dir', '--png'}
        if args.paper_r_distribution:
            detailed_flags -= {'--device', '--data-dir'}
        if any(flag.split('=')[0] in detailed_flags for flag in raw_args):
            parser.error('Detailed evaluation options require --detailed.')
    args.results_dir = args.results_dir or Path(args.results_root)

    diagnostics = (args.guide_memorization or args.rg_weight_allocation
                   or args.selected_hyperparameters)
    if not diagnostics or args.stats or args.detailed or args.paper_figures or args.paper_r_distribution:
        paths = find_csv_files(args.results_root)
        print(f'Found {len(paths)} CSV files')
        data = load_and_concat(paths)
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output, index=False)
        print(f'Saved in: {output}')
        print(f'Final shape: {data.shape}')
        if args.stats or args.detailed:
            save_condition_stats(args.results_root, args.stats_out)
    if args.detailed:
        import core.evaluation_distributions as kde_module
        import matplotlib
        print(f'[Figure source] {kde_module.__file__}', flush=True)
        print(f'[Figure font mode] pdf.fonttype={matplotlib.rcParams["pdf.fonttype"]}', flush=True)
        kde_module.run_error_distributions(args)
    if args.paper_figures:
        from core.evaluation_paper_figures import generate_sweeps
        generate_sweeps(args.results_dir, args.output_dir, png=args.png)
    if args.paper_r_distribution:
        from core.evaluation_paper_figures import generate_r_distribution
        generate_r_distribution(args.results_dir, args.output_dir, device=args.device, data_dir=args.data_dir)
    if args.guide_memorization:
        evaluate_guide_memorization()
    if args.rg_weight_allocation:
        evaluate_rg_weight_allocation()
    if args.selected_hyperparameters:
        export_selected_hyperparameters(args.results_root)


if __name__ == '__main__':
    os.chdir(ROOT)
    main()
