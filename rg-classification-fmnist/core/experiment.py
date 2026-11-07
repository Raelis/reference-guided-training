from dataclasses import dataclass
from importlib import import_module
from typing import Optional

from .experiment_paths import method_dir


# Stage-1 architecture studies and guide fits used LR 0.005.
# Stage-2 trials select from LR_GRID; fixed-LR paths use the same default.
BASE_SEED = 42
BASE_LR = 0.005
FIXED_WEIGHT_DECAY = 5e-4
FIXED_BATCH_SIZE = 256
FIXED_MOMENTUM = 0.9
LR_GRID = (0.01, 0.005, 0.001)


@dataclass(frozen=True)
class SearchDimension:

    name: str
    low: float
    high: float
    integer: bool = False
    log: bool = False

    def suggest(self, trial):
        if self.integer:
            return trial.suggest_int(self.name, int(self.low), int(self.high))
        if self.log:
            return trial.suggest_float(self.name, self.low, self.high, log=True)
        return trial.suggest_float(self.name, self.low, self.high)


_RG_SEARCH = (
    SearchDimension('lamb', 0.25, 16.0, log=True),
    SearchDimension('delta', 1e-4, 1e-1, log=True),
    SearchDimension('margin', 0.8, 1.2),
)
_SPL_SEARCH = (
    SearchDimension('spl_lambda0', 0.01, 2.0, log=True),
    SearchDimension('spl_k', 1.05, 2.0),
)
_MENTORNET_PD_SEARCH = (
    SearchDimension('mnpd_loss_p_percentile', 0.50, 0.95),
    SearchDimension('mnpd_burn_in_pct', 0, 40, integer=True),
    SearchDimension('mnpd_moving_average_decay', 0.10, 0.90),
)
_KD_SEARCH = (
    SearchDimension('alpha', 0.1, 0.9),
    SearchDimension('temperature', 1.0, 8.0, log=True),
)


@dataclass(frozen=True)
class MethodSpec:

    name: str
    objective: str
    guide: bool = False
    study_from_rg: bool = False
    tune: bool = True
    trainer: str = 'standard'
    search_space: tuple = ()

    def fixed_parameters(self):
        return {
            'lr': BASE_LR,
            'weight_decay': FIXED_WEIGHT_DECAY,
            'batch_size': FIXED_BATCH_SIZE,
            'momentum': FIXED_MOMENTUM,
        }

    def sample_parameters(self, trial, *, lr_grid=False):
        """Return fixed settings plus Optuna suggestions in their original order."""
        params = self.fixed_parameters()
        if lr_grid:
            params['lr'] = trial.suggest_categorical('lr', list(LR_GRID))
        for dimension in self.search_space:
            params[dimension.name] = dimension.suggest(trial)
        return params

    def selected_parameters(self, trial):
        return {**self.fixed_parameters(), **trial.params}

    def make_loss(self, parameters):
        from .loss_factory import LossFactory

        if self.objective == 'rg':
            return LossFactory('rg').create(
                lamb=parameters['lamb'], delta=parameters['delta'],
                margin=parameters['margin'],
            )
        if self.objective == 'rg-mean-one':
            return LossFactory('rg-mean-one').create(
                lamb=parameters['lamb'], delta=parameters['delta'],
                margin=parameters['margin'],
            )
        if self.objective == 'kd':
            return LossFactory('kd').create(
                alpha=parameters['alpha'], temperature=parameters['temperature'],
            )
        return LossFactory('ce').create()

    def output_dir(self, model, noise, guide_noise):
        return method_dir(
            model, noise, self.name,
            'ResNet20' if self.guide else None, guide_noise,
        )

    def study_path(self, model, noise, guide_noise):
        # Optimization controls reuse the same selected RG study, not a new
        # study or another hyperparameter search.
        source = method_spec('rg') if self.study_from_rg else self
        return source.output_dir(model, noise, guide_noise) / 'optuna_study.db'

    def result_type(self):
        if self.name == 'rg-mean-one':
            return 'guided_normw'
        if self.guide:
            return 'guided'
        if self.name in ('erm', 'erm-lr-matched'):
            return 'baseline'
        return self.name


# This is the single registry used by the CLI and the runner.
METHOD_SPECS = {
    'erm': MethodSpec('erm', 'ce', tune=False),
    'erm-lr-matched': MethodSpec(
        'erm-lr-matched', 'ce', study_from_rg=True, tune=False,
    ),
    'rg': MethodSpec('rg', 'rg', guide=True,
                     search_space=_RG_SEARCH),
    'rg-mean-one': MethodSpec(
        'rg-mean-one', 'rg-mean-one', guide=True, study_from_rg=True, tune=False,
        search_space=_RG_SEARCH,
    ),
    'kd': MethodSpec('kd', 'kd', guide=True, search_space=_KD_SEARCH),
    'spl': MethodSpec('spl', 'ce', trainer='spl', search_space=_SPL_SEARCH),
    'mentornet-pd': MethodSpec(
        'mentornet-pd', 'ce', trainer='mentornet-pd',
        search_space=_MENTORNET_PD_SEARCH,
    ),
}


def method_spec(method):
    try:
        return METHOD_SPECS[method]
    except KeyError:
        raise ValueError(f'Unknown training method: {method}') from None


@dataclass(frozen=True)
class ScenarioCondition:
    # Ablations studies experiment condition.

    scenario: str
    model: str
    noise: float
    guide_noise: float
    method: str
    directory: str
    parameter_source: str = 'rg-study'
    parameter_guide_noise: Optional[float] = None
    fixed_parameters: tuple = ()
    expected_parameters: tuple = ()
    guide_kind: str = 'none'
    fold: int = -1
    runs: int = 5
    seed: int = BASE_SEED
    metadata_type: str = ''
    metadata_guide: str = ''
    lr_grid: bool = False


# Load scenario modules only when selected: they import ScenarioCondition here.
SCENARIOS = {
    'crossfit': 'crossfit_conditions',
    'crossfit-guides': 'crossfit_guide_conditions',
    'normweights': 'normweights',
    'ablations': 'ablations',
    'guide-quality': 'guide_quality_conditions',
    'erm-lr-sweep': 'erm_lr_sweep',
}


class Experiment:

    def __init__(self, exp_model_name=None, exp_loss_name='ce', exp_guide_model='no',
                 noise_ratio=0., guide_noise_ratio=0., SEED=BASE_SEED, *, cli_args=None):
        self.exp_model_name = exp_model_name
        self.exp_loss_name = exp_loss_name
        self.exp_guide_model = exp_guide_model
        self.noise_ratio = noise_ratio
        self.guide_noise_ratio = guide_noise_ratio
        self.SEED = SEED
        self.cli_args = cli_args

    @classmethod
    def from_cli(cls, args):
        return cls(
            exp_model_name=args.model, exp_loss_name='ce',
            exp_guide_model='ResNet20', noise_ratio=args.noise,
            guide_noise_ratio=args.guide_noise, SEED=BASE_SEED,
            cli_args=args,
        )

    def run(self):
        if self.cli_args is None:
            raise ValueError('A CLI configuration is required for search/fit.')
        from .experiment_runner import run
        return run(self.cli_args, method_spec(self.cli_args.method))

    def run_arch_hp_tuning(self):
        # This method delegates Stage-1 setup, Optuna, and validation to the runner.
        from .experiment_runner import run_architecture_stage
        return run_architecture_stage(
            self.exp_model_name, loss_name=self.exp_loss_name,
            noise_ratio=self.noise_ratio, seed=self.SEED,
            device=getattr(self.cli_args, 'device', None),
        )

    def pretrain_mentornet_pd(self, *, device='cpu', steps=30000, sample_size=100000):
        from models.mentornet_pd import pretrain_mentornet_pd
        return pretrain_mentornet_pd(
            device=device, steps=steps, sample_size=sample_size,
            seed=BASE_SEED,
        )

    def run_scenario(self, scenario, *, condition_index=None):
        # Execute conditions from one scenario (ablation studies) through the shared runner.
        if scenario not in SCENARIOS:
            raise ValueError(
                f'Unknown scenario: {scenario!r}; choose from {sorted(SCENARIOS)}'
            )

        module = import_module('core.experiment_scenarios.' + SCENARIOS[scenario])
        conditions = module.conditions()
        if condition_index is not None:
            if not 1 <= condition_index <= len(conditions):
                raise ValueError(
                    f'--scenario-index must be 1..{len(conditions)} for {scenario}'
                )
            conditions = [conditions[condition_index - 1]]

        from .experiment_runner import run_scenario
        args = self.cli_args
        return run_scenario(
            conditions,
            device=getattr(args, 'device', None),
            quiet=getattr(args, 'quiet', False),
            lr_grid=getattr(args, 'lr_grid', False),
            trials=getattr(args, 'trials', 30),
            search_epochs=getattr(args, 'search_epochs', 300),
            resume=getattr(args, 'resume', False),
        )
