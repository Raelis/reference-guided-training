# Output directories for models, methods and experimental conditions.
from pathlib import Path

RESULTS = Path('results')


def architecture_dir(model, noise=0.0):
    return RESULTS / f'{model}_n{noise:.2f}_architecture'


def method_dir(model, noise, method, guide_model=None, guide_noise=None):
    name = f'{model}_n{noise:.2f}_{method}'
    if guide_model is not None:
        guide_noise = noise if guide_noise is None else guide_noise
        name += f'_guide-{guide_model}_n{guide_noise:.2f}'
    return RESULTS / name


def crossfit_guide_dir(noise, fold):
    return RESULTS / f'ResNet20_n{noise:.2f}_erm-crossfit_fold{fold}'


def crossfit_learner_dir(model, noise):
    return method_dir(model, noise, 'rg-crossfit', 'ResNet20', noise)


def ablation_dir(index, guide_model):
    return method_dir('CNN', .4, f'rg-ablation{index}', guide_model, 0.)


def erm_lr_sweep_dir(model, noise, lr):
    return RESULTS / f'{model}_n{noise:.2f}_erm-lr-{lr:g}'
