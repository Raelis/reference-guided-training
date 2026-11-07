#Explicit LR control study
from core.experiment import ScenarioCondition, LR_GRID
from core.experiment_paths import erm_lr_sweep_dir


def conditions():
    return [ScenarioCondition(
        scenario='erm-lr-sweep', model=model, noise=noise, guide_noise=noise,
        method='erm', directory=str(erm_lr_sweep_dir(model, noise, lr)),
        parameter_source='fixed', fixed_parameters=(('lr', lr),),
        guide_kind='none', metadata_type='baseline_lr_sweep', runs=5,
    ) for model in ('CNN', 'MLP', 'Transformer')
      for noise in (0., .4, .8) for lr in LR_GRID]
