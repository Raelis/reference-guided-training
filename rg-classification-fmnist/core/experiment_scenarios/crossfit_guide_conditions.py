#Two-fold ResNet20 guide conditions.
from core.experiment import ScenarioCondition
from core.experiment_paths import crossfit_guide_dir


def conditions():
    return [ScenarioCondition(
        scenario='crossfit-guides', model='ResNet20', noise=noise,
        guide_noise=noise, method='erm',
        directory=str(crossfit_guide_dir(noise, fold)),
        parameter_source='fixed', fixed_parameters=(('lr', 0.005),),
        guide_kind='fold', fold=fold, runs=1,
        metadata_type='baseline',
    ) for noise in (.4, .8) for fold in (0, 1)]
