# Fixed seven RG ablation configurations (definitions only).
from core.experiment import ScenarioCondition
from core.experiment_paths import ablation_dir

def conditions():
    configurations = [
        (.5, 1e-3, 1., 'ResNet20'),
        (3., 1e-3, 1., 'ResNet20'),
        (18., 1e-3, 1., 'ResNet20'),
        (3., 1e-1, 1., 'ResNet20'),
        (3., 1e-3, .5, 'ResNet20'),
        (3., 1e-3, 1.5, 'ResNet20'),
        (3., 1e-3, 1., 'MLP-guide'),
    ]
    return [ScenarioCondition(
        scenario='ablations', model='CNN', noise=.4, guide_noise=0.,
        method='rg',
        directory=str(ablation_dir(i, guide)),
        parameter_source='fixed',
        fixed_parameters=tuple({'lamb': lamb, 'delta': delta, 'margin': margin}.items()),
        guide_kind='resnet' if guide == 'ResNet20' else 'mlp',
        metadata_type='guided', metadata_guide=guide,
    ) for i, (lamb, delta, margin, guide) in enumerate(configurations, start=1)]
