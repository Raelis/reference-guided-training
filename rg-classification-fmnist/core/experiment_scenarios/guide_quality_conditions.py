# CNN guide-quality ablation studies sharing the RG study at 40% guide noise.
from core.experiment import ScenarioCondition, method_spec


def conditions():
    return [ScenarioCondition(
        scenario='guide-quality', model='CNN', noise=.4, guide_noise=guide_noise,
        method='rg',
        directory=str(method_spec('rg').output_dir('CNN', .4, guide_noise)),
        parameter_source='rg-study', parameter_guide_noise=.4,
        guide_kind='resnet', metadata_type='guided', metadata_guide='ResNet20',
    ) for guide_noise in (0., .8)]
