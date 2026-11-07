# Fixed normweights conditions.
from core.experiment import ScenarioCondition, method_spec

def conditions():
    configurations = [("Transformer", .4), ("Transformer", .8),
                      ("CNN", .4), ("CNN", .8),
                      ("MLP", .4), ("MLP", .8),
                      ("Transformer", .0), ("CNN", .0), ("MLP", .0)]
    return [ScenarioCondition(
        scenario='normweights', model=model, noise=noise, guide_noise=noise,
        method='rg-mean-one', directory=str(method_spec('rg-mean-one').output_dir(model, noise, noise)),
        parameter_source='rg-study',
        guide_kind='resnet', metadata_type='guided_normw', metadata_guide='ResNet20',
    ) for model, noise in configurations]
