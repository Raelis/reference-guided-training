#Fixed crossfit conditions.
from core.experiment import ScenarioCondition
from core.experiment_paths import crossfit_learner_dir

def conditions():
    configurations = [("Transformer", .4), ("Transformer", .8),
                      ("CNN", .4), ("CNN", .8),
                      ("MLP", .4), ("MLP", .8)]
    return [ScenarioCondition(
        scenario='crossfit', model=model, noise=noise, guide_noise=noise,
        method='rg', directory=str(crossfit_learner_dir(model, noise)),
        parameter_source='rg-study',
        guide_kind='crossfit', metadata_type='guided_crossfit', metadata_guide='ResNet20-cf2',
    ) for model, noise in configurations]
