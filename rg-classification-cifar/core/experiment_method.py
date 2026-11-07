# Optuna parameters and loss construction for ERM and RG.
from dataclasses import dataclass
from core.experiment_protocol import LR_CHOICES


@dataclass(frozen=True)
class MethodSpec:
    name: str
    guided: bool = False

    def sample_parameters(self, trial):
        params = {"lr": trial.suggest_categorical("lr", list(LR_CHOICES))}
        if self.name == "rg":
            params["lamb"] = trial.suggest_float("lamb", 0.25, 20.0, log=True)
            params["delta"] = trial.suggest_float("delta", 1e-4, 1.0, log=True)
            params["margin"] = trial.suggest_float("margin", 0.9, 1.1)
        return params

    def learning_rate(self, params):
        return params["lr"]

    def make_loss(self, loss_factory, params):
        if self.name == "rg":
            return loss_factory.create(lamb=params["lamb"], delta=params["delta"],
                                       margin=params["margin"])
        return loss_factory.create()

    @property
    def result_type(self):
        return {"erm": "baseline", "rg": "guided"}[self.name]


METHODS = {"erm": MethodSpec("erm"), "rg": MethodSpec("rg", guided=True)}


def method_spec(method):
    try:
        return METHODS[method]
    except KeyError:
        raise ValueError(f"Unknown CIFAR method: {method}") from None
