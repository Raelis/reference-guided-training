import torch
import torch.nn as nn


MAX_PARAMS = 1_000_000


class MyModel(nn.Module):
    model_name = "MLP"

    def __init__(
        self,
        hidden_dims=(128, 64),
        dropout=0.2,
        num_features=(3, 32, 32),
        output_dim=10
    ):
        super().__init__()


        self.num_features = 3*32*32

        layers = []
        prev_dim = self.num_features

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        return self.mlp(x)

    @staticmethod
    def prepare_trial(trial):
        n_layers = trial.suggest_int("mlp_layers", 1, 3)


        hidden_dims = [
            trial.suggest_categorical(
                f"mlp_hidden_{i}",
                [128, 256, 384]
            )
            for i in range(n_layers)
        ]

        dropout = trial.suggest_float("mlp_dropout", 0.05, 0.3)


        import optuna
        n_params = MyModel.count_params(hidden_dims)
        if n_params > MAX_PARAMS:
            raise optuna.TrialPruned(
                f"MLP with hidden_dims={tuple(hidden_dims)} has "
                f"{n_params/1e6:.2f}M parameters, over the {MAX_PARAMS/1e6:.0f}M cap"
            )

        return {
            "hidden_dims": tuple(hidden_dims),
            "dropout": dropout,
        }

    @staticmethod
    def count_params(hidden_dims, num_features=3 * 32 * 32, output_dim=10):
        # Parameter count of a candidate.
        total = 0
        prev = num_features
        for h in hidden_dims:
            total += prev * h + h
            prev = h
        total += prev * output_dim + output_dim
        return total

    @staticmethod
    def prepare_model_params(model_params):
        other_params = {}

        for k in ["batch_size", "window_size", "lr", "weight_decay", "lamb", "delta", "margin"]:
            if k in model_params:
                other_params[k] = model_params[k]
                model_params.pop(k, None)


        hidden_keys = sorted(k for k in model_params if k.startswith("mlp_hidden_"))
        if hidden_keys and "hidden_dims" not in model_params:
            model_params["hidden_dims"] = tuple(model_params[k] for k in hidden_keys)
        if "mlp_dropout" in model_params and "dropout" not in model_params:
            model_params["dropout"] = model_params["mlp_dropout"]

        VALID_MODEL_KEYS = {"hidden_dims", "dropout"}

        model_params = {
            k: v for k, v in model_params.items()
            if k in VALID_MODEL_KEYS
        }

        return model_params, other_params

    @staticmethod
    def model_postprocess(model):
        pass
