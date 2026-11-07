import torch
import torch.nn as nn

class MyModel(nn.Module):
    model_name = "MLP"

    def __init__(
        self,
        hidden_dims=(128, 64),
        dropout=0.2,
        num_features=784,
        output_dim=10
    ):
        super().__init__()

        layers = []
        prev_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):

        x = x.view(x.size(0), -1)
        out = self.mlp(x)
        return out

    @staticmethod
    def prepare_trial(trial):
        n_layers = trial.suggest_int("mlp_layers", 1, 3)
        hidden_dims = [
            trial.suggest_categorical(
                f"mlp_hidden_{i}", [128, 256, 512, 1024]
            )
            for i in range(n_layers)
        ]
        dropout = trial.suggest_float("mlp_dropout", 0.0, 0.3)

        model_params = {
            "hidden_dims": tuple(hidden_dims),
            "dropout": dropout,
        }
        return model_params

    @staticmethod
    def prepare_model_params(model_params):
        other_params = {}
        for k in ["batch_size", "window_size", "lr", "weight_decay", "lamb", "delta", "margin", "alpha", "temperature"]:
            if k in model_params:
                other_params[k] = model_params[k]
                model_params.pop(k, None)

        VALID_MODEL_KEYS = {
            "hidden_dims",
            "dropout",
        }
        model_params = {k: v for k, v in model_params.items() if k in VALID_MODEL_KEYS}
        return model_params, other_params

    @staticmethod
    def model_postprocess(model):
        pass