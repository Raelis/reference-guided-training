import torch
import torch.nn as nn

class MyModel(nn.Module):
    model_name = "MLP"

    def __init__(
        self,
        hidden_dims=(128, 64),
        dropout=0.2,
        num_features=240,  # number of features (window_size)
        output_dim=48 
    ):
        super().__init__()

        layers = []
        prev_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))  # output 1 value (regression)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, window_size)
        if x.dim() == 3:  # case (batch, 1, window_size)
            x = x.squeeze(1)
        x = x.view(x.size(0), -1)  # flatten for (batch, window_size)
        out = self.mlp(x)
        return out.unsqueeze(-1)  # output (batch, output_dim, 1)

    @staticmethod
    def prepare_trial(trial):
        n_layers = trial.suggest_int("mlp_layers", 1, 3)
        hidden_dims = [
            trial.suggest_categorical(
                f"mlp_hidden_{i}", [256, 512, 1024, 2048, 4096]
            )
            for i in range(n_layers)
        ]
        dropout = trial.suggest_float("mlp_dropout", 0.05, 0.3)

        model_params = {
            "hidden_dims": tuple(hidden_dims),
            "dropout": dropout,
        }
        return model_params

    
    @staticmethod
    def prepare_model_params(model_params):
        model_params = dict(model_params)
        other_params = {}

        for k in ["batch_size", "window_size", "lr", "weight_decay", "lamb", "delta", "margin"]:
            if k in model_params:
                other_params[k] = model_params.pop(k)

        if "mlp_layers" in model_params:
            n_layers = int(model_params.pop("mlp_layers"))
            model_params["hidden_dims"] = tuple(
                model_params.pop(f"mlp_hidden_{i}")
                for i in range(n_layers)
            )

        if "mlp_dropout" in model_params:
            model_params["dropout"] = model_params.pop("mlp_dropout")

        VALID_MODEL_KEYS = {
            "hidden_dims",
            "dropout",
        }
        model_params = {k: v for k, v in model_params.items() if k in VALID_MODEL_KEYS}

        return model_params, other_params


    @staticmethod
    def model_postprocess(model):
        pass