import torch
import torch.nn as nn
import numpy as np

class MyModel(nn.Module):
    model_name = "MLP-guide"

    def __init__(
        self,
        dropout=0.2,
        num_features=784,
        output_dim=10
    ):
        super().__init__()

        if isinstance(num_features, tuple):
            num_features = int(np.prod(num_features))

        hidden_dims = (8,)

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
        return {}

    @staticmethod
    def prepare_model_params(model_params):
        other_params = {}

        for k in ["batch_size", "window_size", "lr", "weight_decay", "lamb", "delta", "margin", "alpha", "temperature"]:
            if k in model_params:
                other_params[k] = model_params[k]
                model_params.pop(k, None)

        return {}, other_params

    @staticmethod
    def model_postprocess(model):
        pass