import torch
import torch.nn as nn


class MyModel(nn.Module):
    model_name = "GRU"

    def __init__(
        self,
        num_features=240,
        gru_hidden=128,
        gru_layers=1,
        gru_dropout=0.2,
        dense_dim=64,
        window_size=1,
        output_dim=48 
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_features = num_features

        # GRU
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            batch_first=True
        )

        # Final dense layer
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden, dense_dim),
            nn.ReLU(),
            nn.Dropout(gru_dropout),
            nn.Linear(dense_dim, self.output_dim)
        )

    def forward(self, x):

        # x arrives as (batch, num_features)
        if x.dim() == 3:
            x = x.squeeze(1)

        # reorganize for GRU: (batch, seq_len, input_size)
        x = x.view(x.size(0), self.num_features, 1)

        gru_out, h_n = self.gru(x)

        last_hidden = h_n[-1]

        out = self.fc(last_hidden)

        return out.unsqueeze(-1)


    @staticmethod
    def prepare_trial(trial):

        gru_hidden = trial.suggest_categorical(
            "gru_hidden", [64, 128, 256, 384, 512]
        )

        gru_layers = trial.suggest_int("gru_layers", 1, 3)

        gru_dropout = trial.suggest_float("gru_dropout", 0.05, 0.3)

        dense_dim = trial.suggest_categorical(
            "dense_dim", [64, 128, 256]
        )

        model_params = {
            "gru_hidden": gru_hidden,
            "gru_layers": gru_layers,
            "gru_dropout": gru_dropout,
            "dense_dim": dense_dim
        }

        return model_params


    @staticmethod
    def prepare_model_params(model_params):

        other_params = {}

        for k in ["batch_size", "window_size", "lr", "weight_decay", "lamb", "delta", "margin"]:
            if k in model_params:
                other_params[k] = model_params[k]
                model_params.pop(k, None)

        VALID_MODEL_KEYS = {
            "gru_hidden",
            "gru_layers",
            "gru_dropout",
            "dense_dim"
        }

        model_params = {k: v for k, v in model_params.items() if k in VALID_MODEL_KEYS}

        return model_params, other_params


    @staticmethod
    def model_postprocess(model):
        pass
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)