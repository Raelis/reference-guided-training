import torch
import torch.nn as nn
from pytorch_tcn import TCN

# ==============================================
# DEFINITION OF THE PURE TCN MODEL
# ==============================================
class MyModel(nn.Module):
    model_name = "TCN"

    def __init__(
        self,
        num_features,
        tcn_channels=[64, 128, 256],
        tcn_kernel=3,
        tcn_dropout=0.2,
        dense_dim=64,
        window_size=1,
        output_dim=48
    ):
        super().__init__()

        self.output_dim = output_dim
        self.num_features = num_features  # here = 240 (seq_len)

        # causal TCN 
        self.tcn = TCN(
            num_inputs=1,   # each timestep has 1 feature
            num_channels=tcn_channels,
            kernel_size=tcn_kernel,
            dropout=tcn_dropout,
            causal=True
        )

        # Final dense layer
        self.fc = nn.Sequential(
            nn.Linear(tcn_channels[-1], dense_dim),
            nn.ReLU(),
            nn.Dropout(tcn_dropout),
            nn.Linear(dense_dim, self.output_dim)
        )

    def forward(self, x):

        if x.dim() == 3:
            x = x.squeeze(1)

        # input: (batch, 240)
        x = x.view(x.size(0), self.num_features, 1)   # (batch, seq_len, features)

        # TCN > (batch, channels, seq_len)
        x = x.permute(0, 2, 1)                        # (batch, 1, 240)

        tcn_out = self.tcn(x)                         # (batch, channels, seq_len)

        # use the last timestep (causal)
        last_step = tcn_out[:, :, -1]                 # (batch, channels)

        out = self.fc(last_step)

        return out.unsqueeze(-1)

    # ==============================================
    # OPTUNA
    # ==============================================
    @staticmethod
    def prepare_trial(trial):
        num_levels = trial.suggest_int("num_levels", 2, 6)
        channels_options = [32, 64, 128, 256]

        tcn_channels = [
            trial.suggest_categorical(f"tcn_channel_l{i}", channels_options)
            for i in range(num_levels)
        ]

        tcn_kernel = trial.suggest_categorical("tcn_kernel", [2, 3, 5])
        tcn_dropout = trial.suggest_float("tcn_dropout", 0.05, 0.5)

        dense_dim = trial.suggest_int("dense_dim", 32, 256, step=32)

        model_params = {
            "tcn_channels": tcn_channels,
            "tcn_kernel": tcn_kernel,
            "tcn_dropout": tcn_dropout,
            "dense_dim": dense_dim,
        }
        return model_params

    @staticmethod
    def prepare_model_params(model_params):

        num_levels_key = "num_levels"
        channel_key_template = "tcn_channel_l{}"

        tcn_channels = []
        if num_levels_key in model_params:
            num_levels = int(model_params[num_levels_key])
            for i in range(num_levels):
                k = channel_key_template.format(i)
                if k in model_params:
                    tcn_channels.append(model_params[k])
                    del model_params[k]
            model_params.pop(num_levels_key, None)

        model_params["tcn_channels"] = tcn_channels

        other_params = {}
        for k in ["batch_size", "window_size", "lr", "weight_decay", "lamb", "delta", "margin"]:
            if k in model_params:
                other_params[k] = model_params[k]
                model_params.pop(k)

        VALID_MODEL_KEYS = {
            "tcn_channels",
            "tcn_kernel",
            "tcn_dropout",
            "dense_dim"
        }

        model_params = {k: v for k, v in model_params.items() if k in VALID_MODEL_KEYS}

        return model_params, other_params

    @staticmethod
    def model_postprocess(model):
        pass