import torch
import torch.nn as nn




class MyModel(nn.Module):
    model_name = "CNN"

    def __init__(
        self,
        input_shape=(28,28),
        cnn_channels=[32, 64, 128],
        cnn_kernel=3,
        cnn_dropout=0.2,
        dense_dim=64,
        output_dim=10,
        num_features=784
    ):
        super().__init__()

        in_ch = 1
        layers = []
        for out_ch in cnn_channels:
            pad_h = (cnn_kernel - 1) // 2
            layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=(cnn_kernel, cnn_kernel), padding=pad_h)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cnn_dropout))
            layers.append(nn.MaxPool2d(kernel_size=2))
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)


        dummy_input = torch.zeros(1, 1, *input_shape)
        cnn_out_dim = self.cnn(dummy_input).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim, dense_dim),
            nn.ReLU(),
            nn.Dropout(cnn_dropout),
            nn.Linear(dense_dim, output_dim)
        )

    def forward(self, x):

        y = self.cnn(x)
        y = y.view(y.size(0), -1)
        out = self.fc(y)
        return out




    @staticmethod
    def prepare_trial(trial):
        channels_options = [[16, 32, 64], [32, 64, 128], [64, 128, 256]]
        cnn_channels = [trial.suggest_categorical(f"cnn_channel_l{i}", options)
                        for i, options in enumerate(channels_options)]
        cnn_kernel = trial.suggest_categorical("cnn_kernel", [3, 5])
        cnn_dropout = trial.suggest_float("cnn_dropout", 0.0, 0.3)
        dense_dim = trial.suggest_categorical("dense_dim", [64, 128, 196])

        return {
            "cnn_channels": cnn_channels,
            "cnn_kernel": cnn_kernel,
            "cnn_dropout": cnn_dropout,
            "dense_dim": dense_dim
        }

    @staticmethod
    def prepare_model_params(model_params):
        other_params = {}
        for k in ["batch_size", "window_size", "lr", "weight_decay", "lamb", "delta", "margin", "alpha", "temperature"]:
            if k in model_params:
                other_params[k] = model_params[k]
                model_params.pop(k, None)

        VALID_MODEL_KEYS = {"cnn_channels", "cnn_kernel", "cnn_dropout", "dense_dim"}
        model_params = {k: v for k, v in model_params.items() if k in VALID_MODEL_KEYS}

        return model_params, other_params

    @staticmethod
    def model_postprocess(model):



        pass
