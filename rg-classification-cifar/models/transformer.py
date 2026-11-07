import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel


IMAGE_SIZE = 32
PATCH_SIZE = 4
NUM_CHANNELS = 3

MAX_PARAMS = 1_000_000


class MyModel(nn.Module):
    model_name = "Transformer"

    def __init__(
        self,
        hidden_size=96,
        num_attention_heads=4,
        num_hidden_layers=4,
        mlp_ratio=4,
        num_features=(3, 32, 32),
        dropout=0.1,
        output_dim=10,
    ):
        super().__init__()
        self.num_labels = output_dim

        self.config = ViTConfig(
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            num_channels=NUM_CHANNELS,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * mlp_ratio,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )

        self.vit = ViTModel(self.config)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.num_labels)
        )

    def forward(self, x):

        outputs = self.vit(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)


    @staticmethod
    def count_params(hidden_size, num_attention_heads, num_hidden_layers, mlp_ratio):
        # Parameter count of a candidate configuration.
        cfg = ViTConfig(
            image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, num_channels=NUM_CHANNELS,
            hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * mlp_ratio,
        )
        backbone = ViTModel(cfg)
        head = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(0.0), nn.Linear(128, 10)
        )
        return (sum(p.numel() for p in backbone.parameters())
                + sum(p.numel() for p in head.parameters()))

    @staticmethod
    def prepare_trial(trial):


        hidden_size = trial.suggest_categorical("hidden_size", [64, 96, 128])
        num_heads = trial.suggest_categorical("num_attention_heads", [2, 4])
        num_layers = trial.suggest_int("num_hidden_layers", 2, 6)
        mlp_ratio = trial.suggest_categorical("mlp_ratio", [2, 4])
        dropout = trial.suggest_float("dropout", 0.05, 0.3)


        import optuna
        n = MyModel.count_params(hidden_size, num_heads, num_layers, mlp_ratio)
        if n > MAX_PARAMS:
            raise optuna.TrialPruned(
                f"ViT hidden={hidden_size} layers={num_layers} mlp_ratio={mlp_ratio} "
                f"has {n/1e6:.2f}M parameters, over the {MAX_PARAMS/1e6:.0f}M cap"
            )

        return {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_hidden_layers": num_layers,
            "mlp_ratio": mlp_ratio,
            "dropout": dropout
        }


    @staticmethod
    def prepare_model_params(model_params):
        other_params = {}

        for k in [
            "batch_size", "window_size", "lr", "weight_decay",
            "lamb", "delta", "margin"
        ]:
            if k in model_params:
                other_params[k] = model_params[k]
                model_params.pop(k, None)

        VALID_MODEL_KEYS = {
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "mlp_ratio",
            "dropout"
        }

        model_params = {
            k: v for k, v in model_params.items()
            if k in VALID_MODEL_KEYS
        }

        return model_params, other_params

    @staticmethod
    def model_postprocess(model):
        pass
