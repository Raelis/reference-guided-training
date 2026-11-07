import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel

class MyModel(nn.Module):
    model_name = "Transformer"

    def __init__(
        self,
        hidden_size=192,
        num_attention_heads=3,
        num_hidden_layers=4,
        mlp_ratio=4,
        num_features=784,
        dropout=0.1,
        output_dim=10,
    ):
        super().__init__()
        self.num_labels = output_dim




        self.config = ViTConfig(
            image_size=28,
            patch_size=4,
            num_channels=1,
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
    def prepare_trial(trial):

        hidden_size = trial.suggest_categorical("hidden_size", [64, 96, 128])
        num_heads = trial.suggest_categorical("num_attention_heads", [2, 4])
        num_layers = trial.suggest_int("num_hidden_layers", 2, 6)
        mlp_ratio = trial.suggest_categorical("mlp_ratio", [2, 4])
        dropout = trial.suggest_float("dropout", 0.05, 0.3)

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
            "batch_size", "lr", "weight_decay",
            "lamb", "delta", "margin", "alpha", "temperature"
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