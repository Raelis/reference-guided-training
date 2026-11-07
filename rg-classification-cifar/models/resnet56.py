import torch.nn as nn

from models.resnet20 import ResNet, BasicBlock


class MyModel(nn.Module):
    model_name = "ResNet56"

    def __init__(self, num_features=1024, output_dim=10):
        super().__init__()
        self.model = ResNet(BasicBlock, [9, 9, 9], num_classes=output_dim)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def prepare_trial(trial):
        return {}

    @staticmethod
    def prepare_model_params(model_params):
        other_params = {}

        for k in ["batch_size", "window_size", "lr", "weight_decay",
                  "lamb", "delta", "margin"]:
            if k in model_params:
                other_params[k] = model_params[k]
                model_params.pop(k, None)

        return {}, other_params

    @staticmethod
    def model_postprocess(model):
        pass
