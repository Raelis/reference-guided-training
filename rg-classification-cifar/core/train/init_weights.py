import torch
import random
import numpy as np

def custom_init_weights(model, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for layer in model.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()


