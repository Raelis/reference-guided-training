# Select and construct the CIFAR learner/reference architecture.
import copy
from importlib import import_module

MODEL_MODULES = {
    "CNN": "models.cnn2d",
    "MLP": "models.mlp_model",
    "Transformer": "models.transformer",
    "ResNet56": "models.resnet56",
}

def import_model_class(model_name):
    try:
        module = MODEL_MODULES[model_name]
    except KeyError:
        raise ValueError(f"Unsupported CIFAR model: {model_name}") from None
    return import_module(module).MyModel

def model_parameters(model_class, architecture_params):
    params = copy.deepcopy(architecture_params)
    params, _ = model_class.prepare_model_params(params)
    params["num_features"] = (3, 32, 32)
    params["output_dim"] = 10
    return params

def instantiate(model_class, model_params, device):
    """Construct an already prepared model and run its post-processing hook."""
    model = model_class(**model_params).to(device)
    model_class.model_postprocess(model)
    return model
