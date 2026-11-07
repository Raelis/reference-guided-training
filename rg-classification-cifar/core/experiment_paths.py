# Result paths for the CIFAR-10N experiment.
from pathlib import Path

RESULTS = Path("results")


def architecture_dir(model, noise):
    return RESULTS / f"{model}_{noise}_architecture"


def method_dir(model, noise, method, guide_model=None, guide_noise=None, tag=None):
    name = f"{model}_{noise}_{method}"
    if guide_model is not None:
        name += f"_guide-{guide_model}_{guide_noise}"
    if tag:
        name += f"_{tag}"
    return RESULTS / name


def guide_dir(noise):
    return method_dir("ResNet56", noise, "erm")
