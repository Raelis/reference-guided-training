# CIFAR-10N experiment protocol, training settings, and Optuna search spaces.

SCENARIOS = {
    "independent": ("random1", "random3"),
    "matched": ("worst", "worst"),
    "cleaner-guide": ("worst", "aggre"),
}

STUDY_NAME = "experiment_with_indicators"
PROTOCOL_ID = "cifar10n_erm_rg_categorical_lr_no_scaling"
LR_CHOICES = (0.001, 0.005, 0.01)
STAGE1_EPOCHS = 30
STAGE1_PATIENCE = 15
STAGE1_LR = 0.01
FIXED_WEIGHT_DECAY = 5e-4
FIXED_BATCH_SIZE = 256

ERM_SEARCH_SPACE = {"lr": ("categorical", LR_CHOICES)}
RG_SEARCH_SPACE = {
    "lr": ("categorical", LR_CHOICES),
    "lamb": ("log-uniform", 0.25, 20.0),
    "delta": ("log-uniform", 1e-4, 1.0),
    "margin": ("uniform", 0.9, 1.1),
}
SEARCH_SPACES = {"erm": ERM_SEARCH_SPACE, "rg": RG_SEARCH_SPACE}
MODE_TO_METHOD = {"baseline": "erm", "guided": "rg"}
GUIDED_METHODS = ("rg",)
EDGE_WARNING_FRACTION = 0.05
