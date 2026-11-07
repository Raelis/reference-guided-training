from core.experiment import Experiment

SEED = 42

experiments_stack = [   
    ("MLP", "no", "Hourly"),
    ("GRU", "no", "Hourly"),
    ("TCN", "no", "Hourly"),

    ("MLP", "no", "Weekly"),
    ("GRU", "no", "Weekly"),
    ("TCN", "no", "Weekly"),

    ("MLP", "no", "Daily"),
    ("GRU", "no", "Daily"),
    ("TCN", "no", "Daily"),

    ("MLP", "no", "Monthly"),
    ("GRU", "no", "Monthly"),
    ("TCN", "no", "Monthly"),

    ("MLP", "no", "Quarterly"),
    ("GRU", "no", "Quarterly"),
    ("TCN", "no", "Quarterly"),
    
    ("MLP", "no", "Yearly"),
    ("GRU", "no", "Yearly"),
    ("TCN", "no", "Yearly"),

    ("MLP", "NARMAX", "Hourly"),
    ("GRU", "NARMAX", "Hourly"),
    ("TCN", "NARMAX", "Hourly"),

    ("MLP", "NARMAX", "Weekly"),
    ("GRU", "NARMAX", "Weekly"),
    ("TCN", "NARMAX", "Weekly"),

    ("MLP", "NARMAX", "Daily"),
    ("GRU", "NARMAX", "Daily"),
    ("TCN", "NARMAX", "Daily"),

    ("MLP", "NARMAX", "Monthly"),
    ("GRU", "NARMAX", "Monthly"),
    ("TCN", "NARMAX", "Monthly"),

    ("MLP", "NARMAX", "Quarterly"),
    ("GRU", "NARMAX", "Quarterly"),
    ("TCN", "NARMAX", "Quarterly"),
    
    ("MLP", "NARMAX", "Yearly"),
    ("GRU", "NARMAX", "Yearly"),
    ("TCN", "NARMAX", "Yearly"),

    ("MLP", "seasonal-naive", "Hourly"),
    ("GRU", "seasonal-naive", "Hourly"),
    ("TCN", "seasonal-naive", "Hourly"),

    ("MLP", "seasonal-naive", "Weekly"),
    ("GRU", "seasonal-naive", "Weekly"),
    ("TCN", "seasonal-naive", "Weekly"),

    ("MLP", "seasonal-naive", "Daily"),
    ("GRU", "seasonal-naive", "Daily"),
    ("TCN", "seasonal-naive", "Daily"),

    ("MLP", "seasonal-naive", "Monthly"),
    ("GRU", "seasonal-naive", "Monthly"),
    ("TCN", "seasonal-naive", "Monthly"),

    ("MLP", "seasonal-naive", "Quarterly"),
    ("GRU", "seasonal-naive", "Quarterly"),
    ("TCN", "seasonal-naive", "Quarterly"),
    
    ("MLP", "seasonal-naive", "Yearly"),
    ("GRU", "seasonal-naive", "Yearly"),
    ("TCN", "seasonal-naive", "Yearly"),
]

for exp_model_name, loss_baseline, subset in experiments_stack:
    experiment = Experiment(exp_model_name, "mse", subset, SEED)

    if loss_baseline == "no":
        experiment.run_pre_train()
        experiment.run_baseline_train()
    else:
        experiment.run_guided_train("baseline-mse", baseline_method=loss_baseline)