from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR

class SchedulerFactory:
    def __init__(self, scheduler_name="plateau", **kwargs):
        self.scheduler_name = scheduler_name.lower()
        self.kwargs = kwargs.copy()

    def create(self, optimizer):
        if self.scheduler_name == "plateau":
            params = {
                "mode": "min",
                "factor": 0.95,
                "patience": 20,
                "min_lr": 1e-6
            }
            params.update(self.kwargs)
            return ReduceLROnPlateau(optimizer, **params)

        elif self.scheduler_name == "step":
            params = {"step_size": 10, "gamma": 0.1}
            params.update(self.kwargs)
            return StepLR(optimizer, **params)

        elif self.scheduler_name == "exp":
            params = {"gamma": 0.95}
            params.update(self.kwargs)
            return ExponentialLR(optimizer, **params)

        else:
            raise ValueError(f"Scheduler '{self.scheduler_name}' desconhecido")
