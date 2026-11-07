from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR, CosineAnnealingLR

class SchedulerFactory:
    def __init__(self, scheduler_name="plateau", **kwargs):
        self.scheduler_name = scheduler_name.lower()
        self.kwargs = kwargs.copy()

    def create(self, optimizer):
        if self.scheduler_name == "plateau":
            params = {
                "mode": "min",
                "factor": 0.8,
                "patience": 10,
                "min_lr": 1e-5
            }
            params.update(self.kwargs)
            return ReduceLROnPlateau(optimizer, **params)

        elif self.scheduler_name == "step":
            params = {"step_size": 30, "gamma": 0.8}
            params.update(self.kwargs)
            return StepLR(optimizer, **params)

        elif self.scheduler_name == "exp":
            params = {"gamma": 0.95}
            params.update(self.kwargs)
            return ExponentialLR(optimizer, **params)
        
        elif self.scheduler_name == "cosine":
            params = {
                "T_max": 300,
                "eta_min": 1e-5
            }
            params.update(self.kwargs)
            return CosineAnnealingLR(optimizer, **params)

        else:
            raise ValueError(f"Scheduler '{self.scheduler_name}' desconhecido")
