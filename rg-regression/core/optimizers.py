import torch

class OptimizerFactory:
    def __init__(self, model, optimizer_name="adamw", lr=1e-3, weight_decay=0.0, **kwargs):
        self.model = model
        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.kwargs = kwargs.copy()
        self.last_optimizer = None

    def create(self):
        if self.optimizer_name == "adam":
            self.last_optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.lr, weight_decay=self.weight_decay, **self.kwargs)
        elif self.optimizer_name == "adamw":
            self.last_optimizer = torch.optim.AdamW(list(self.model.parameters()), lr=self.lr, weight_decay=self.weight_decay, **self.kwargs)
        elif self.optimizer_name == "sgd":
            self.last_optimizer = torch.optim.SGD(list(self.model.parameters()), lr=self.lr, weight_decay=self.weight_decay, **self.kwargs)
        elif self.optimizer_name == "rmsprop":
            self.last_optimizer = torch.optim.RMSprop(list(self.model.parameters()), lr=self.lr, weight_decay=self.weight_decay, **self.kwargs)
        else:
            raise ValueError(f"Optimizer '{self.optimizer_name}' não suportado.")
        
        return self.last_optimizer

