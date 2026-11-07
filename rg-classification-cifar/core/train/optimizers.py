import torch

class OptimizerFactory:
    def __init__(self, model, optimizer_name="sgd", lr=5e-3, weight_decay=5e-4, **kwargs):
        self.model = model
        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.kwargs = kwargs.copy()
        self.last_optimizer = None

    def create(self):
        # Construct an optimizer over the supplied model parameters
        if self.optimizer_name == "adam":
            self.last_optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.lr, weight_decay=self.weight_decay, **self.kwargs)
        elif self.optimizer_name == "adamw":
            self.last_optimizer = torch.optim.AdamW(list(self.model.parameters()), lr=self.lr, weight_decay=self.weight_decay, **self.kwargs)
        elif self.optimizer_name == "sgd":
            self.last_optimizer = torch.optim.SGD(list(self.model.parameters()), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9, **self.kwargs)
        elif self.optimizer_name == "rmsprop":
            self.last_optimizer = torch.optim.RMSprop(list(self.model.parameters()), lr=self.lr, weight_decay=self.weight_decay, **self.kwargs)
        else:
            raise ValueError(f"Optimizer '{self.optimizer_name}' not supported.")

        return self.last_optimizer

