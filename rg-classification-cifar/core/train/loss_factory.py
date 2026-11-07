# Cross-entropy and numerically stable reference-guided loss.
import inspect
import torch
import torch.nn.functional as F


class LossFactory:
    def __init__(self, loss_name="ce"):
        self.loss_name = loss_name.lower()

    def create(self, **create_kwargs):
        if self.loss_name == "ce":
            valid_params = set(inspect.signature(F.cross_entropy).parameters)

            def ce_loss(y_pred, y_true, *args, **kwargs):
                filtered = {key: value for key, value in kwargs.items() if key in valid_params}
                return F.cross_entropy(y_pred, y_true, *args, **filtered)

            return ce_loss

        if self.loss_name == "rg":
            delta = create_kwargs.get("delta", 0.1)
            lamb = create_kwargs.get("lamb", 1.0)
            margin = create_kwargs.get("margin", 1.0)

            def linear_quadratic_smooth_penalty(x, delta=delta):
                if delta <= 0:
                    raise ValueError("RG delta must be positive")
                # Avoid squaring residuals beyond the quadratic region.
                quadratic = torch.clamp(x, min=0.0, max=delta)
                linear = F.relu(x - delta)
                return 0.5 * quadratic.square() / delta + linear

            def rg_loss(y_pred, y_true, y_baseline, delta=delta, lamb=lamb,
                        margin=margin, *args, **kwargs):
                e_model = F.cross_entropy(y_pred, y_true, reduction="none")
                e_base = F.cross_entropy(y_baseline, y_true, reduction="none")
                pen = linear_quadratic_smooth_penalty(e_model - margin * e_base, delta=delta)
                return torch.mean(e_model + lamb * pen)

            return rg_loss

        raise ValueError(f"Unsupported classification loss: {self.loss_name}")
