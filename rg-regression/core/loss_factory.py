import torch
import torch.nn.functional as F
from math import exp

class LossFactory:
    def __init__(self, loss_name="mse"):
        self.loss_name = loss_name.lower()

    def create(self, **create_kwargs):

        if self.loss_name == "mse":
            def mse_loss(y_pred, y_true, *args, **kwargs):
                return torch.mean((y_pred - y_true) ** 2)
            return mse_loss

        if self.loss_name == "mae":
            def mae_loss(y_pred, y_true, *args, **kwargs):
                return torch.mean(torch.abs(y_pred - y_true))
            return mae_loss

        if self.loss_name == "rmse":
            def rmse_loss(y_pred, y_true, *args, **kwargs):
                return torch.sqrt(torch.mean((y_pred - y_true) ** 2))
            return rmse_loss

        if self.loss_name == "huber":
            def huber_loss(y_pred, y_true, *args, **kwargs):
                error = y_pred - y_true
                delta = kwargs.get("delta", 0.1)
                cond = torch.abs(error) < delta
                loss = torch.where(cond, 0.5 * error**2, delta * (torch.abs(error) - 0.5 * delta))
                return torch.mean(loss)
            return huber_loss

        if self.loss_name == "pinball":
            def pinball_loss(y_pred, y_true, *args, **kwargs):
                quantile = kwargs.get("quantile", 0.5)
                error = y_true - y_pred
                loss = torch.max((quantile - 1) * error, quantile * error)
                return torch.mean(loss)
            return pinball_loss

        if self.loss_name == "smape":
            def smape_loss(y_pred, y_true, *args, **kwargs):
                numerator = torch.abs(y_true - y_pred)
                denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2
                return torch.mean(numerator / (denominator + 1e-8))
            return smape_loss

        if self.loss_name == "msle":
            def msle_loss(y_pred, y_true, *args, **kwargs):
                return torch.mean((torch.log1p(y_pred) - torch.log1p(y_true)) ** 2)
            return msle_loss

        # ----------------------
        # RG MSE LOSS
        # ----------------------
        if self.loss_name == "baseline-mse":
            delta = create_kwargs.get("delta", 0.1)
            lamb = create_kwargs.get("lamb", 1.0)
            margin = create_kwargs.get("margin", 1.0)

            def linear_quadratic_smooth_penalty(x, delta=delta):
                return torch.where(
                    x <= 0,
                    torch.zeros_like(x),
                    torch.where(
                        x < delta,
                        0.5 * x**2 / delta,
                        x - 0.5 * delta
                    )
                )

            def baseline_mse_loss(y_pred, y_true, y_baseline, delta=delta, lamb=lamb, margin=margin, *args, **kwargs):
                e_model = (y_pred - y_true) ** 2
                e_base  = (y_baseline - y_true) ** 2
                pen = linear_quadratic_smooth_penalty(e_model - margin*e_base, delta=delta)
                # pen = 0.5 * (e_model - margin*e_base / (e_model + margin*e_base + 1e-9)) + 0.5
                return torch.mean(e_model + lamb * pen)

            return baseline_mse_loss

        raise ValueError(f"Not supported loss -> '{self.loss_name}'.")
