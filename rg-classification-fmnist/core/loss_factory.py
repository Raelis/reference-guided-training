import torch
import torch.nn.functional as F
from math import exp
import inspect

class LossFactory:
    def __init__(self, loss_name="mse"):
        self.loss_name = loss_name.lower()

    def create(self, **create_kwargs):

        if self.loss_name == "ce":

            _ce_signature = inspect.signature(F.cross_entropy)
            _ce_valid_params = set(_ce_signature.parameters.keys())

            def ce_loss(y_pred, y_true, *args, **kwargs):
                filtered_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k in _ce_valid_params
                }

                return F.cross_entropy(
                    y_pred,
                    y_true,
                    *args,
                    **filtered_kwargs
                )

            return ce_loss



        # Default implementation of RG training objective.
        if self.loss_name == "rg":
            delta = create_kwargs.get("delta", 0.1)
            lamb = create_kwargs.get("lamb", 1.0)
            margin = create_kwargs.get("margin", 1.0)

            print(f"Creating RG CE loss with delta={delta}, lamb={lamb}, margin={margin}")

            def linear_quadratic_smooth_penalty(x, delta=delta):


                if delta <= 0:
                    raise ValueError("RG delta must be positive")
                quadratic = torch.clamp(x, min=0.0, max=delta)
                linear = F.relu(x - delta)
                return 0.5 * quadratic.square() / delta + linear


            def rg_cross_entropy_loss(y_pred, y_true, y_baseline, delta=delta, lamb=lamb, margin=margin, *args, **kwargs):
                e_model = torch.nn.functional.cross_entropy(y_pred, y_true, reduction='none')
                e_base  = torch.nn.functional.cross_entropy(y_baseline, y_true, reduction='none')

                pen = linear_quadratic_smooth_penalty(e_model - margin * e_base, delta=delta)
                result = torch.mean(e_model + lamb*pen)
                return result

            return rg_cross_entropy_loss



        # RG mean-one, which normalizes the RG weights to have a batch mean of one. This was used in the ablation studies.
        if self.loss_name == "rg-mean-one":
            delta = create_kwargs.get("delta", 0.1)
            lamb = create_kwargs.get("lamb", 1.0)
            margin = create_kwargs.get("margin", 1.0)

            print(f"Creating RG CE loss (normw) with delta={delta}, lamb={lamb}, margin={margin}")

            def rg_mean_one_loss(y_pred, y_true, y_baseline, delta=delta, lamb=lamb, margin=margin, *args, **kwargs):
                e_model = torch.nn.functional.cross_entropy(y_pred, y_true, reduction='none')
                e_base  = torch.nn.functional.cross_entropy(y_baseline, y_true, reduction='none')

                # Normalize detached RG weights to unit batch mean.
                with torch.no_grad():
                    r = e_model.detach() - margin * e_base.detach()
                    w = 1.0 + lamb * torch.clamp(r / delta, min=0.0, max=1.0)
                    w = w / w.mean()

                return (w * e_model).mean()

            return rg_mean_one_loss



        # Created just to investigate a different formulation to RG, but not used in the paper.
        if self.loss_name == "rg-detached-weights":
            delta = create_kwargs.get("delta", 0.1)
            lamb = create_kwargs.get("lamb", 1.0)
            margin = create_kwargs.get("margin", 1.0)

            print(f"Creating RG CE loss (expw) with delta={delta}, lamb={lamb}, margin={margin}")

            def rg_detached_weights_loss(y_pred, y_true, y_baseline,
                                                  delta=delta, lamb=lamb, margin=margin,
                                                  *args, **kwargs):
                e_model = torch.nn.functional.cross_entropy(y_pred, y_true, reduction='none')
                e_base  = torch.nn.functional.cross_entropy(y_baseline, y_true, reduction='none')

                with torch.no_grad():
                    r = e_model.detach() - margin * e_base.detach()
                    w = 1.0 + lamb * torch.clamp(r / delta, min=0.0, max=1.0)

                return (w * e_model).mean()

            return rg_detached_weights_loss

        # Knowledge distillation loss, following Hinton et al. (2015).
        if self.loss_name == "kd":
            alpha = create_kwargs.get("alpha", 0.5)
            T = create_kwargs.get("temperature", 4.0)

            def kd_loss(
                y_pred,
                y_true,
                y_baseline,
                alpha=alpha,
                T=T,
                *args,
                **kwargs
            ):

                ce_loss = torch.nn.functional.cross_entropy(
                    y_pred,
                    y_true
                )

                teacher_logits = y_baseline.detach()


                log_p_student = torch.nn.functional.log_softmax(
                    y_pred / T,
                    dim=1
                )

                p_teacher = torch.nn.functional.softmax(
                    teacher_logits / T,
                    dim=1
                )


                kd_term = torch.nn.functional.kl_div(
                    log_p_student,
                    p_teacher,
                    reduction="batchmean"
                )


                loss = (
                    (1 - alpha) * ce_loss
                    + alpha * (T ** 2) * kd_term
                )
                return loss

            return kd_loss

        raise ValueError(f"Loss '{self.loss_name}' not supported.")
