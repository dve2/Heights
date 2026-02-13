from torchmetrics import MeanSquaredError, MeanAbsoluteError, Metric
from torch import Tensor
import torch
import numpy as np
from scipy import stats

def zero_aware_filter(preds: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    mask = target != 0
    if torch.any(mask):
        # We want to calculate metrics only on globules not surface
        return preds[mask], target[mask]

    # Keep metric update valid on all-zero targets while contributing zero error.
    preds_fallback = torch.zeros(1, dtype=preds.dtype, device=preds.device)
    target_fallback = torch.zeros(1, dtype=target.dtype, device=target.device)
    return preds_fallback, target_fallback


def calc_confidence_interval(e):
    mean = e.mean()
    ci_low, ci_high = stats.t.interval(
        confidence=0.95,
        df=len(e)-1,
        loc=mean,
        scale=stats.sem(e)  # std / sqrt(n)
    )
    return {'ci_low': ci_low, 'ci_high': ci_high, 'mean': mean}


class ZeroAwareMetric(Metric):
    def __init__(self, base_metric: Metric):
        super().__init__()
        self.base_metric = base_metric()
        self.history = []


    def update(self, preds: Tensor, target: Tensor) -> None:
        filtered_preds, filtered_target = zero_aware_filter(preds, target)
        self.history.append((filtered_preds, filtered_target))
        self.base_metric.update(filtered_preds, filtered_target)

    def compute(self):
        return self.base_metric.compute()

    def reset(self) -> None:
        self.base_metric.reset()
    
    def ci(self):
        errors = []
        for pred, gt in self.history:
            self.base_metric.reset()
            e = self.base_metric(pred,gt)
            errors.append(e.item())
        errors_np = np.array(errors)
        return calc_confidence_interval(errors_np)
            


class NormalizeNonZero(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        mask = x == 0
        x -= self.mean
        x /= self.std
        x[mask] = 0
        return x.to(torch.float32)

    def denorm(self, x):
        mask = x == 0
        x *= self.std
        x += self.mean
        x[mask] = 0
        return x
