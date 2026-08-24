from torchmetrics import MeanSquaredError, MeanAbsoluteError, Metric
from torch import Tensor
import torch
import numpy as np
from scipy import stats
import platform
import resource
import sys

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


def get_peak_rss_mb():
    """Return peak resident set size in MiB for the current process."""
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak_rss / (1024 * 1024)
    return peak_rss / 1024


def get_hardware_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)

    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor().strip() or platform.uname().processor.strip() or "CPU"


def collect_inference_metrics(
    device,
    records,
    total_sec,
    images_count=None,
    extra_preprocess_sec=0.0,
):
    peak_memory_mb = get_peak_rss_mb()
    if device.type == "cuda":
        peak_memory_mb += torch.cuda.max_memory_reserved(device) / (1024 * 1024)

    return {
        "device": str(device),
        "hardware_name": get_hardware_name(device),
        "images_count": images_count,
        "tiles_count": sum(record["tiles_count"] for record in records),
        "total_sec": total_sec,
        "preprocess_sec": extra_preprocess_sec + sum(
            record["preprocess_sec"] for record in records
        ),
        "inference_sec": sum(record["inference_sec"] for record in records),
        "postprocess_sec": sum(record["postprocess_sec"] for record in records),
        "peak_memory_mb": peak_memory_mb,
    }


def print_inference_metrics(metrics):
    tiles = metrics["tiles_count"]
    inference_sec = metrics["inference_sec"]
    tiles_per_sec = tiles / inference_sec if inference_sec > 0 else 0.0

    print("\n=== Inference profiling ===")
    print(f"Device: {metrics['device']}")
    print(f"Hardware: {metrics['hardware_name']}")
    if metrics["images_count"] is not None:
        print(f"Images processed: {metrics['images_count']}")
    print(f"Tiles processed: {tiles}")
    print(f"Total pipeline time: {metrics['total_sec']:.3f} s")
    print(f"Preprocess time: {metrics['preprocess_sec']:.3f} s")
    print(f"Inference loop time: {inference_sec:.3f} s")
    print(f"Postprocess/save time: {metrics['postprocess_sec']:.3f} s")
    print(f"Throughput: {tiles_per_sec:.2f} tiles/s")
    print(f"Peak memory usage: {metrics['peak_memory_mb']:.2f} MiB")
    print("===========================\n")
