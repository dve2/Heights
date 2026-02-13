import unittest
import numpy as np
import torch

from heights.metrics import calc_confidence_interval, ZeroAwareMetric
from torchmetrics import MeanSquaredError


class MetricsCITest(unittest.TestCase):
    def test_calc_confidence_interval_contains_mean(self):
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        low, high, mean = calc_confidence_interval(errors)

        self.assertTrue(np.isclose(mean, 3.0))
        self.assertLess(low, mean)
        self.assertGreater(high, mean)

    def test_zero_aware_mse_ci_method(self):
        metric = ZeroAwareMetric(MeanSquaredError)
        metric.update(torch.tensor([1.0, 2.0, 0.0]), torch.tensor([1.0, 1.0, 0.0]))
        metric.update(torch.tensor([2.0, 4.0, 0.0]), torch.tensor([1.0, 1.0, 0.0]))

        low, high, mean = metric.ci()

        self.assertTrue(np.isfinite(low))
        self.assertTrue(np.isfinite(high))
        self.assertTrue(np.isfinite(mean))
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)


if __name__ == "__main__":
    unittest.main()
