import unittest
import torch
from torch.utils.data import non_deterministic

from src.dataset import DoubleMaskDataset, GroundDataset
from src.utils import get_max_inside_blobs
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from numpy.testing import assert_array_equal

import numpy as np


class DatasetTest(unittest.TestCase):
    @unittest.skip("")
    def test_second_mask(self):
        transforms = A.Compose(
            [
                A.CenterCrop(192, 192),
                ToTensorV2(),
            ],
            additional_targets={'mask2': 'mask'}
        )

        ds = DoubleMaskDataset("tests/data", transform = transforms )
        image, mask1, mask2, _ = ds[0]
        self.assertEqual(mask1.shape, mask2.shape)

    def test_ground_smoke(self):
        ds = GroundDataset("tests/data")
        image, mask, ground, meta = ds[0]

    def test_ground_smoke(self):
        ds = GroundDataset("tests/data")
        image, mask, ground, meta = ds[0]
        naive_target = ds.create_1ch_target(mask, ground)
        np_target = ds.create_1ch_target_v1(mask, ground)
        self.assertTrue(torch.equal(naive_target, np_target))
        pil = Image.fromarray(ground*255)
        pil.save("tmp.png")

    def test_mask2point(self):
        ds = DoubleMaskDataset("tests/data")
        image, mask1, mask2, _ = ds[0]

        max_points_2d = get_max_inside_blobs(image, mask1)
        non_zero_mask = max_points_2d > 0
        self.assertTrue(np.allclose(max_points_2d[non_zero_mask], image[non_zero_mask]))



if __name__ == '__main__':
    unittest.main()
