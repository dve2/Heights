import unittest
from src.dataset import DoubleMaskDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DatasetTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
