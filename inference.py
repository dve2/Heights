from src.model import load
from src.dataset import DoubleMaskDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


model = load("weights/epoch=931-step=11184.ckpt")


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

transforms = A.Compose(
    [
        A.Normalize([8.489298], [9.06547]),
        A.CenterCrop(192, 192),
        ToTensorV2(),
    ],
    additional_targets={'mask2': 'mask'}
)

nnz = NormalizeNonZero(3.016509424749255, 2.452459479074767)
ds = DoubleMaskDataset("tests/data", transform=transforms)

im_masks, img, mask, meta = ds[0]
pred = model(im_masks.unsqueeze(0))
d_pred = nnz.denorm(pred)
d_pred = d_pred.squeeze(0).squeeze(0)