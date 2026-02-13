from heights.model import load
from heights.dataset2chdm import CustomDataset2chdm, FixedCropTestDataset
from heights.metrics import ZeroAwareMetric, NormalizeNonZero
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import lightning as L
import torch
from torchmetrics import MetricCollection
from torchmetrics import MeanSquaredError, MeanAbsoluteError

test_transforms = A.Compose(
    [
        A.Normalize([8.489298], [9.06547]),
        ToTensorV2(),
    ]
)

nnz = NormalizeNonZero(3.016509424749255, 2.452459479074767)
target_transform = T.Compose([nnz])
ds = CustomDataset2chdm("tests/data", transform=test_transforms, target_transform=target_transform)
ds_test_cropped = FixedCropTestDataset(ds)
loader_test = DataLoader(ds_test_cropped, batch_size=4, shuffle=False, num_workers=2)

class Lit(L.LightningModule):
    def __init__(self, model, lr=0.01):
        super().__init__()
        self.model = model
        self.lr = lr
        self.save_hyperparameters()
        self.metric_test = MetricCollection(
            {
                "MSE": ZeroAwareMetric(MeanSquaredError),
                "MAE": ZeroAwareMetric(MeanAbsoluteError),
            }
        )
        self.metric_test_dn = self.metric_test.clone()

    
    def get_prediction(self,batch):
        im_masks, imgs, masks, meta = batch
        predicted_masks = self.model(im_masks).squeeze(1)
        predicted_masks = self.postprocess(predicted_masks, masks)
        return predicted_masks

    def postprocess(self, pred, mask):
        pred = pred.squeeze(1)
        pred[mask == 0] = 0
        return pred

    def test_step(self, batch, batch_idx):
        im_masks, imgs, masks, _ = batch
        predicted_masks = self.get_prediction(batch)
        self.metric_test.update(predicted_masks, masks)
        self.metric_test_dn.update(nnz.denorm(predicted_masks.clone()), nnz.denorm(masks.clone()))

    def on_test_epoch_end(self):
        metrics = {f"{name}/test": value for name, value in self.metric_test.compute().items()}
        metrics_dn = {f"{name}/test_dn": value for name, value in self.metric_test_dn.compute().items()}
       
        self.log_dict(metrics, prog_bar=True)
        self.log_dict(metrics_dn, prog_bar=True)
        self.log_confidence_intervals(self.metric_test, "test")
        self.log_confidence_intervals(self.metric_test_dn, "test_dn")
        

        self.metric_test.reset()
        self.metric_test_dn.reset()
    
    def log_confidence_intervals(self, metric_collection,postfix, log = False):
        ci_metrics = {}
        for name, metric in metric_collection.items():
            d = metric.ci()
            for key , value in d.items():
                ci_metrics[f"{name}/{postfix}_{key}"] = torch.tensor(value, device="cpu")
            print(f"{name}/{postfix} 95% CI: ({d['ci_low']:.6f}, {d['ci_high']:.6f}) mean={d['mean']:.6f}")
        if log:
            self.log_dict(ci_metrics, prog_bar=False)



lit_model1 = Lit.load_from_checkpoint("weights/epoch=931-step=11184.ckpt", weights_only=False)



trainer = L.Trainer()
trainer.test(model=lit_model1, dataloaders=loader_test)
