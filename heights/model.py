import segmentation_models_pytorch as smp
import torch


model = smp.Unet(
    encoder_name="efficientnet-b0",  # choose encoder
    encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=2,
    classes=1,  # model output channels (number of classes in mask)
)

def get_torch_weights_from_lightning_ckpt(checkpoint):
    from collections import OrderedDict
    # remove 'model.' from all keys
    weights = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if key.startswith("model."):
            weights[key[6:]] = value
        else:
            weights[key] = value
    return weights


def load(path_to_ckpt, device = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path_to_ckpt, map_location=device,weights_only=False)
    weights = get_torch_weights_from_lightning_ckpt(checkpoint)
    model.load_state_dict(weights, strict=False)
    return model

