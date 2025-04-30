import torch
import  numpy as np
import segmentation_models_pytorch as smp
from src.h_norm import NormalizeNonZero
from dataset_inst import BaseDataset
import albumentations as A
import cv2

def check_neigh(mask, coord):
    neighbours = []
    y0,x0 = coord
    Ny, Nx = mask.shape
    ymin = max(0, y0-1)
    ymax = min(y0+1, Ny-1)
    xmin = max(0, x0-1)
    xmax = min(x0+1, Nx-1)
    for y in range(ymin, ymax+1):
        for x in range(xmin, xmax+1):
            if mask[y][x] != 0:
                neighbours.append((y,x))
    return neighbours

def crop192(image, xmin, ymin):
    return image[ymin:ymin+192, xmin:xmin+192]


def remain_max_dots(mask, image):
    Ny, Nx = mask.shape
    obj_coord = []
    for i in range(Ny):
        for j in range(Nx):
            if mask[i][j] != 0:
                obj_coord.append((i, j))

    d = {}
    for i in range(len(obj_coord)):
        d[obj_coord[i]] = check_neigh(mask, obj_coord[i])

    for i in range(len(obj_coord)):
        keys = []
        for key in d:
            lc = d[key]
            if obj_coord[i] in lc:
                keys.append(key)  # собираем все ключи, где есть элемент obj_coord[i]
        for i2, elmt in enumerate(keys):
            if i2 >= 1:
                d[keys[0]].extend(d.pop(
                    elmt)) 

    for key in d:
        d[key] = list(set(d[key]))

    dotted_mask_from_mask = np.zeros((Ny, Nx), dtype=np.float32)
    for key in d:
        obj_coord = d[key]
        z_values = [image[obj_coord[i][0]][obj_coord[i][1]] for i in range(len(obj_coord))]
        max_val, idx = max((v, i) for i, v in enumerate(z_values))
        dotted_mask_from_mask[obj_coord[idx][0]][obj_coord[idx][1]] = 1
    return torch.from_numpy(dotted_mask_from_mask)


def main():
    model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights=None,
    in_channels=1,
    classes=2,
    )


    model_weights = torch.load("weights/model1.pt",weights_only=True)
    model.load_state_dict(model_weights)
    model.eval()

    model2 = smp.Unet(
    encoder_name="efficientnet-b0",  
    encoder_weights=None,  
    in_channels=2,  
    classes=1,  
    )

    model2_weights = torch.load("weights/model2.pt",weights_only=True)
    model2.load_state_dict(model2_weights)
    model2.eval()


    #Normalization by the dotted mask
    dot_target_mean, dot_target_std = 3.016509424749255, 2.452459479074767
    nnz = NormalizeNonZero(dot_target_mean, dot_target_std)


    mean, std = [8.489298], [9.06547]



    val_transforms = A.Compose(
    [
        A.Normalize(mean, std),
        #A.CenterCrop(192, 192),
        #ToTensorV2(),
    ],
    )
    ds_inference = BaseDataset(root_dir = "Inference", transform  = val_transforms)
    image, meta = ds_inference[0]


    Ny, Nx = image.shape
    k=10

    x_mesh = [i*(192-2*k) for i in range(Nx//(192-2*k)+1)]
    x_mesh[-1] = (Nx - 192)

    y_mesh = [i*(192-2*k) for i in range(Ny//(192-2*k)+1)]
    y_mesh[-1] = (Ny - 192)

    pred_whole_image = torch.empty((Ny, 0), dtype=torch.float32)
    for i, xmin in enumerate(x_mesh):
        column = torch.empty((0, 192), dtype=torch.float32)
        for j, ymin in enumerate(y_mesh):
            cropped = crop192(image, xmin,
                              ymin)  # numpy.ndarray (192, 192)    #crops 192*192 fragment starting from ymin line, xmin column
            cropped_torch = torch.from_numpy(cropped)  # torch.Size([192, 192])
            out = model(cropped_torch.unsqueeze(0).unsqueeze(
                0)).detach().cpu()  # torch.Size([1, 2, 192, 192])    #predicts globules
            out_merged = out.squeeze(0).argmax(0)  # torch.Size([192, 192])    #creates mask of globules from the prediction
            out_merged = remain_max_dots(out_merged.numpy(),
                                         cropped)  # torch.Size([192, 192])
            cropped_torch = cropped_torch.unsqueeze(0)  # torch.Size([1, 192, 192])
            out_merged = out_merged.unsqueeze(0)  # torch.Size([1, 192, 192])
            im_mask = torch.cat((cropped_torch, out_merged),
                                0)  # torch.Size([1, 192, 192]) #creates two channel tensor, where 1st
            # channel is image, 2nd channel is mask
            hpred = model2(im_mask.unsqueeze(0)).detach().cpu()  # torch.Size([1, 1, 192, 192])
            d_hpred = nnz.denorm(hpred)  # torch.Size([1, 1, 192, 192])
            d_hpred = d_hpred.squeeze(0).squeeze(0)  # torch.Size([192, 192])
            d_hpred[out_merged.squeeze(0) == 0] = 0  # torch.Size([192, 192]) height map

            if j == 0:
                # column = torch.cat((column, d_hpred[:(192-k),:]), 0)
                d_hpred = d_hpred[:(192 - k), :]
            if (j > 0 and j < (len(y_mesh) - 1)):
                d_hpred = d_hpred[k:(192 - k), :]
            if j == (len(y_mesh) - 1):
                lsp = y_mesh[-2] + 192 - y_mesh[-1] - k
                d_hpred = d_hpred[lsp:, :]

            column = torch.cat((column, d_hpred), 0) 
        if i == 0:
            column = column[:, :(192 - k)]
        if (i > 0 and i < (len(x_mesh) - 1)):
            column = column[:, k:(192 - k)]
        if i == (len(x_mesh) - 1):
            lsp = x_mesh[-2] + 192 - x_mesh[-1] - k
            column = column[:, lsp:]
        pred_whole_image = torch.cat((pred_whole_image, column),
                                     1) 
    result = (pred_whole_image.detach()).long().numpy()
    cv2.imwrite('results/tmp.png', result)


if __name__ == '__main__':
    main()
