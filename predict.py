from email import parser
import torch
import  numpy as np
import segmentation_models_pytorch as smp
from heights.metrics import NormalizeNonZero
from heights.dataset import BaseDataset
import albumentations as A
import cv2
import os
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import time
import resource
import sys
import platform


"""
    Model work on 192*192 fragments of the image. This script crops the image into 192*192 fragments,
    predict heights on them and then merges the predictions into one whole image.

"""


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


def save(pred_whole_image, filename):
    maxcoordinates = []
    all_heights = []
    for i in range(len(pred_whole_image)):
        for j in range(len(pred_whole_image[0])):
            if pred_whole_image[i][j]:
                all_heights.append(pred_whole_image[i][j].item())
                maxcoordinates.append((i, j))

    with open(f"{filename}_N={len(all_heights)}.txt", 'w') as f:
        f.writelines(f"{item}\n" for item in all_heights)

    plt.figure(figsize=(5,4))
    plt.suptitle(f"Heights {filename}, N = {len(all_heights)}")
    plt.xlabel('Height, nm')
    plt.ylabel('Number of particles')
    plt.hist(all_heights, bins=20)
    plt.savefig(f"{filename}_hist.png")



def load_unet_state_dict(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {ckpt_path} does not contain a state_dict")

    # Lightning module saved params under "model.*"
    if any(k.startswith("model.") for k in state):
        state = {
            k[len("model."):]: v
            for k, v in state.items()
            if k.startswith("model.")
        }

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()


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
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass

    cpu_name = platform.processor().strip()
    if cpu_name:
        return cpu_name

    cpu_name = platform.uname().processor.strip()
    if cpu_name:
        return cpu_name

    return "CPU"


def collect_metrics(device, tiles_count, total_t0, preprocess_t1, infer_t0, infer_t1):
    total_t1 = time.perf_counter()
    peak_memory_mb = get_peak_rss_mb()
    if device.type == "cuda":
        peak_memory_mb += torch.cuda.max_memory_reserved(device) / (1024 * 1024)

    return {
        "device": str(device),
        "hardware_name": get_hardware_name(device),
        "tiles_count": tiles_count,
        "total_sec": total_t1 - total_t0,
        "preprocess_sec": preprocess_t1 - total_t0,
        "inference_sec": infer_t1 - infer_t0,
        "postprocess_sec": total_t1 - infer_t1,
        "peak_memory_mb": peak_memory_mb,
    }


def print_metrics(metrics):
    tiles = metrics["tiles_count"]
    inf_sec = metrics["inference_sec"]
    total_sec = metrics["total_sec"]
    tiles_per_sec = (tiles / inf_sec) if inf_sec > 0 else 0.0

    print("\n=== Inference profiling ===")
    print(f"Device: {metrics['device']}")
    print(f"Hardware: {metrics['hardware_name']}")
    print(f"Tiles processed: {tiles}")
    print(f"Total pipeline time: {total_sec:.3f} s")
    print(f"Preprocess time: {metrics['preprocess_sec']:.3f} s")
    print(f"Inference loop time: {inf_sec:.3f} s")
    print(f"Postprocess/save time: {metrics['postprocess_sec']:.3f} s")
    print(f"Throughput: {tiles_per_sec:.2f} tiles/s")
    print(f"Peak memory usage: {metrics['peak_memory_mb']:.2f} MiB")
    print("===========================\n")



def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("`--device cuda` requested, but CUDA is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    total_t0 = time.perf_counter()

    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights=None,
    in_channels=1,
    classes=2,
    )

    load_unet_state_dict(model, args.areas_model_checkpoint, device)
    model.eval()

    model2 = smp.Unet(
    encoder_name="efficientnet-b0",  
    encoder_weights=None,  
    in_channels=2,  
    classes=1,  
    )

    load_unet_state_dict(model2, args.height_model_checkpoint, device)
    model2.eval()


    # Normalization by the dotted mask
    dot_target_mean, dot_target_std = 3.016509424749255, 2.452459479074767
    nnz = NormalizeNonZero(dot_target_mean, dot_target_std)

    mean, std = [8.489298], [9.06547]

    val_transforms = A.Compose(
    [
        A.Normalize(mean, std),
    ],
    )
    ds_parser = BaseDataset(root_dir=".", transform=None)
    image, _ = ds_parser.txt2pil(args.input_file)
    image = val_transforms(image=image)["image"]
    base_filename = os.path.splitext(os.path.basename(args.input_file))[0]
    preprocess_t1 = time.perf_counter()


    Ny, Nx = image.shape
    k=10

    x_mesh = [i*(192-2*k) for i in range(Nx//(192-2*k)+1)]
    x_mesh[-1] = (Nx - 192)

    y_mesh = [i*(192-2*k) for i in range(Ny//(192-2*k)+1)]
    y_mesh[-1] = (Ny - 192)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    infer_t0 = time.perf_counter()

    pred_whole_image = torch.empty((Ny, 0), dtype=torch.float32)
    tiles_count = 0
    for i, xmin in tqdm(enumerate(x_mesh)):
        column = torch.empty((0, 192), dtype=torch.float32)
        for j, ymin in enumerate(y_mesh):
            tiles_count += 1
            cropped = crop192(image, xmin,
                              ymin)  # numpy.ndarray (192, 192)    #crops 192*192 fragment starting from ymin line, xmin column
            cropped_torch = torch.from_numpy(cropped).to(device)  # torch.Size([192, 192])
            out = model(cropped_torch.unsqueeze(0).unsqueeze(
                0)).detach().cpu()  # torch.Size([1, 2, 192, 192])    #predicts globules
            out_merged = out.squeeze(0).argmax(0)  # torch.Size([192, 192])    #creates mask of globules from the prediction
            out_merged = remain_max_dots(out_merged.numpy(),
                                         cropped)  # torch.Size([192, 192])
            cropped_torch = cropped_torch.unsqueeze(0)  # torch.Size([1, 192, 192])
            out_merged = out_merged.unsqueeze(0)  # torch.Size([1, 192, 192])
            im_mask = torch.cat((cropped_torch, out_merged.to(device)),
                                0)  # torch.Size([1, 192, 192]) #creates two channel tensor, where 1st
            # channel is image, 2nd channel is mask
            hpred = model2(im_mask.unsqueeze(0).to(device)).detach().cpu()  # torch.Size([1, 1, 192, 192])
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
        pred_whole_image = torch.cat((pred_whole_image, column),1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    infer_t1 = time.perf_counter()

    # Dump results
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    save(pred_whole_image.detach(), f"{output_folder}{os.sep}Heights {base_filename}")

    result = pred_whole_image.detach().cpu().numpy()
    result_vis = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(f"{output_folder}{os.sep}{base_filename}_img.png", result_vis)
    print(f"Processed {args.input_file}, saved heights and visualization to {output_folder}{os.sep}")

    if args.profile_metrics:
        metrics = collect_metrics(
            device=device,
            tiles_count=tiles_count,
            total_t0=total_t0,
            preprocess_t1=preprocess_t1,
            infer_t0=infer_t0,
            infer_t1=infer_t1,
        )
        print_metrics(metrics)
    


def parse_args():
    parser = argparse.ArgumentParser(description="Predict globular object heights")
    parser.add_argument(
        "--input-file",
        default="tests/inference/gluncl21-2.0_00013.txt",
        help="Path to one microscope.txt input file",
    )

    parser.add_argument(
        "--areas_model_checkpoint",
        default="weights/Areas_epoch=691-step=4152(1).ckpt",
        help="Path to `Areas` model  weights file",
    )

    parser.add_argument("--height_model_checkpoint",
        default="weights/Heights_epoch=4993-step=59928.ckpt",
        help="Path to Heights - model checkpoint file",)
    
    parser.add_argument(
        "--output-folder",
        default="results",
        help="Folder where prediction outputs will be saved",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device: auto (prefer CUDA), cpu, or cuda.",
    )

    parser.add_argument(
        "--profile-metrics",
        action="store_true",
        help="Print timing and memory usage metrics for the full pipeline.",
    )
        
    return parser.parse_args()


if __name__ == '__main__':
    main()
