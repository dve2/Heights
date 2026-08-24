from scipy import ndimage
import numpy as np
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import os
import segmentation_models_pytorch as smp
import time
import torch
from tqdm import tqdm

from heights.dataset import BaseDataset
from heights.metrics import NormalizeNonZero



def get_max_inside_blobs(image, mask):
    labels, num_labels = ndimage.label(mask)

    # Get max for each blob (indices 1 to num_labels inclusive)
    max_values = ndimage.maximum(image, labels=labels, index=np.arange(1, num_labels + 1))
    max_points_2d = np.zeros_like(image)

    # Populate the 2D array with max values for each blob
    for label in range(1, num_labels + 1):
        y, x = np.where(labels == label)  # Coordinates of the blob
        if len(y) == 0:
            continue
        max_idx = np.argmax(image[y, x])  # Index of the max value within the blob
        y_max = y[max_idx]
        x_max = x[max_idx]
        max_points_2d[y_max, x_max] = image[y_max, x_max]
    return max_points_2d


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("`--device cuda` requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def check_neigh(mask, coord):
    neighbours = []
    y0, x0 = coord
    ny, nx = mask.shape
    ymin = max(0, y0 - 1)
    ymax = min(y0 + 1, ny - 1)
    xmin = max(0, x0 - 1)
    xmax = min(x0 + 1, nx - 1)
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            if mask[y][x] != 0:
                neighbours.append((y, x))
    return neighbours


def crop192(image, xmin, ymin):
    return image[ymin:ymin + 192, xmin:xmin + 192]


def remain_max_dots(mask, image):
    ny, nx = mask.shape
    obj_coord = []
    for i in range(ny):
        for j in range(nx):
            if mask[i][j] != 0:
                obj_coord.append((i, j))

    neighbours_by_coord = {}
    for coord in obj_coord:
        neighbours_by_coord[coord] = check_neigh(mask, coord)

    for coord in obj_coord:
        keys = []
        for key in neighbours_by_coord:
            if coord in neighbours_by_coord[key]:
                keys.append(key)
        for key_index, key in enumerate(keys):
            if key_index >= 1:
                neighbours_by_coord[keys[0]].extend(neighbours_by_coord.pop(key))

    for key in neighbours_by_coord:
        neighbours_by_coord[key] = list(set(neighbours_by_coord[key]))

    dotted_mask = np.zeros((ny, nx), dtype=np.float32)
    for coordinates in neighbours_by_coord.values():
        z_values = [image[y][x] for y, x in coordinates]
        _, max_index = max((value, index) for index, value in enumerate(z_values))
        y, x = coordinates[max_index]
        dotted_mask[y][x] = 1
    return torch.from_numpy(dotted_mask)


def save_prediction(pred_whole_image, filename):
    all_heights = []
    for i in range(len(pred_whole_image)):
        for j in range(len(pred_whole_image[0])):
            if pred_whole_image[i][j]:
                all_heights.append(pred_whole_image[i][j].item())

    with open(f"{filename}_N={len(all_heights)}.txt", "w") as output_file:
        output_file.writelines(f"{item}\n" for item in all_heights)

    plt.figure(figsize=(5, 4))
    plt.suptitle(f"Heights {filename}, N = {len(all_heights)}")
    plt.xlabel("Height, nm")
    plt.ylabel("Number of particles")
    plt.hist(all_heights, bins=20)
    plt.savefig(f"{filename}_hist.png")
    plt.close()


def load_unet_state_dict(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a state_dict")

    if any(key.startswith("model.") for key in state):
        state = {
            key[len("model."):]: value
            for key, value in state.items()
            if key.startswith("model.")
        }

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()


def load_models(areas_checkpoint, height_checkpoint, device):
    areas_model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=1,
        classes=2,
    )
    load_unet_state_dict(areas_model, areas_checkpoint, device)

    height_model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=2,
        classes=1,
    )
    load_unet_state_dict(height_model, height_checkpoint, device)
    return areas_model, height_model


def predict_file(input_file, output_folder, areas_model, height_model, device):
    """Predict and save one image, returning phase durations and tile count."""
    preprocess_t0 = time.perf_counter()
    normalizer = NormalizeNonZero(3.016509424749255, 2.452459479074767)
    transforms = A.Compose([A.Normalize([8.489298], [9.06547])])
    dataset_parser = BaseDataset(root_dir=".", transform=None)
    image, _ = dataset_parser.txt2pil(input_file)
    image = transforms(image=image)["image"]
    base_filename = os.path.splitext(os.path.basename(input_file))[0]

    ny, nx = image.shape
    overlap = 10
    x_mesh = [i * (192 - 2 * overlap) for i in range(nx // (192 - 2 * overlap) + 1)]
    x_mesh[-1] = nx - 192
    y_mesh = [i * (192 - 2 * overlap) for i in range(ny // (192 - 2 * overlap) + 1)]
    y_mesh[-1] = ny - 192
    preprocess_t1 = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_t0 = time.perf_counter()

    pred_whole_image = torch.empty((ny, 0), dtype=torch.float32)
    tiles_count = 0
    for i, xmin in tqdm(enumerate(x_mesh), total=len(x_mesh)):
        column = torch.empty((0, 192), dtype=torch.float32)
        for j, ymin in enumerate(y_mesh):
            tiles_count += 1
            cropped = crop192(image, xmin, ymin)
            cropped_torch = torch.from_numpy(cropped).to(device)
            out = areas_model(cropped_torch.unsqueeze(0).unsqueeze(0)).detach().cpu()
            out_merged = out.squeeze(0).argmax(0)
            out_merged = remain_max_dots(out_merged.numpy(), cropped)
            cropped_torch = cropped_torch.unsqueeze(0)
            out_merged = out_merged.unsqueeze(0)
            image_mask = torch.cat((cropped_torch, out_merged.to(device)), 0)
            height_prediction = height_model(image_mask.unsqueeze(0).to(device)).detach().cpu()
            denormalized = normalizer.denorm(height_prediction).squeeze(0).squeeze(0)
            denormalized[out_merged.squeeze(0) == 0] = 0

            if j == 0:
                denormalized = denormalized[:192 - overlap, :]
            if 0 < j < len(y_mesh) - 1:
                denormalized = denormalized[overlap:192 - overlap, :]
            if j == len(y_mesh) - 1:
                last_start = y_mesh[-2] + 192 - y_mesh[-1] - overlap
                denormalized = denormalized[last_start:, :]
            column = torch.cat((column, denormalized), 0)

        if i == 0:
            column = column[:, :192 - overlap]
        if 0 < i < len(x_mesh) - 1:
            column = column[:, overlap:192 - overlap]
        if i == len(x_mesh) - 1:
            last_start = x_mesh[-2] + 192 - x_mesh[-1] - overlap
            column = column[:, last_start:]
        pred_whole_image = torch.cat((pred_whole_image, column), 1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_t1 = time.perf_counter()

    os.makedirs(output_folder, exist_ok=True)
    save_prediction(
        pred_whole_image.detach(),
        os.path.join(output_folder, f"Heights {base_filename}"),
    )
    result = pred_whole_image.detach().cpu().numpy()
    result_vis = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_folder, f"{base_filename}_img.png"), result_vis)
    print(f"Processed {input_file}, saved heights and visualization to {output_folder}{os.sep}")
    postprocess_t1 = time.perf_counter()
    return {
        "tiles_count": tiles_count,
        "preprocess_sec": preprocess_t1 - preprocess_t0,
        "inference_sec": inference_t1 - inference_t0,
        "postprocess_sec": postprocess_t1 - inference_t1,
    }
