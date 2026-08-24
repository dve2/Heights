import argparse
from pathlib import Path
import time

import torch

from heights.metrics import collect_inference_metrics, print_inference_metrics
from heights.utils import load_models, predict_file, resolve_device


def find_input_files(input_folder):
    folder = Path(input_folder)
    if not folder.is_dir():
        raise ValueError(f"Input folder does not exist: {folder}")

    input_files = sorted(folder.glob("*.txt"))
    if not input_files:
        raise ValueError(f"No .txt input files found in: {folder}")
    return input_files


def main():
    args = parse_args()
    try:
        input_files = find_input_files(args.input_folder)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    areas_model, height_model = load_models(
        args.areas_model_checkpoint,
        args.height_model_checkpoint,
        device,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_t0 = time.perf_counter()
    records = []
    for input_file in input_files:
        records.append(
            predict_file(
                str(input_file),
                args.output_folder,
                areas_model,
                height_model,
                device,
            )
        )

    metrics = collect_inference_metrics(
        device=device,
        records=records,
        total_sec=time.perf_counter() - total_t0,
        images_count=len(input_files),
    )
    print_inference_metrics(metrics)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile height prediction over a folder of microscope images"
    )
    parser.add_argument(
        "--input-folder",
        default="tests/data/10_img_profiling",
        help="Folder containing microscope .txt input files",
    )
    parser.add_argument(
        "--areas_model_checkpoint",
        default="weights/Areas_epoch=691-step=4152(1).ckpt",
        help="Path to Areas model weights file",
    )
    parser.add_argument(
        "--height_model_checkpoint",
        default="weights/Heights_epoch=4993-step=59928.ckpt",
        help="Path to Heights model checkpoint file",
    )
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
    return parser.parse_args()


if __name__ == "__main__":
    main()
