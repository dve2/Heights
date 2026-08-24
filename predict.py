import argparse
import time

import torch

from heights.metrics import collect_inference_metrics, print_inference_metrics
from heights.utils import load_models, predict_file, resolve_device


def main():
    args = parse_args()
    device = resolve_device(args.device)
    total_t0 = time.perf_counter()

    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    areas_model, height_model = load_models(
        args.areas_model_checkpoint,
        args.height_model_checkpoint,
        device,
    )
    prediction_t0 = time.perf_counter()
    record = predict_file(
        args.input_file,
        args.output_folder,
        areas_model,
        height_model,
        device,
    )

    if args.profile_metrics:
        metrics = collect_inference_metrics(
            device=device,
            records=[record],
            total_sec=time.perf_counter() - total_t0,
            extra_preprocess_sec=prediction_t0 - total_t0,
        )
        print_inference_metrics(metrics)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict globular object heights")
    parser.add_argument(
        "--input-file",
        default="tests/inference/gluncl21-2.0_00013.txt",
        help="Path to one microscope .txt input file",
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
    parser.add_argument(
        "--profile-metrics",
        action="store_true",
        help="Print timing and memory usage metrics for the full pipeline.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
