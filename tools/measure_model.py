import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.resnet_cifar_custom4stage import ResNet as Custom4StageResNet
from models.resnet20_cifar import ResNet20
from models.resnet8_cifar import ResNet8


def build_model(model_name, model_kwargs=None):
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    if model_name == "custom4stage":
        return Custom4StageResNet(**model_kwargs)
    if model_name == "resnet20":
        return ResNet20(**model_kwargs)
    if model_name == "resnet8":
        return ResNet8(**model_kwargs)
    raise ValueError(f"Unknown model_name: {model_name}")


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def measure_latency(model, device, batch_size, warmup=20, iters=100):
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    for _ in range(warmup):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()
    total_s = end - start

    return {
        "batch_size": batch_size,
        "iters": iters,
        "total_time_s": total_s,
        "avg_time_per_batch_ms": 1000.0 * total_s / iters,
        "avg_time_per_image_ms": 1000.0 * total_s / (iters * batch_size)
    }


def default_output_path(config_path, config):
    config_path = Path(config_path)
    drive_run_root = config.get("drive_run_root")

    if drive_run_root:
        measurements_root = Path(drive_run_root).parent / "measurements"
    else:
        measurements_root = Path("/content/drive/MyDrive/CIFAR-10/CIFAR10_runs/measurements")

    measurements_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    device_tag = "cuda" if torch.cuda.is_available() else "cpu"
    filename = f"{timestamp}_{config_path.stem}_{device_tag}_measure.json"
    return measurements_root / filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_model(config["model_name"], config.get("model_kwargs")).to(device)

    result = {
        "config": args.config,
        "run_name": config.get("run_name"),
        "model_name": config["model_name"],
        "model_kwargs": config.get("model_kwargs", {}),
        "device": str(device),
        "param_count": count_trainable_parameters(model),
        "latency_bs1": measure_latency(model, device, batch_size=1, warmup=args.warmup, iters=args.iters),
        "latency_bs64": measure_latency(model, device, batch_size=64, warmup=args.warmup, iters=args.iters)
    }

    out_path = Path(args.out) if args.out else default_output_path(args.config, config)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
