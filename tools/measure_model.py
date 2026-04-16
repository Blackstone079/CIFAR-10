import argparse
import json
import time

import torch

from models.resnet_cifar_custom4stage import ResNet as Custom4StageResNet
from models.resnet20_cifar import ResNet20
from models.resnet8_cifar import ResNet8


def build_model(model_name):
    if model_name == "custom4stage":
        return Custom4StageResNet()
    if model_name == "resnet20":
        return ResNet20()
    if model_name == "resnet8":
        return ResNet8()
    raise ValueError(f"Unknown model_name: {model_name}")


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def measure_latency(model, batch_size, device, warmup=20, iters=100):
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
        "num_iters": iters,
        "total_time_s": total_s,
        "avg_time_per_batch_ms": 1000.0 * total_s / iters,
        "avg_time_per_image_ms": 1000.0 * total_s / (iters * batch_size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["custom4stage", "resnet20", "resnet8"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_model(args.model).to(device)
    result = {
        "model_name": args.model,
        "device": str(device),
        "param_count": count_trainable_parameters(model),
        "latency": measure_latency(model, args.batch_size, device, warmup=args.warmup, iters=args.iters),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
