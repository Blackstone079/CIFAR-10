import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

from data import get_cifar10_dataloaders
from models.resnet_cifar_custom4stage import ResNet as Custom4StageResNet
from models.resnet20_cifar import ResNet20
from models.resnet8_cifar import ResNet8
from utils.run_logging import save_json, save_text, append_metrics_row, save_checkpoint, prepare_run_dir, prepare_drive_run_dir, mirror_run_files, validate_run_roots


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, test_loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def build_model(model_name, model_kwargs=None):
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)

    if model_name == "custom4stage":
        return Custom4StageResNet(**model_kwargs)
    if model_name == "resnet20":
        return ResNet20(**model_kwargs)
    if model_name == "resnet8":
        return ResNet8(**model_kwargs)

    raise ValueError(f"Unknown model_name: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/custom4stage_noaug_bs64.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    validate_run_roots(config["run_root"], config.get("drive_run_root"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    use_amp = (device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)
    pin_memory = (device.type == "cuda") if config["pin_memory"] == "auto" else config["pin_memory"]

    local_run_dir = prepare_run_dir(config["run_root"], config["run_name"])
    drive_run_dir = prepare_drive_run_dir(config.get("drive_run_root"), local_run_dir.name)

    config["device"] = str(device)
    config["torch_version"] = torch.__version__
    config["amp"] = use_amp
    config["local_run_dir"] = str(local_run_dir)
    if drive_run_dir is not None:
        config["drive_run_dir"] = str(drive_run_dir)

    save_json(local_run_dir / "config.json", config)
    save_text(local_run_dir / "notes.txt", "Logged training run.\n")
    mirror_run_files(local_run_dir, drive_run_dir, ["config.json", "notes.txt"])

    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=config["batch_size"],
        test_batch_size=config["test_batch_size"],
        data_root=config["data_root"],
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
        augmentation=config.get("augmentation", False),
        train_subset_size=config.get("train_subset_size"),
        test_subset_size=config.get("test_subset_size"),
        seed=config.get("seed", 0),
    )

    model = build_model(config["model_name"], config.get("model_kwargs")).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])

    if config["scheduler"] == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["milestones"], gamma=config["gamma"])
    else:
        raise ValueError(f"Unknown scheduler: {config['scheduler']}")

    best_acc = -1.0
    total_start = time.time()

    print(f"Local run directory: {local_run_dir}")
    if drive_run_dir is not None:
        print(f"Drive mirror directory: {drive_run_dir}")
    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()

        train_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        train_time = time.time() - train_start

        test_loss = ""
        test_acc = ""
        eval_time = 0.0

        do_eval = (epoch == 1) or (epoch % config["eval_every"] == 0) or (epoch == config["epochs"])
        if do_eval:
            eval_start = time.time()
            test_loss, test_acc = evaluate(model, test_loader, criterion, device, use_amp)
            eval_time = time.time() - eval_start

            if test_acc > best_acc:
                best_acc = test_acc
                save_checkpoint(local_run_dir / "best.pt", epoch, model, optimizer, scheduler, best_acc, config)

        epoch_time = time.time() - epoch_start
        total_time = time.time() - total_start
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "train_time_s": train_time,
            "eval_time_s": eval_time,
            "epoch_time_s": epoch_time,
            "total_time_s": total_time,
            "best_acc_so_far": best_acc,
        }

        append_metrics_row(local_run_dir / "metrics.csv", row)
        save_checkpoint(local_run_dir / "last.pt", epoch, model, optimizer, scheduler, best_acc, config)

        if drive_run_dir is not None and (do_eval or epoch == config["epochs"]):
            mirror_run_files(local_run_dir, drive_run_dir, ["metrics.csv", "last.pt", "best.pt"])

        if do_eval:
            print(f"Epoch {epoch:03d}/{config['epochs']} | lr={current_lr:.4f} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | test_loss={test_loss:.4f} | test_acc={test_acc:.4f} | train_time={train_time:.1f}s | eval_time={eval_time:.1f}s | epoch_time={epoch_time:.1f}s")
        else:
            print(f"Epoch {epoch:03d}/{config['epochs']} | lr={current_lr:.4f} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | train_time={train_time:.1f}s | epoch_time={epoch_time:.1f}s")

        scheduler.step()

    save_text(local_run_dir / "RUN_COMPLETE.txt", "ok\n")
    mirror_run_files(local_run_dir, drive_run_dir, ["config.json", "notes.txt", "metrics.csv", "last.pt", "best.pt", "RUN_COMPLETE.txt"])

    if drive_run_dir is not None:
        print("Drive mirror files:")
        for p in sorted(drive_run_dir.iterdir()):
            print(" ", p.name)


if __name__ == "__main__":
    main()
