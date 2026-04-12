import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_cifar10_dataloaders
from models.resnet_cifar import ResNet

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def append_metrics_row(csv_path, row_dict):
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def save_checkpoint(path, epoch, model, optimizer, scheduler, best_acc, config):
    checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "best_acc": best_acc, "config": config}
    torch.save(checkpoint, path)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    config = {"run_name": "custom4stage_noaug_meeting", "model_name": "resnet_cifar_custom4stage", "batch_size": 256, "test_batch_size": 256, "epochs": 200, "lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4, "scheduler": "MultiStepLR", "milestones": [100, 150], "gamma": 0.1, "eval_every": 5, "data_root": "/content/cifar_data", "num_workers": 2, "pin_memory": (device.type == "cuda"), "device": str(device), "torch_version": torch.__version__}

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs") / f"{timestamp}_{config['run_name']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / "config.json", config)
    with open(run_dir / "notes.txt", "w") as f:
        f.write("Quick logged run before meeting.\n")
        f.write("Current model kept fixed. Added logging/checkpoints and basic throughput fixes.\n")

    train_loader, test_loader = get_cifar10_dataloaders(batch_size=config["batch_size"], test_batch_size=config["test_batch_size"], data_root=config["data_root"], num_workers=config["num_workers"], pin_memory=config["pin_memory"])

    model = ResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["milestones"], gamma=config["gamma"])

    best_acc = -1.0
    total_start = time.time()

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Eval every: {config['eval_every']} epochs")

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()

        train_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_time = time.time() - train_start

        test_loss = ""
        test_acc = ""
        eval_time = 0.0

        do_eval = (epoch == 1) or (epoch % config["eval_every"] == 0) or (epoch == config["epochs"])
        if do_eval:
            eval_start = time.time()
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            eval_time = time.time() - eval_start
            if test_acc > best_acc:
                best_acc = test_acc
                save_checkpoint(run_dir / "best.pt", epoch, model, optimizer, scheduler, best_acc, config)

        epoch_time = time.time() - epoch_start
        total_time = time.time() - total_start
        current_lr = optimizer.param_groups[0]["lr"]

        row = {"epoch": epoch, "lr": current_lr, "train_loss": train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc, "train_time_s": train_time, "eval_time_s": eval_time, "epoch_time_s": epoch_time, "total_time_s": total_time, "best_acc_so_far": best_acc}
        append_metrics_row(run_dir / "metrics.csv", row)
        save_checkpoint(run_dir / "last.pt", epoch, model, optimizer, scheduler, best_acc, config)

        if do_eval:
            print(f"Epoch {epoch:03d}/{config['epochs']} | lr={current_lr:.4f} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | test_loss={test_loss:.4f} | test_acc={test_acc:.4f} | train_time={train_time:.1f}s | eval_time={eval_time:.1f}s | epoch_time={epoch_time:.1f}s")
        else:
            print(f"Epoch {epoch:03d}/{config['epochs']} | lr={current_lr:.4f} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | train_time={train_time:.1f}s | epoch_time={epoch_time:.1f}s")

        scheduler.step()

if __name__ == "__main__":
    main()