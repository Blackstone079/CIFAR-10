import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim

from data import get_cifar10_dataloaders
from models.resnet_cifar_custom4stage import ResNet as Custom4StageResNet
from utils.run_logging import save_json, append_metrics_row, save_checkpoint, prepare_run_dir

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

    return total_loss / total_samples, total_correct / total_samples

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

    return total_loss / total_samples, total_correct / total_samples

def build_model(model_name):
    if model_name == "custom4stage":
        return Custom4StageResNet()
    raise ValueError(f"Unknown model_name: {model_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/custom4stage_noaug_bs64.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    pin_memory = (device.type == "cuda") if config["pin_memory"] == "auto" else config["pin_memory"]

    run_dir = prepare_run_dir(config["run_name"])

    config["device"] = str(device)
    config["torch_version"] = torch.__version__
    save_json(run_dir / "config.json", config)

    with open(run_dir / "notes.txt", "w") as f:
        f.write("Logged training run.\n")

    train_loader, test_loader = get_cifar10_dataloaders(batch_size=config["batch_size"], test_batch_size=config["test_batch_size"], data_root=config["data_root"], num_workers=config["num_workers"], pin_memory=pin_memory)

    model = build_model(config["model_name"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])

    if config["scheduler"] == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["milestones"], gamma=config["gamma"])
    else:
        raise ValueError(f"Unknown scheduler: {config['scheduler']}")

    best_acc = -1.0
    total_start = time.time()

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")

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
