import csv
import json
from datetime import datetime
from pathlib import Path
import torch

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

def prepare_run_dir(run_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs") / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
