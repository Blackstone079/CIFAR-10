import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch


def validate_run_roots(run_root, drive_run_root=None):
    run_root = Path(run_root).as_posix()
    if run_root.startswith("/content/drive/"):
        raise ValueError("run_root must be a local path such as /content/CIFAR10_runs/full. Use drive_run_root for the Google Drive mirror.")
    if drive_run_root is not None:
        Path(drive_run_root).mkdir(parents=True, exist_ok=True)


def _atomic_tmp_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, path.with_name(path.name + ".tmp")


def save_json(path, obj):
    path, tmp = _atomic_tmp_path(path)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_text(path, text):
    path, tmp = _atomic_tmp_path(path)
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_metrics_row(csv_path, row_dict):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)
        f.flush()
        os.fsync(f.fileno())


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_acc, config):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
        "config": config,
    }
    path, tmp = _atomic_tmp_path(path)
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


def prepare_run_dir(run_root, run_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(run_root) / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def prepare_drive_run_dir(drive_run_root, run_dir_name):
    if not drive_run_root:
        return None
    drive_run_dir = Path(drive_run_root) / run_dir_name
    drive_run_dir.mkdir(parents=True, exist_ok=True)
    return drive_run_dir


def mirror_file(src_path, dst_dir):
    if dst_dir is None:
        return

    src_path = Path(src_path)
    if not src_path.exists():
        return

    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_path = dst_dir / src_path.name
    tmp_path = dst_path.with_name(dst_path.name + ".tmp")
    shutil.copy2(src_path, tmp_path)
    os.replace(tmp_path, dst_path)


def mirror_run_files(local_run_dir, drive_run_dir, filenames):
    if drive_run_dir is None:
        return
    for name in filenames:
        mirror_file(Path(local_run_dir) / name, drive_run_dir)
