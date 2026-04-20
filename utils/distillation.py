from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

def _maybe_bool(enabled):
    return bool(enabled) if enabled is not None else False

def distillation_enabled(config):
    kd_cfg = config.get("distillation")
    return isinstance(kd_cfg, dict) and _maybe_bool(kd_cfg.get("enabled"))

def load_teacher_from_checkpoint(checkpoint_path, build_model, device):
    checkpoint_path = Path(checkpoint_path)
    if "REPLACE_WITH_" in checkpoint_path.as_posix():
        raise ValueError("distillation.teacher_checkpoint still contains the placeholder path. Replace it with an actual best.pt path first.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain config/model_state_dict.")

    teacher_config = checkpoint["config"]
    teacher = build_model(teacher_config["model_name"], teacher_config.get("model_kwargs")).to(device)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False

    return teacher, teacher_config

def build_distillation(config, build_model, device):
    if not distillation_enabled(config):
        return None

    kd_cfg = dict(config["distillation"])
    alpha = float(kd_cfg.get("alpha", 0.5))
    temperature = float(kd_cfg.get("temperature", 4.0))

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"distillation.alpha must be in [0, 1], got {alpha}")
    if temperature <= 0.0:
        raise ValueError(f"distillation.temperature must be > 0, got {temperature}")

    teacher, teacher_config = load_teacher_from_checkpoint(kd_cfg["teacher_checkpoint"], build_model, device)

    return {
        "teacher": teacher,
        "teacher_config": teacher_config,
        "teacher_checkpoint": str(Path(kd_cfg["teacher_checkpoint"])),
        "alpha": alpha,
        "temperature": temperature,
        "criterion": nn.KLDivLoss(reduction="batchmean"),
    }

def compute_distillation_loss(student_logits, labels, ce_criterion, kd_state, teacher_logits=None):
    ce_loss = ce_criterion(student_logits, labels)

    if kd_state is None:
        return ce_loss, ce_loss, None

    if teacher_logits is None:
        raise ValueError("teacher_logits must be provided when distillation is enabled.")

    temperature = kd_state["temperature"]
    kd_loss = kd_state["criterion"](
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
    ) * (temperature * temperature)

    alpha = kd_state["alpha"]
    total_loss = (1.0 - alpha) * ce_loss + alpha * kd_loss
    return total_loss, ce_loss, kd_loss
