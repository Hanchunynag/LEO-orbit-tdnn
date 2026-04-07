from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import optuna
import torch
from optuna.importance import get_param_importances
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SEARCH_SPACE = {
    "narx_hidden_layers": [1, 2, 3],
    "dense_layers": [1, 2, 3],
    "hidden_width": [8, 10, 16, 32, 64],
    "delay_choices": [50, 100, 200, 400, 800],
    "activation": ["linear", "relu", "tanh", "sigmoid", "snake"],
    "optimizer": ["Adam", "Adagrad", "SGD", "Yogi"],
    "lr_min": 1.0e-4,
    "lr_max": 5.0e-2,
    "batch_size": [32, 64, 128, 256],
    "weight_decay": [0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "loss_type": ["mse", "huber"],
    "scheduler": ["none", "plateau", "cosine"],
    "input_state_mode": ["pos_only", "vel_only", "pos_vel"],
    "feedback_mode": ["residual_feedback", "zero_feedback_baseline"],
}


def load_optional_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path).resolve()
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_rtn_frame(
    reference_pos_eci_m: np.ndarray,
    reference_vel_eci_mps: np.ndarray,
) -> np.ndarray:
    """Return the ECI->RTN rotation matrix using the HPOP reference orbit."""
    r_hat = reference_pos_eci_m / np.linalg.norm(reference_pos_eci_m)
    n_hat = np.cross(reference_pos_eci_m, reference_vel_eci_mps)
    n_hat = n_hat / np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, r_hat)
    t_hat = t_hat / np.linalg.norm(t_hat)
    return np.vstack((r_hat, t_hat, n_hat))


def compute_rtn_residual_if_missing(npz_data: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Current stage: use HPOP as the EKF-corrected ephemeris surrogate
    only for offline single-satellite experiments.
    """
    hpop_eci_pos_m = np.asarray(npz_data["hpop_eci_pos_m"], dtype=np.float64)
    hpop_eci_vel_mps = np.asarray(npz_data["hpop_eci_vel_mps"], dtype=np.float64)

    frames_eci_to_rtn = np.stack(
        [build_rtn_frame(hpop_eci_pos_m[idx], hpop_eci_vel_mps[idx]) for idx in range(hpop_eci_pos_m.shape[0])],
        axis=0,
    )

    if "residual_rtn_pos_m" in npz_data:
        residual_rtn_pos_m = np.asarray(npz_data["residual_rtn_pos_m"], dtype=np.float64)
        return residual_rtn_pos_m, frames_eci_to_rtn

    if "sgp4_to_hpop_pos_rtn_m" in npz_data:
        residual_rtn_pos_m = np.asarray(npz_data["sgp4_to_hpop_pos_rtn_m"], dtype=np.float64)
        return residual_rtn_pos_m, frames_eci_to_rtn

    sgp4_eci_pos_m = np.asarray(npz_data["sgp4_eci_pos_m"], dtype=np.float64)
    residual_eci_pos_m = hpop_eci_pos_m - sgp4_eci_pos_m
    residual_rtn_pos_m = np.einsum("nij,nj->ni", frames_eci_to_rtn, residual_eci_pos_m)
    return residual_rtn_pos_m, frames_eci_to_rtn


def build_pass_masks(npz_data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def maybe_bool_field(*names: str) -> np.ndarray | None:
        for name in names:
            if name in npz_data:
                return np.asarray(npz_data[name], dtype=bool)
        return None

    time_seconds = np.asarray(npz_data["time_seconds"], dtype=np.float64)
    train_mask_first_pass = maybe_bool_field("train_mask_first_pass", "first_pass_mask")
    predict_mask_gap = maybe_bool_field("predict_mask_gap", "prediction_gap_mask")
    eval_mask_second_pass = maybe_bool_field("eval_mask_second_pass", "second_pass_mask")

    if (
        train_mask_first_pass is not None
        and predict_mask_gap is not None
        and eval_mask_second_pass is not None
    ):
        return train_mask_first_pass, predict_mask_gap, eval_mask_second_pass

    pass_windows_time_sec = np.asarray(npz_data["pass_windows_time_sec"], dtype=np.float64)
    if pass_windows_time_sec.ndim != 2 or pass_windows_time_sec.shape != (2, 2):
        raise ValueError(
            f"Current script only supports exactly 2 pass windows; got shape {pass_windows_time_sec.shape}."
        )

    first_start_sec, first_end_sec = pass_windows_time_sec[0]
    second_start_sec, second_end_sec = pass_windows_time_sec[1]
    train_mask_first_pass = (time_seconds >= first_start_sec) & (time_seconds <= first_end_sec)
    predict_mask_gap = (time_seconds > first_end_sec) & (time_seconds < second_start_sec)
    eval_mask_second_pass = (time_seconds >= second_start_sec) & (time_seconds <= second_end_sec)
    return train_mask_first_pass, predict_mask_gap, eval_mask_second_pass


def contiguous_indices_from_mask(mask: np.ndarray, name: str) -> np.ndarray:
    indices = np.where(mask)[0]
    if indices.size == 0:
        raise ValueError(f"{name} is empty.")
    if indices.size > 1 and not np.all(np.diff(indices) == 1):
        raise ValueError(f"{name} must be contiguous for this NARX rollout script.")
    return indices


def compute_norm_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def compute_rtn_position_series(
    frames_eci_to_rtn: np.ndarray,
    positions_eci_m: np.ndarray,
) -> np.ndarray:
    return np.einsum("nij,nj->ni", frames_eci_to_rtn, positions_eci_m)


def prepare_single_satellite_bundle(npz_path: str | Path) -> dict[str, Any]:
    npz_path = Path(npz_path).resolve()
    with np.load(npz_path, allow_pickle=False) as npz_data:
        time_seconds = np.asarray(npz_data["time_seconds"], dtype=np.float64)
        sgp4_eci_pos_m = np.asarray(npz_data["sgp4_eci_pos_m"], dtype=np.float64)
        sgp4_eci_vel_mps = np.asarray(npz_data["sgp4_eci_vel_mps"], dtype=np.float64)
        hpop_eci_pos_m = np.asarray(npz_data["hpop_eci_pos_m"], dtype=np.float64)
        hpop_eci_vel_mps = np.asarray(npz_data["hpop_eci_vel_mps"], dtype=np.float64)
        residual_rtn_pos_m, frames_eci_to_rtn = compute_rtn_residual_if_missing(npz_data)
        train_mask_first_pass, predict_mask_gap, eval_mask_second_pass = build_pass_masks(npz_data)

    dt = float(np.median(np.diff(time_seconds)))
    first_pass_indices = contiguous_indices_from_mask(train_mask_first_pass, "train_mask_first_pass")
    gap_indices = contiguous_indices_from_mask(predict_mask_gap, "predict_mask_gap")
    second_pass_indices = contiguous_indices_from_mask(eval_mask_second_pass, "eval_mask_second_pass")
    hpop_rtn_pos_m = compute_rtn_position_series(frames_eci_to_rtn, hpop_eci_pos_m)

    return {
        "npz_path": str(npz_path),
        "time_seconds": time_seconds,
        "dt": dt,
        "sgp4_eci_pos_m": sgp4_eci_pos_m,
        "sgp4_eci_vel_mps": sgp4_eci_vel_mps,
        "hpop_eci_pos_m": hpop_eci_pos_m,
        "hpop_eci_vel_mps": hpop_eci_vel_mps,
        "frames_eci_to_rtn": frames_eci_to_rtn,
        "residual_rtn_pos_m": residual_rtn_pos_m,
        "train_mask_first_pass": train_mask_first_pass,
        "predict_mask_gap": predict_mask_gap,
        "eval_mask_second_pass": eval_mask_second_pass,
        "first_pass_indices": first_pass_indices,
        "gap_indices": gap_indices,
        "second_pass_indices": second_pass_indices,
        "hpop_rtn_pos_m": hpop_rtn_pos_m,
    }


def build_exogenous_input_series(bundle: dict[str, Any], input_state_mode: str) -> np.ndarray:
    if input_state_mode == "pos_only":
        return bundle["sgp4_eci_pos_m"].astype(np.float64)
    if input_state_mode == "vel_only":
        return bundle["sgp4_eci_vel_mps"].astype(np.float64)
    if input_state_mode == "pos_vel":
        return np.concatenate((bundle["sgp4_eci_pos_m"], bundle["sgp4_eci_vel_mps"]), axis=1).astype(np.float64)
    raise ValueError(f"Unsupported input_state_mode: {input_state_mode}")


def feedback_history_slice(
    y_norm: np.ndarray,
    start_idx: int,
    end_idx: int,
    feedback_mode: str,
) -> np.ndarray:
    history = y_norm[start_idx:end_idx]
    if feedback_mode == "residual_feedback":
        return history
    if feedback_mode == "zero_feedback_baseline":
        return np.zeros_like(history)
    raise ValueError(f"Unsupported feedback_mode: {feedback_mode}")


def build_narx_open_loop_samples(
    u_norm: np.ndarray,
    y_norm: np.ndarray,
    sequence_start_idx: int,
    target_indices: np.ndarray,
    delay: int,
    feedback_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valid_indices: list[int] = []

    for raw_idx in target_indices:
        if raw_idx < sequence_start_idx + delay:
            continue
        u_hist = u_norm[raw_idx - delay + 1 : raw_idx + 1][::-1].reshape(-1)
        y_hist = feedback_history_slice(
            y_norm=y_norm,
            start_idx=raw_idx - delay,
            end_idx=raw_idx,
            feedback_mode=feedback_mode,
        )[::-1].reshape(-1)
        features.append(np.concatenate((u_hist, y_hist), axis=0).astype(np.float32))
        targets.append(y_norm[raw_idx].astype(np.float32))
        valid_indices.append(int(raw_idx))

    if not features:
        raise ValueError(
            "No valid NARX samples were created. Check delay length against first-pass sequence length."
        )

    return (
        np.stack(features, axis=0),
        np.stack(targets, axis=0),
        np.asarray(valid_indices, dtype=np.int64),
    )


class Snake(nn.Module):
    def __init__(self, features: int, init_alpha: float = 1.0) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((features,), math.log(math.expm1(init_alpha))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.nn.functional.softplus(self.log_alpha).unsqueeze(0)
        return x + torch.sin(alpha * x).pow(2) / alpha.clamp_min(1.0e-6)


def build_activation_module(name: str, features: int) -> nn.Module:
    activation_name = name.lower()
    if activation_name == "linear":
        return nn.Identity()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "tanh":
        return nn.Tanh()
    if activation_name == "sigmoid":
        return nn.Sigmoid()
    if activation_name == "snake":
        return Snake(features)
    raise ValueError(f"Unsupported activation function: {name}")


class NARXTDNN(nn.Module):
    def __init__(
        self,
        delay: int,
        input_state_dim: int,
        narx_hidden_layers: int,
        dense_layers: int,
        hidden_width: int,
        activation: str,
        dropout: float,
        feedback_dim: int = 3,
    ) -> None:
        super().__init__()
        input_dim = delay * input_state_dim + delay * feedback_dim
        hidden_structure = [hidden_width] * narx_hidden_layers + [hidden_width] * dense_layers
        layers: list[nn.Module] = []
        last_dim = input_dim

        for hidden_dim in hidden_structure:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(build_activation_module(activation, hidden_dim))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, 3))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Yogi(Optimizer):
    """Minimal Yogi optimizer implementation for this script."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-3,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if weight_decay != 0.0:
                    grad = grad.add(param, alpha=weight_decay)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                grad_sq = grad * grad
                exp_avg_sq.addcmul_(
                    torch.sign(exp_avg_sq - grad_sq),
                    grad_sq,
                    value=-(1.0 - beta2),
                )

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                denom = exp_avg_sq_hat.sqrt().add_(eps)
                param.addcdiv_(exp_avg_hat, denom, value=-lr)

        return loss


def build_model_from_trial(hparams: dict[str, Any], input_state_dim: int, device: torch.device) -> nn.Module:
    model = NARXTDNN(
        delay=int(hparams["delay"]),
        input_state_dim=input_state_dim,
        narx_hidden_layers=int(hparams["narx_hidden_layers"]),
        dense_layers=int(hparams["dense_layers"]),
        hidden_width=int(hparams["hidden_width"]),
        activation=str(hparams["activation"]),
        dropout=float(hparams["dropout"]),
    )
    return model.to(device)


def build_optimizer_from_trial(model: nn.Module, hparams: dict[str, Any]) -> Optimizer:
    optimizer_name = str(hparams["optimizer"]).lower()
    lr = float(hparams["lr"])
    weight_decay = float(hparams["weight_decay"])

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if optimizer_name == "yogi":
        return Yogi(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {hparams['optimizer']}")


def build_scheduler_from_trial(
    optimizer: Optimizer,
    hparams: dict[str, Any],
    epochs: int,
    lr_patience: int,
    lr_decay_factor: float,
    min_lr: float,
):
    scheduler_name = str(hparams["scheduler"]).lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_decay_factor,
            patience=lr_patience,
            min_lr=min_lr,
        )
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)
    raise ValueError(f"Unsupported scheduler: {hparams['scheduler']}")


def build_loss_function(hparams: dict[str, Any]) -> nn.Module:
    loss_type = str(hparams["loss_type"]).lower()
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "huber":
        return nn.HuberLoss(delta=float(hparams.get("huber_delta", 1.0)))
    raise ValueError(f"Unsupported loss_type: {hparams['loss_type']}")


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * batch_x.shape[0]
        sample_count += int(batch_x.shape[0])

    return running_loss / max(sample_count, 1)


def compute_metrics_core(
    y_true_rtn: np.ndarray,
    y_pred_rtn: np.ndarray,
    y_true_norm: np.ndarray,
    y_pred_norm: np.ndarray,
    frames_eci_to_rtn: np.ndarray,
    r_sgp4_eci: np.ndarray,
    r_hpop_eci: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    residual_error_rtn = y_pred_rtn - y_true_rtn
    mse_loss = float(np.mean((y_pred_norm - y_true_norm) ** 2))
    rmse_r = float(np.sqrt(np.mean(residual_error_rtn[:, 0] ** 2)))
    rmse_t = float(np.sqrt(np.mean(residual_error_rtn[:, 1] ** 2)))
    rmse_n = float(np.sqrt(np.mean(residual_error_rtn[:, 2] ** 2)))
    total_rtn_rmse = float(np.sqrt(np.mean(np.sum(residual_error_rtn**2, axis=1))))

    delta_r_pred_eci = np.einsum("nij,nj->ni", np.transpose(frames_eci_to_rtn, (0, 2, 1)), y_pred_rtn)
    r_corrected_eci = r_sgp4_eci + delta_r_pred_eci
    corrected_error_eci = r_corrected_eci - r_hpop_eci
    corrected_position_rmse_eci_3d = float(np.sqrt(np.mean(np.sum(corrected_error_eci**2, axis=1))))

    metrics = {
        "mse_loss_normalized": mse_loss,
        "rmse_r_m": rmse_r,
        "rmse_t_m": rmse_t,
        "rmse_n_m": rmse_n,
        "total_rtn_rmse_m": total_rtn_rmse,
        "corrected_position_rmse_eci_3d_m": corrected_position_rmse_eci_3d,
    }
    artifacts = {
        "delta_r_pred_eci": delta_r_pred_eci,
        "r_corrected_eci": r_corrected_eci,
    }
    return metrics, artifacts


def evaluate_open_loop(
    model: nn.Module,
    u_norm: np.ndarray,
    y_norm: np.ndarray,
    y_raw: np.ndarray,
    time_seconds: np.ndarray,
    target_indices: np.ndarray,
    sequence_start_idx: int,
    delay: int,
    batch_size: int,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    frames_eci_to_rtn: np.ndarray,
    r_sgp4_eci: np.ndarray,
    r_hpop_eci: np.ndarray,
    feedback_mode: str,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    x_eval, y_eval_norm, valid_indices = build_narx_open_loop_samples(
        u_norm=u_norm,
        y_norm=y_norm,
        sequence_start_idx=sequence_start_idx,
        target_indices=target_indices,
        delay=delay,
        feedback_mode=feedback_mode,
    )
    eval_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_eval), torch.from_numpy(y_eval_norm.astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()
    pred_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_x, _ in eval_loader:
            pred_batches.append(model(batch_x.to(device)).cpu().numpy())

    y_pred_norm = np.concatenate(pred_batches, axis=0)
    y_true_norm = y_eval_norm
    y_pred_rtn = y_pred_norm * y_std + y_mean
    y_true_rtn = y_raw[valid_indices]
    metrics, artifacts = compute_metrics_core(
        y_true_rtn=y_true_rtn,
        y_pred_rtn=y_pred_rtn,
        y_true_norm=y_true_norm,
        y_pred_norm=y_pred_norm,
        frames_eci_to_rtn=frames_eci_to_rtn[valid_indices],
        r_sgp4_eci=r_sgp4_eci[valid_indices],
        r_hpop_eci=r_hpop_eci[valid_indices],
    )
    payload = {
        "indices": valid_indices,
        "time_seconds": time_seconds[valid_indices],
        "y_true_rtn": y_true_rtn,
        "y_pred_rtn": y_pred_rtn,
        "delta_r_pred_eci": artifacts["delta_r_pred_eci"],
        "r_corrected_eci": artifacts["r_corrected_eci"],
        "r_hpop_eci": r_hpop_eci[valid_indices],
        "r_sgp4_eci": r_sgp4_eci[valid_indices],
    }
    return metrics, payload


def rollout_closed_loop(
    model: nn.Module,
    u_norm: np.ndarray,
    y_norm: np.ndarray,
    start_index: int,
    end_index: int,
    delay: int,
    device: torch.device,
    feedback_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if start_index < delay:
        raise ValueError(f"Closed-loop rollout start index {start_index} is smaller than delay {delay}.")

    history_y = [row.copy() for row in y_norm[start_index - delay : start_index]]
    pred_indices = np.arange(start_index, end_index + 1, dtype=np.int64)
    pred_norm: list[np.ndarray] = []

    model.eval()
    with torch.inference_mode():
        for current_idx in pred_indices:
            u_hist = u_norm[current_idx - delay + 1 : current_idx + 1][::-1].reshape(-1)
            if feedback_mode == "residual_feedback":
                y_hist = np.asarray(history_y[-delay:], dtype=np.float32)[::-1].reshape(-1)
            elif feedback_mode == "zero_feedback_baseline":
                y_hist = np.zeros((delay, y_norm.shape[1]), dtype=np.float32)[::-1].reshape(-1)
            else:
                raise ValueError(f"Unsupported feedback_mode: {feedback_mode}")

            x_input = np.concatenate((u_hist, y_hist), axis=0).astype(np.float32)
            pred = model(torch.from_numpy(x_input).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            pred_norm.append(pred)
            history_y.append(pred)

    return pred_indices, np.asarray(pred_norm, dtype=np.float32)


def evaluate_closed_loop(
    model: nn.Module,
    u_norm: np.ndarray,
    y_norm: np.ndarray,
    y_raw: np.ndarray,
    gap_mask: np.ndarray,
    second_pass_mask: np.ndarray,
    delay: int,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    time_seconds: np.ndarray,
    frames_eci_to_rtn: np.ndarray,
    r_sgp4_eci: np.ndarray,
    r_hpop_eci: np.ndarray,
    feedback_mode: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]]]:
    gap_indices = np.where(gap_mask)[0]
    second_indices = np.where(second_pass_mask)[0]
    rollout_indices, rollout_pred_norm = rollout_closed_loop(
        model=model,
        u_norm=u_norm,
        y_norm=y_norm,
        start_index=int(gap_indices[0]),
        end_index=int(second_indices[-1]),
        delay=delay,
        device=device,
        feedback_mode=feedback_mode,
    )

    rollout_pred_rtn = rollout_pred_norm * y_std + y_mean
    rollout_true_norm = y_norm[rollout_indices]
    rollout_true_rtn = y_raw[rollout_indices]

    metrics: dict[str, dict[str, float]] = {}
    payloads: dict[str, dict[str, np.ndarray]] = {}

    split_map = {
        "gap": gap_mask,
        "second_pass": second_pass_mask,
        "full_rollout": gap_mask | second_pass_mask,
    }
    for split_name, split_mask in split_map.items():
        local_selector = split_mask[rollout_indices]
        split_indices = rollout_indices[local_selector]
        split_pred_norm = rollout_pred_norm[local_selector]
        split_true_norm = rollout_true_norm[local_selector]
        split_pred_rtn = rollout_pred_rtn[local_selector]
        split_true_rtn = rollout_true_rtn[local_selector]
        split_metrics, split_artifacts = compute_metrics_core(
            y_true_rtn=split_true_rtn,
            y_pred_rtn=split_pred_rtn,
            y_true_norm=split_true_norm,
            y_pred_norm=split_pred_norm,
            frames_eci_to_rtn=frames_eci_to_rtn[split_indices],
            r_sgp4_eci=r_sgp4_eci[split_indices],
            r_hpop_eci=r_hpop_eci[split_indices],
        )
        metrics[split_name] = split_metrics
        payloads[split_name] = {
            "indices": split_indices,
            "time_seconds": time_seconds[split_indices],
            "y_true_rtn": split_true_rtn,
            "y_pred_rtn": split_pred_rtn,
            "delta_r_pred_eci": split_artifacts["delta_r_pred_eci"],
            "r_corrected_eci": split_artifacts["r_corrected_eci"],
            "r_hpop_eci": r_hpop_eci[split_indices],
            "r_sgp4_eci": r_sgp4_eci[split_indices],
        }

    return metrics, payloads


def assemble_full_segment_predictions(
    total_length: int,
    train_payload: dict[str, np.ndarray],
    val_payload: dict[str, np.ndarray] | None,
    closed_loop_payloads: dict[str, dict[str, np.ndarray]],
    time_seconds: np.ndarray,
    y_true_rtn: np.ndarray,
    r_sgp4_eci: np.ndarray,
    r_hpop_eci: np.ndarray,
) -> dict[str, np.ndarray]:
    y_pred_rtn_full = np.full((total_length, 3), np.nan, dtype=np.float64)
    delta_r_pred_eci_full = np.full((total_length, 3), np.nan, dtype=np.float64)
    r_corrected_eci_full = np.full((total_length, 3), np.nan, dtype=np.float64)

    def write_payload(payload: dict[str, np.ndarray]) -> None:
        indices = payload["indices"]
        y_pred_rtn_full[indices] = payload["y_pred_rtn"]
        delta_r_pred_eci_full[indices] = payload["delta_r_pred_eci"]
        r_corrected_eci_full[indices] = payload["r_corrected_eci"]

    write_payload(train_payload)
    if val_payload is not None:
        write_payload(val_payload)
    write_payload(closed_loop_payloads["gap"])
    write_payload(closed_loop_payloads["second_pass"])

    valid_pred_mask = np.all(np.isfinite(y_pred_rtn_full), axis=1)
    return {
        "time_seconds": time_seconds,
        "y_true_rtn": y_true_rtn,
        "y_pred_rtn": y_pred_rtn_full,
        "delta_r_pred_eci": delta_r_pred_eci_full,
        "r_corrected_eci": r_corrected_eci_full,
        "r_hpop_eci": r_hpop_eci,
        "r_sgp4_eci": r_sgp4_eci,
        "valid_pred_mask": valid_pred_mask,
    }


def save_prediction_file(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        path,
        time_seconds=payload["time_seconds"],
        y_true_rtn=payload["y_true_rtn"],
        y_pred_rtn=payload["y_pred_rtn"],
        delta_r_pred_eci=payload["delta_r_pred_eci"],
        r_corrected_eci=payload["r_corrected_eci"],
        r_hpop_eci=payload["r_hpop_eci"],
        r_sgp4_eci=payload["r_sgp4_eci"],
    )


def save_full_segment_rtn_position_plot(
    figure_path: Path,
    time_seconds: np.ndarray,
    corrected_rtn_pos_m: np.ndarray,
    hpop_rtn_pos_m: np.ndarray,
    valid_pred_mask: np.ndarray,
    first_pass_indices: np.ndarray,
    gap_indices: np.ndarray,
    second_pass_indices: np.ndarray,
    delay_anchor_time_sec: float,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    component_names = ["R", "T", "N"]
    phase_specs = [
        ("First Pass", first_pass_indices, "#d9edf7"),
        ("Gap", gap_indices, "#fcf8e3"),
        ("Second Pass", second_pass_indices, "#dff0d8"),
    ]

    for axis_idx, axis in enumerate(axes):
        for label, indices, color in phase_specs:
            axis.axvspan(
                time_seconds[indices[0]],
                time_seconds[indices[-1]],
                color=color,
                alpha=0.45,
                label=label if axis_idx == 0 else None,
            )
        axis.plot(
            time_seconds,
            hpop_rtn_pos_m[:, axis_idx],
            color="tab:blue",
            linewidth=1.6,
            label="HPOP RTN Position" if axis_idx == 0 else None,
        )
        axis.plot(
            time_seconds[valid_pred_mask],
            corrected_rtn_pos_m[valid_pred_mask, axis_idx],
            color="tab:red",
            linewidth=1.2,
            label="Corrected SGP4 RTN Position" if axis_idx == 0 else None,
        )
        axis.axvline(
            delay_anchor_time_sec,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="NARX Prediction Start" if axis_idx == 0 else None,
        )
        axis.set_ylabel(f"{component_names[axis_idx]} (m)")
        axis.grid(True, alpha=0.25)

    axes[0].set_title("Full-Segment RTN Position Comparison: Corrected SGP4 vs HPOP")
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Time Since Segment Start (s)")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def save_full_segment_rtn_residual_plot(
    figure_path: Path,
    time_seconds: np.ndarray,
    y_true_rtn: np.ndarray,
    y_pred_rtn: np.ndarray,
    valid_pred_mask: np.ndarray,
    first_pass_indices: np.ndarray,
    gap_indices: np.ndarray,
    second_pass_indices: np.ndarray,
    delay_anchor_time_sec: float,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    component_names = ["R", "T", "N"]
    phase_specs = [
        ("First Pass", first_pass_indices, "#d9edf7"),
        ("Gap", gap_indices, "#fcf8e3"),
        ("Second Pass", second_pass_indices, "#dff0d8"),
    ]

    for axis_idx, axis in enumerate(axes):
        for label, indices, color in phase_specs:
            axis.axvspan(
                time_seconds[indices[0]],
                time_seconds[indices[-1]],
                color=color,
                alpha=0.45,
                label=label if axis_idx == 0 else None,
            )
        axis.plot(
            time_seconds,
            y_true_rtn[:, axis_idx],
            color="tab:blue",
            linewidth=1.6,
            label="True RTN Residual" if axis_idx == 0 else None,
        )
        axis.plot(
            time_seconds[valid_pred_mask],
            y_pred_rtn[valid_pred_mask, axis_idx],
            color="tab:red",
            linewidth=1.2,
            label="Predicted RTN Residual" if axis_idx == 0 else None,
        )
        axis.axvline(
            delay_anchor_time_sec,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="NARX Prediction Start" if axis_idx == 0 else None,
        )
        axis.set_ylabel(f"{component_names[axis_idx]} (m)")
        axis.grid(True, alpha=0.25)

    axes[0].set_title("Full-Segment RTN Residual Comparison: Predicted vs True")
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Time Since Segment Start (s)")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def save_training_loss_curve(figure_path: Path, history: list[dict[str, float]]) -> None:
    epochs = np.asarray([row["epoch"] for row in history], dtype=np.float64)
    train_loss = np.asarray([row["train_loss"] for row in history], dtype=np.float64)
    monitor_loss = np.asarray([row["monitor_loss"] for row in history], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(epochs, train_loss, label="Train Loss", color="tab:blue", linewidth=1.5)
    axis.plot(epochs, monitor_loss, label="Monitor Loss", color="tab:red", linewidth=1.5)
    axis.set_title("Training Loss Curve")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def save_optuna_optimization_history_plot(study: optuna.Study, figure_path: Path) -> None:
    complete_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        return

    trial_numbers = [trial.number for trial in complete_trials]
    objective_values = [trial.value for trial in complete_trials]
    best_so_far = np.minimum.accumulate(np.asarray(objective_values, dtype=np.float64))

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(trial_numbers, objective_values, marker="o", linestyle="-", color="tab:blue", label="Trial Objective")
    axis.plot(trial_numbers, best_so_far, linestyle="--", color="tab:red", label="Best So Far")
    axis.set_title("Optuna Optimization History")
    axis.set_xlabel("Trial")
    axis.set_ylabel("Objective")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def save_optuna_parameter_importance_plot(study: optuna.Study, figure_path: Path) -> None:
    importances = get_param_importances(study)
    if not importances:
        return

    names = list(importances.keys())
    values = list(importances.values())
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.barh(names[::-1], values[::-1], color="tab:green")
    axis.set_title("Optuna Parameter Importance")
    axis.set_xlabel("Importance")
    axis.grid(True, axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available in this environment.")
    return torch.device(device_name)


def convert_delay_seconds_to_steps(delay_seconds: float, dt: float) -> int:
    return max(1, int(round(delay_seconds / dt)))


def resolve_delay_from_params(params: dict[str, Any], dt: float) -> int:
    if params.get("delay") is not None:
        return int(params["delay"])
    if params.get("delay_seconds") is not None:
        return convert_delay_seconds_to_steps(float(params["delay_seconds"]), dt)
    raise ValueError("Either delay or delay_seconds must be provided.")


def merge_hparams(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update({key: value for key, value in overrides.items() if value is not None})
    return merged


def extract_search_space(config: dict[str, Any], dt: float) -> dict[str, Any]:
    search_space = dict(DEFAULT_SEARCH_SPACE)
    search_overrides = dict(config.get("search_space", {}))
    search_space.update(search_overrides)

    if "delay_seconds_choices" in search_space:
        search_space["delay_choices"] = [
            convert_delay_seconds_to_steps(float(value), dt) for value in search_space["delay_seconds_choices"]
        ]
    return search_space


def build_base_hparams(args: argparse.Namespace, config: dict[str, Any], dt: float) -> dict[str, Any]:
    cli_hparams = {
        "narx_hidden_layers": args.narx_hidden_layers,
        "dense_layers": args.dense_layers,
        "hidden_width": args.hidden_width,
        "delay": args.delay,
        "delay_seconds": args.delay_seconds,
        "activation": args.activation,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "loss_type": args.loss_type,
        "scheduler": args.scheduler,
        "input_state_mode": args.input_state_mode,
        "feedback_mode": args.feedback_mode,
        "lr": args.lr,
        "huber_delta": args.huber_delta,
    }
    fixed_hparams = dict(config.get("fixed_hyperparameters", config.get("fixed", {})))
    merged = merge_hparams(cli_hparams, fixed_hparams)
    if args.delay_seconds is not None or ("delay_seconds" in fixed_hparams and "delay" not in fixed_hparams):
        merged["delay"] = None
    merged["delay"] = resolve_delay_from_params(merged, dt)
    merged.pop("delay_seconds", None)
    return merged


def sample_hyperparameters_from_trial(
    trial: optuna.Trial,
    args: argparse.Namespace,
    config: dict[str, Any],
    dt: float,
) -> dict[str, Any]:
    search_space = extract_search_space(config, dt)
    fixed_hparams = dict(config.get("fixed_hyperparameters", config.get("fixed", {})))
    sampled: dict[str, Any] = {"huber_delta": args.huber_delta}

    if "narx_hidden_layers" in fixed_hparams:
        sampled["narx_hidden_layers"] = fixed_hparams["narx_hidden_layers"]
    else:
        sampled["narx_hidden_layers"] = trial.suggest_categorical(
            "narx_hidden_layers", search_space["narx_hidden_layers"]
        )

    if "dense_layers" in fixed_hparams:
        sampled["dense_layers"] = fixed_hparams["dense_layers"]
    else:
        sampled["dense_layers"] = trial.suggest_categorical("dense_layers", search_space["dense_layers"])

    if "hidden_width" in fixed_hparams:
        sampled["hidden_width"] = fixed_hparams["hidden_width"]
    else:
        sampled["hidden_width"] = trial.suggest_categorical("hidden_width", search_space["hidden_width"])

    if "delay" in fixed_hparams or "delay_seconds" in fixed_hparams:
        sampled["delay"] = resolve_delay_from_params(fixed_hparams, dt)
    else:
        sampled["delay"] = trial.suggest_categorical("delay", search_space["delay_choices"])

    for key in ("activation", "optimizer", "input_state_mode", "feedback_mode", "loss_type", "scheduler"):
        if key in fixed_hparams:
            sampled[key] = fixed_hparams[key]
        else:
            sampled[key] = trial.suggest_categorical(key, search_space[key])

    for key in ("batch_size", "weight_decay", "dropout"):
        if key in fixed_hparams:
            sampled[key] = fixed_hparams[key]
        else:
            sampled[key] = trial.suggest_categorical(key, search_space[key])

    if "lr" in fixed_hparams:
        sampled["lr"] = fixed_hparams["lr"]
    else:
        sampled["lr"] = trial.suggest_float("lr", search_space["lr_min"], search_space["lr_max"], log=True)

    sampled.pop("delay_seconds", None)
    return sampled


def flatten_summary_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        "val_open_loop_loss": float(metrics["first_pass_validation"]["mse_loss_normalized"]),
        "val_open_loop_rtn_rmse": float(metrics["first_pass_validation"]["total_rtn_rmse_m"]),
        "gap_closed_loop_rtn_rmse": float(metrics["gap"]["total_rtn_rmse_m"]),
        "second_pass_closed_loop_rtn_rmse": float(metrics["second_pass"]["total_rtn_rmse_m"]),
        "full_rollout_rtn_rmse": float(metrics["full_rollout"]["total_rtn_rmse_m"]),
        "val_rmse_r_m": float(metrics["first_pass_validation"]["rmse_r_m"]),
        "val_rmse_t_m": float(metrics["first_pass_validation"]["rmse_t_m"]),
        "val_rmse_n_m": float(metrics["first_pass_validation"]["rmse_n_m"]),
        "gap_rmse_r_m": float(metrics["gap"]["rmse_r_m"]),
        "gap_rmse_t_m": float(metrics["gap"]["rmse_t_m"]),
        "gap_rmse_n_m": float(metrics["gap"]["rmse_n_m"]),
        "second_rmse_r_m": float(metrics["second_pass"]["rmse_r_m"]),
        "second_rmse_t_m": float(metrics["second_pass"]["rmse_t_m"]),
        "second_rmse_n_m": float(metrics["second_pass"]["rmse_n_m"]),
    }


def compute_objective_value(metrics: dict[str, dict[str, float]], objective_mode: str) -> float:
    val_rmse = metrics["first_pass_validation"]["total_rtn_rmse_m"]
    gap_rmse = metrics["gap"]["total_rtn_rmse_m"]
    second_rmse = metrics["second_pass"]["total_rtn_rmse_m"]

    if objective_mode == "open_loop_only":
        return float(val_rmse)
    if objective_mode == "rollout_only":
        return float(0.4 * gap_rmse + 0.6 * second_rmse)
    if objective_mode == "hybrid":
        return float(0.2 * val_rmse + 0.3 * gap_rmse + 0.5 * second_rmse)
    raise ValueError(f"Unsupported objective_mode: {objective_mode}")


def run_one_trial(
    bundle: dict[str, Any],
    hparams: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    objective_mode: str,
    trial: optuna.Trial | None = None,
    save_full_artifacts: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    delay = int(hparams["delay"])
    first_pass_indices = bundle["first_pass_indices"]

    if first_pass_indices.size <= delay + 1:
        raise ValueError(f"First pass has {first_pass_indices.size} points, insufficient for delay={delay}.")

    requested_val_points = int(round(first_pass_indices.size * args.val_fraction))
    max_val_points = first_pass_indices.size - (delay + 2)
    use_validation = requested_val_points > 0 and max_val_points >= 1
    if use_validation:
        val_point_count = min(max(requested_val_points, 1), max_val_points)
        train_first_pass_indices = first_pass_indices[:-val_point_count]
        val_first_pass_indices = first_pass_indices[-val_point_count:]
    else:
        train_first_pass_indices = first_pass_indices
        val_first_pass_indices = np.asarray([], dtype=np.int64)

    if train_first_pass_indices.size <= delay + 1:
        raise ValueError(
            "Training portion of the first pass is too short after validation split. "
            "Reduce delay or val_fraction."
        )

    u_raw = build_exogenous_input_series(bundle, hparams["input_state_mode"]).astype(np.float64)
    y_raw = bundle["residual_rtn_pos_m"].astype(np.float64)
    u_mean, u_std = compute_norm_stats(u_raw[train_first_pass_indices])
    y_mean, y_std = compute_norm_stats(y_raw[train_first_pass_indices])
    u_norm = ((u_raw - u_mean) / u_std).astype(np.float32)
    y_norm = ((y_raw - y_mean) / y_std).astype(np.float32)

    sequence_start_idx = int(first_pass_indices[0])
    x_train, y_train, train_sample_indices = build_narx_open_loop_samples(
        u_norm=u_norm,
        y_norm=y_norm,
        sequence_start_idx=sequence_start_idx,
        target_indices=train_first_pass_indices,
        delay=delay,
        feedback_mode=hparams["feedback_mode"],
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=int(hparams["batch_size"]),
        shuffle=False,
    )

    model = build_model_from_trial(hparams=hparams, input_state_dim=u_raw.shape[1], device=device)
    optimizer = build_optimizer_from_trial(model, hparams)
    scheduler = build_scheduler_from_trial(
        optimizer=optimizer,
        hparams=hparams,
        epochs=args.epochs,
        lr_patience=args.lr_patience,
        lr_decay_factor=args.lr_decay_factor,
        min_lr=args.min_lr,
    )
    criterion = build_loss_function(hparams)

    history: list[dict[str, float]] = []
    best_score = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    checkpoint_path = output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        if val_first_pass_indices.size > 0:
            val_metrics, _ = evaluate_open_loop(
                model=model,
                u_norm=u_norm,
                y_norm=y_norm,
                y_raw=y_raw,
                time_seconds=bundle["time_seconds"],
                target_indices=val_first_pass_indices,
                sequence_start_idx=sequence_start_idx,
                delay=delay,
                batch_size=int(hparams["batch_size"]),
                device=device,
                y_mean=y_mean,
                y_std=y_std,
                frames_eci_to_rtn=bundle["frames_eci_to_rtn"],
                r_sgp4_eci=bundle["sgp4_eci_pos_m"],
                r_hpop_eci=bundle["hpop_eci_pos_m"],
                feedback_mode=hparams["feedback_mode"],
            )
            monitor_score = val_metrics["total_rtn_rmse_m"]
        else:
            monitor_score = train_loss

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor_score)
            else:
                scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "monitor_loss": float(monitor_score),
                "lr": current_lr,
            }
        )

        if trial is not None:
            trial.report(float(monitor_score), step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if monitor_score < best_score:
            best_score = float(monitor_score)
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hyperparameters": hparams,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "npz_path": bundle["npz_path"],
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_metrics, train_payload = evaluate_open_loop(
        model=model,
        u_norm=u_norm,
        y_norm=y_norm,
        y_raw=y_raw,
        time_seconds=bundle["time_seconds"],
        target_indices=train_first_pass_indices,
        sequence_start_idx=sequence_start_idx,
        delay=delay,
        batch_size=int(hparams["batch_size"]),
        device=device,
        y_mean=y_mean,
        y_std=y_std,
        frames_eci_to_rtn=bundle["frames_eci_to_rtn"],
        r_sgp4_eci=bundle["sgp4_eci_pos_m"],
        r_hpop_eci=bundle["hpop_eci_pos_m"],
        feedback_mode=hparams["feedback_mode"],
    )

    val_metrics: dict[str, float] | None = None
    val_payload: dict[str, np.ndarray] | None = None
    if val_first_pass_indices.size > 0:
        val_metrics, val_payload = evaluate_open_loop(
            model=model,
            u_norm=u_norm,
            y_norm=y_norm,
            y_raw=y_raw,
            time_seconds=bundle["time_seconds"],
            target_indices=val_first_pass_indices,
            sequence_start_idx=sequence_start_idx,
            delay=delay,
            batch_size=int(hparams["batch_size"]),
            device=device,
            y_mean=y_mean,
            y_std=y_std,
            frames_eci_to_rtn=bundle["frames_eci_to_rtn"],
            r_sgp4_eci=bundle["sgp4_eci_pos_m"],
            r_hpop_eci=bundle["hpop_eci_pos_m"],
            feedback_mode=hparams["feedback_mode"],
        )

    closed_loop_metrics, closed_loop_payloads = evaluate_closed_loop(
        model=model,
        u_norm=u_norm,
        y_norm=y_norm,
        y_raw=y_raw,
        gap_mask=bundle["predict_mask_gap"],
        second_pass_mask=bundle["eval_mask_second_pass"],
        delay=delay,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
        time_seconds=bundle["time_seconds"],
        frames_eci_to_rtn=bundle["frames_eci_to_rtn"],
        r_sgp4_eci=bundle["sgp4_eci_pos_m"],
        r_hpop_eci=bundle["hpop_eci_pos_m"],
        feedback_mode=hparams["feedback_mode"],
    )

    metrics = {
        "first_pass_train": train_metrics,
        "first_pass_validation": val_metrics if val_metrics is not None else train_metrics,
        "gap": closed_loop_metrics["gap"],
        "second_pass": closed_loop_metrics["second_pass"],
        "full_rollout": closed_loop_metrics["full_rollout"],
    }
    objective_value = compute_objective_value(metrics, objective_mode)

    result = {
        "hyperparameters": hparams,
        "split_sizes": {
            "first_pass_points": int(bundle["first_pass_indices"].size),
            "train_first_pass_points": int(train_first_pass_indices.size),
            "val_first_pass_points": int(val_first_pass_indices.size),
            "gap_points": int(bundle["gap_indices"].size),
            "second_pass_points": int(bundle["second_pass_indices"].size),
            "train_samples": int(train_sample_indices.size),
        },
        "training": {
            "best_epoch": int(best_epoch),
            "best_monitor_score": float(best_score),
            "history": history,
        },
        "metrics": metrics,
        "summary_metrics": flatten_summary_metrics(metrics),
        "objective_mode": objective_mode,
        "objective_value": float(objective_value),
        "artifacts": {
            "checkpoint": str(checkpoint_path.resolve()),
        },
    }

    (output_dir / "trial_config.json").write_text(json.dumps(hparams, indent=2), encoding="utf-8")

    np.savez_compressed(output_dir / "norm_stats.npz", u_mean=u_mean, u_std=u_std, y_mean=y_mean, y_std=y_std)
    if save_full_artifacts:
        save_prediction_file(output_dir / "predictions_gap.npz", closed_loop_payloads["gap"])
        save_prediction_file(output_dir / "predictions_second_pass.npz", closed_loop_payloads["second_pass"])
        full_segment_payload = assemble_full_segment_predictions(
            total_length=bundle["time_seconds"].shape[0],
            train_payload=train_payload,
            val_payload=val_payload,
            closed_loop_payloads=closed_loop_payloads,
            time_seconds=bundle["time_seconds"],
            y_true_rtn=y_raw,
            r_sgp4_eci=bundle["sgp4_eci_pos_m"],
            r_hpop_eci=bundle["hpop_eci_pos_m"],
        )
        np.savez_compressed(output_dir / "predictions_full_segment.npz", **full_segment_payload)

        corrected_rtn_pos_m = np.full_like(bundle["hpop_rtn_pos_m"], np.nan)
        corrected_valid_mask = full_segment_payload["valid_pred_mask"]
        corrected_rtn_pos_m[corrected_valid_mask] = compute_rtn_position_series(
            bundle["frames_eci_to_rtn"][corrected_valid_mask],
            full_segment_payload["r_corrected_eci"][corrected_valid_mask],
        )
        delay_anchor_time_sec = bundle["time_seconds"][int(bundle["first_pass_indices"][0]) + delay]
        position_plot_path = output_dir / "full_segment_rtn_position_comparison.png"
        residual_plot_path = output_dir / "full_segment_rtn_residual_comparison.png"
        loss_curve_path = output_dir / "training_loss_curve.png"

        save_full_segment_rtn_position_plot(
            figure_path=position_plot_path,
            time_seconds=bundle["time_seconds"],
            corrected_rtn_pos_m=corrected_rtn_pos_m,
            hpop_rtn_pos_m=bundle["hpop_rtn_pos_m"],
            valid_pred_mask=corrected_valid_mask,
            first_pass_indices=bundle["first_pass_indices"],
            gap_indices=bundle["gap_indices"],
            second_pass_indices=bundle["second_pass_indices"],
            delay_anchor_time_sec=delay_anchor_time_sec,
        )
        save_full_segment_rtn_residual_plot(
            figure_path=residual_plot_path,
            time_seconds=bundle["time_seconds"],
            y_true_rtn=full_segment_payload["y_true_rtn"],
            y_pred_rtn=full_segment_payload["y_pred_rtn"],
            valid_pred_mask=corrected_valid_mask,
            first_pass_indices=bundle["first_pass_indices"],
            gap_indices=bundle["gap_indices"],
            second_pass_indices=bundle["second_pass_indices"],
            delay_anchor_time_sec=delay_anchor_time_sec,
        )
        save_training_loss_curve(loss_curve_path, history)
        result["artifacts"].update(
            {
                "predictions_gap": str((output_dir / "predictions_gap.npz").resolve()),
                "predictions_second_pass": str((output_dir / "predictions_second_pass.npz").resolve()),
                "predictions_full_segment": str((output_dir / "predictions_full_segment.npz").resolve()),
                "full_segment_rtn_position_comparison": str(position_plot_path.resolve()),
                "full_segment_rtn_residual_comparison": str(residual_plot_path.resolve()),
                "training_loss_curve": str(loss_curve_path.resolve()),
            }
        )

    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def objective(
    trial: optuna.Trial,
    bundle: dict[str, Any],
    args: argparse.Namespace,
    config: dict[str, Any],
    device: torch.device,
    study_dir: Path,
) -> float:
    hparams = sample_hyperparameters_from_trial(trial, args, config, bundle["dt"])
    trial_dir = study_dir / "trials" / f"trial_{trial.number:04d}"
    try:
        result = run_one_trial(
            bundle=bundle,
            hparams=hparams,
            args=args,
            output_dir=trial_dir,
            device=device,
            objective_mode=args.objective_mode,
            trial=trial,
            save_full_artifacts=False,
        )
    except ValueError as exc:
        trial.set_user_attr("error", str(exc))
        raise optuna.TrialPruned() from exc
    trial.set_user_attr("trial_dir", str(trial_dir.resolve()))
    trial.set_user_attr("objective_value", float(result["objective_value"]))
    for key, value in result["summary_metrics"].items():
        trial.set_user_attr(key, float(value))
    return float(result["objective_value"])


def save_trials_csv(study: optuna.Study, csv_path: Path) -> None:
    rows = []
    for trial in study.trials:
        row = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "objective_value": trial.value,
        }
        for key, value in trial.params.items():
            row[f"param_{key}"] = value
        for key, value in trial.user_attrs.items():
            if isinstance(value, (int, float, str)):
                row[f"attr_{key}"] = value
        rows.append(row)

    all_keys = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def save_top_k_trials_csv(study: optuna.Study, csv_path: Path, top_k: int) -> list[dict[str, Any]]:
    complete_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    ranked_trials = sorted(complete_trials, key=lambda item: item.value)[:top_k]
    rows: list[dict[str, Any]] = []
    for rank, trial in enumerate(ranked_trials, start=1):
        row = {
            "rank": rank,
            "trial_number": trial.number,
            "objective_value": trial.value,
            "val_open_loop_rtn_rmse_m": trial.user_attrs.get("val_open_loop_rtn_rmse"),
            "gap_closed_loop_rtn_rmse_m": trial.user_attrs.get("gap_closed_loop_rtn_rmse"),
            "second_pass_closed_loop_rtn_rmse_m": trial.user_attrs.get("second_pass_closed_loop_rtn_rmse"),
            "full_rollout_rtn_rmse_m": trial.user_attrs.get("full_rollout_rtn_rmse"),
        }
        for key, value in trial.params.items():
            row[key] = value
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_top_k_trials(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No completed trials to display.")
        return
    print("\nTop trials:")
    for row in rows:
        print(
            f"rank={row['rank']} trial={row['trial_number']} objective={row['objective_value']:.4f} "
            f"val={row.get('val_open_loop_rtn_rmse_m')} "
            f"gap={row.get('gap_closed_loop_rtn_rmse_m')} "
            f"second={row.get('second_pass_closed_loop_rtn_rmse_m')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-satellite explicit NARX/TDNN RTN residual trainer/searcher.")
    parser.add_argument("--npz", required=True, help="Path to one single-satellite .npz file.")
    parser.add_argument("--save-dir", required=True, help="Output directory.")
    parser.add_argument("--config", default=None, help="Optional JSON config for fixed hyperparameters/search space.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--search", action="store_true", help="Enable Optuna hyperparameter search.")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", default="narx_rtn_search")
    parser.add_argument(
        "--objective-mode",
        default="hybrid",
        choices=["open_loop_only", "rollout_only", "hybrid"],
    )
    parser.add_argument("--top-k", type=int, default=5)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)

    parser.add_argument("--narx-hidden-layers", type=int, default=2)
    parser.add_argument("--dense-layers", type=int, default=2)
    parser.add_argument("--hidden-width", type=int, default=10)
    parser.add_argument("--delay", type=int, default=400)
    parser.add_argument("--delay-seconds", type=float, default=None)
    parser.add_argument("--activation", default="tanh", choices=["linear", "relu", "tanh", "sigmoid", "snake"])
    parser.add_argument("--optimizer", default="Yogi", choices=["Adam", "Adagrad", "SGD", "Yogi"])
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss-type", default="mse", choices=["mse", "huber"])
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--scheduler", default="plateau", choices=["none", "plateau", "cosine"])
    parser.add_argument("--input-state-mode", default="pos_vel", choices=["pos_only", "vel_only", "pos_vel"])
    parser.add_argument(
        "--feedback-mode",
        default="residual_feedback",
        choices=["residual_feedback", "zero_feedback_baseline"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    config = load_optional_json(args.config)
    device = resolve_device(args.device)
    bundle = prepare_single_satellite_bundle(args.npz)
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    if not args.search:
        hparams = build_base_hparams(args, config, bundle["dt"])
        result = run_one_trial(
            bundle=bundle,
            hparams=hparams,
            args=args,
            output_dir=save_dir,
            device=device,
            objective_mode=args.objective_mode,
            trial=None,
            save_full_artifacts=True,
        )
        print("\nBest hyperparameters:")
        print(json.dumps(hparams, indent=2))
        print("\nMetrics:")
        print(json.dumps(result["metrics"], indent=2))
        return 0

    study_dir = save_dir / args.study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        lambda trial: objective(
            trial=trial,
            bundle=bundle,
            args=args,
            config=config,
            device=device,
            study_dir=study_dir,
        ),
        n_trials=args.n_trials,
    )

    complete_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        raise RuntimeError("No completed Optuna trial was produced.")

    best_trial = study.best_trial
    best_hparams = sample_hyperparameters_from_trial(
        trial=optuna.trial.FixedTrial(best_trial.params),
        args=args,
        config=config,
        dt=bundle["dt"],
    )
    best_dir = study_dir / "best_trial"
    best_result = run_one_trial(
        bundle=bundle,
        hparams=best_hparams,
        args=args,
        output_dir=best_dir,
        device=device,
        objective_mode=args.objective_mode,
        trial=None,
        save_full_artifacts=True,
    )

    save_trials_csv(study, study_dir / "trials_summary.csv")
    top_rows = save_top_k_trials_csv(study, study_dir / "top_trials.csv", args.top_k)
    print_top_k_trials(top_rows)
    save_optuna_optimization_history_plot(study, study_dir / "optuna_optimization_history.png")
    save_optuna_parameter_importance_plot(study, study_dir / "optuna_parameter_importance.png")
    shutil.copy2(best_dir / "best_model.pth", study_dir / "best_trial_checkpoint.pth")

    summary = {
        "study_name": args.study_name,
        "objective_mode": args.objective_mode,
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "best_hyperparameters": best_hparams,
        "best_metrics": best_result["metrics"],
        "best_trial_artifacts": best_result["artifacts"],
        "top_trials_csv": str((study_dir / "top_trials.csv").resolve()),
        "trials_summary_csv": str((study_dir / "trials_summary.csv").resolve()),
    }
    (study_dir / "search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBest hyperparameters:")
    print(json.dumps(best_hparams, indent=2))
    print("\nBest trial metrics:")
    print(json.dumps(best_result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
