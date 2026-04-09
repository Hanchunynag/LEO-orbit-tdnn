from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_rtn_frame(
    reference_pos_eci_m: np.ndarray,
    reference_vel_eci_mps: np.ndarray,
) -> np.ndarray:
    r_hat = reference_pos_eci_m / np.linalg.norm(reference_pos_eci_m)
    n_hat = np.cross(reference_pos_eci_m, reference_vel_eci_mps)
    n_hat = n_hat / np.linalg.norm(n_hat)
    t_hat = np.cross(n_hat, r_hat)
    t_hat = t_hat / np.linalg.norm(t_hat)
    return np.vstack((r_hat, t_hat, n_hat))


def build_pass_masks(npz_data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def maybe_bool_field(*names: str) -> np.ndarray | None:
        for name in names:
            if name in npz_data:
                return np.asarray(npz_data[name], dtype=bool)
        return None

    time_seconds = np.asarray(npz_data["time_seconds"], dtype=np.float64)
    first_pass_mask = maybe_bool_field("train_mask_first_pass", "first_pass_mask")
    gap_mask = maybe_bool_field("predict_mask_gap", "prediction_gap_mask")
    second_pass_mask = maybe_bool_field("eval_mask_second_pass", "second_pass_mask")

    if first_pass_mask is not None and gap_mask is not None and second_pass_mask is not None:
        return first_pass_mask, gap_mask, second_pass_mask

    pass_windows_time_sec = np.asarray(npz_data["pass_windows_time_sec"], dtype=np.float64)
    if pass_windows_time_sec.ndim != 2 or pass_windows_time_sec.shape != (2, 2):
        raise ValueError(f"Expected exactly 2 pass windows, got shape {pass_windows_time_sec.shape}.")

    first_start_sec, first_end_sec = pass_windows_time_sec[0]
    second_start_sec, second_end_sec = pass_windows_time_sec[1]
    first_pass_mask = (time_seconds >= first_start_sec) & (time_seconds <= first_end_sec)
    gap_mask = (time_seconds > first_end_sec) & (time_seconds < second_start_sec)
    second_pass_mask = (time_seconds >= second_start_sec) & (time_seconds <= second_end_sec)
    return first_pass_mask, gap_mask, second_pass_mask


def contiguous_indices_from_mask(mask: np.ndarray, name: str) -> np.ndarray:
    indices = np.where(mask)[0]
    if indices.size == 0:
        raise ValueError(f"{name} is empty.")
    if indices.size > 1 and not np.all(np.diff(indices) == 1):
        raise ValueError(f"{name} must be contiguous.")
    return indices


def compute_rtn_residual_and_frames(npz_data: Any) -> tuple[np.ndarray, np.ndarray]:
    sgp4_eci_pos_m = np.asarray(npz_data["sgp4_eci_pos_m"], dtype=np.float64)
    sgp4_eci_vel_mps = np.asarray(npz_data["sgp4_eci_vel_mps"], dtype=np.float64)
    hpop_eci_pos_m = np.asarray(npz_data["hpop_eci_pos_m"], dtype=np.float64)

    if "rtn_frame_eci_to_rtn" in npz_data:
        frames_eci_to_rtn = np.asarray(npz_data["rtn_frame_eci_to_rtn"], dtype=np.float64)
    else:
        frames_eci_to_rtn = np.stack(
            [build_rtn_frame(sgp4_eci_pos_m[idx], sgp4_eci_vel_mps[idx]) for idx in range(sgp4_eci_pos_m.shape[0])],
            axis=0,
        )

    if "residual_rtn_pos_m" in npz_data:
        residual_rtn_pos_m = np.asarray(npz_data["residual_rtn_pos_m"], dtype=np.float64)
    else:
        residual_eci_pos_m = hpop_eci_pos_m - sgp4_eci_pos_m
        residual_rtn_pos_m = np.einsum("nij,nj->ni", frames_eci_to_rtn, residual_eci_pos_m)
    return residual_rtn_pos_m, frames_eci_to_rtn


def prepare_bundle(npz_path: str | Path) -> dict[str, Any]:
    npz_path = Path(npz_path).resolve()
    with np.load(npz_path, allow_pickle=False) as npz_data:
        time_seconds = np.asarray(npz_data["time_seconds"], dtype=np.float64)
        sgp4_eci_pos_m = np.asarray(npz_data["sgp4_eci_pos_m"], dtype=np.float64)
        sgp4_eci_vel_mps = np.asarray(npz_data["sgp4_eci_vel_mps"], dtype=np.float64)
        hpop_eci_pos_m = np.asarray(npz_data["hpop_eci_pos_m"], dtype=np.float64)
        hpop_eci_vel_mps = np.asarray(npz_data["hpop_eci_vel_mps"], dtype=np.float64)
        residual_rtn_pos_m, frames_eci_to_rtn = compute_rtn_residual_and_frames(npz_data)
        first_pass_mask, gap_mask, second_pass_mask = build_pass_masks(npz_data)

    first_pass_indices = contiguous_indices_from_mask(first_pass_mask, "first_pass_mask")
    gap_indices = contiguous_indices_from_mask(gap_mask, "gap_mask")
    second_pass_indices = contiguous_indices_from_mask(second_pass_mask, "second_pass_mask")

    return {
        "npz_path": str(npz_path),
        "time_seconds": time_seconds,
        "sgp4_eci_pos_m": sgp4_eci_pos_m,
        "sgp4_eci_vel_mps": sgp4_eci_vel_mps,
        "hpop_eci_pos_m": hpop_eci_pos_m,
        "hpop_eci_vel_mps": hpop_eci_vel_mps,
        "residual_rtn_pos_m": residual_rtn_pos_m,
        "frames_eci_to_rtn": frames_eci_to_rtn,
        "first_pass_indices": first_pass_indices,
        "gap_indices": gap_indices,
        "second_pass_indices": second_pass_indices,
    }


def split_first_pass(first_pass_indices: np.ndarray, val_fraction: float, min_train_points: int = 8) -> tuple[np.ndarray, np.ndarray]:
    if first_pass_indices.size < min_train_points + 1:
        raise ValueError("First-pass sample count is too small for fitting.")

    requested_val_points = int(round(first_pass_indices.size * val_fraction))
    max_val_points = first_pass_indices.size - min_train_points
    use_validation = requested_val_points > 0 and max_val_points >= 1
    if not use_validation:
        return first_pass_indices, np.asarray([], dtype=np.int64)

    val_points = min(max(requested_val_points, 1), max_val_points)
    return first_pass_indices[:-val_points], first_pass_indices[-val_points:]


def estimate_orbital_phase_rad(reference_pos_eci_m: np.ndarray, reference_vel_eci_mps: np.ndarray) -> np.ndarray:
    h_vec = np.cross(reference_pos_eci_m[0], reference_vel_eci_mps[0])
    h_hat = h_vec / np.linalg.norm(h_vec)
    p_hat = reference_pos_eci_m[0] / np.linalg.norm(reference_pos_eci_m[0])
    q_hat = np.cross(h_hat, p_hat)
    q_hat = q_hat / np.linalg.norm(q_hat)
    phase_rad = np.arctan2(reference_pos_eci_m @ q_hat, reference_pos_eci_m @ p_hat)
    return np.unwrap(phase_rad)


def parse_harmonic_orders(spec: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in spec.split(",") if part.strip()}))
    if not values:
        raise ValueError("At least one harmonic order must be provided.")
    if any(value <= 0 for value in values):
        raise ValueError("Harmonic orders must be positive integers.")
    return values


def build_harmonic_design_matrix(
    time_seconds: np.ndarray,
    phase_rad: np.ndarray,
    phase_origin_rad: float,
    drift_order: int,
    harmonic_orders: tuple[int, ...],
) -> tuple[np.ndarray, list[str]]:
    tau_hours = (time_seconds - time_seconds[0]) / 3600.0
    delta_phase_rad = phase_rad - phase_origin_rad

    columns = [np.ones_like(tau_hours)]
    names = ["bias"]

    for degree in range(1, drift_order + 1):
        columns.append(tau_hours**degree)
        names.append(f"tau_hours^{degree}")

    for harmonic in harmonic_orders:
        columns.append(np.sin(harmonic * delta_phase_rad))
        columns.append(np.cos(harmonic * delta_phase_rad))
        names.extend([f"sin_{harmonic}rev", f"cos_{harmonic}rev"])

    return np.column_stack(columns).astype(np.float64), names


def fit_ridge_regression(design_matrix: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
    gram = design_matrix.T @ design_matrix
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * ridge
    regularizer[0, 0] = 0.0
    rhs = design_matrix.T @ targets
    return np.linalg.solve(gram + regularizer, rhs)


def compute_feature_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def compute_target_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def build_direct_query_features(
    bundle: dict[str, Any],
    phase_rad: np.ndarray,
    harmonic_prediction_rtn_m: np.ndarray,
    harmonic_orders: tuple[int, ...],
) -> tuple[np.ndarray, list[str]]:
    time_seconds = bundle["time_seconds"]
    tau_hours = (time_seconds - time_seconds[0]) / 3600.0
    delta_phase_rad = phase_rad - phase_rad[0]

    scaled_pos = bundle["sgp4_eci_pos_m"] / 1.0e7
    scaled_vel = bundle["sgp4_eci_vel_mps"] / 1.0e4
    scaled_harmonic = harmonic_prediction_rtn_m / 1.0e3

    columns = [tau_hours, tau_hours**2]
    names = ["tau_hours", "tau_hours^2"]

    for harmonic in harmonic_orders:
        columns.append(np.sin(harmonic * delta_phase_rad))
        columns.append(np.cos(harmonic * delta_phase_rad))
        names.extend([f"sin_{harmonic}rev", f"cos_{harmonic}rev"])

    for axis_idx, axis_name in enumerate(("x", "y", "z")):
        columns.append(scaled_pos[:, axis_idx])
        names.append(f"sgp4_pos_{axis_name}")

    for axis_idx, axis_name in enumerate(("vx", "vy", "vz")):
        columns.append(scaled_vel[:, axis_idx])
        names.append(f"sgp4_vel_{axis_name}")

    for axis_idx, axis_name in enumerate(("r", "t", "n")):
        columns.append(scaled_harmonic[:, axis_idx])
        names.append(f"harmonic_pred_{axis_name}")

    return np.column_stack(columns).astype(np.float64), names


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_width: int, hidden_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_width))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_width
        layers.append(nn.Linear(last_dim, 3))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def train_residual_network(
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[ResidualMLP, dict[str, np.ndarray], list[dict[str, float]]]:
    device = torch.device(args.device)
    model = ResidualMLP(
        input_dim=features.shape[1],
        hidden_width=args.nn_hidden_width,
        hidden_layers=args.nn_hidden_layers,
        dropout=args.nn_dropout,
    ).to(device)

    x_mean, x_std = compute_feature_stats(features[train_indices])
    y_mean, y_std = compute_target_stats(targets[train_indices])

    features_norm = ((features - x_mean) / x_std).astype(np.float32)
    targets_norm = ((targets - y_mean) / y_std).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(features_norm[train_indices]),
            torch.from_numpy(targets_norm[train_indices]),
        ),
        batch_size=args.nn_batch_size,
        shuffle=True,
    )

    criterion = nn.HuberLoss(delta=args.nn_huber_delta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.nn_lr, weight_decay=args.nn_weight_decay)

    history: list[dict[str, float]] = []
    best_state = clone_state_dict(model)
    best_score = float("inf")
    epochs_without_improvement = 0

    val_selector = val_indices if val_indices.size > 0 else train_indices
    val_x = torch.from_numpy(features_norm[val_selector]).to(device)
    val_y = torch.from_numpy(targets_norm[val_selector]).to(device)

    for epoch in range(1, args.nn_epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * batch_x.shape[0]
            sample_count += int(batch_x.shape[0])

        train_loss = running_loss / max(sample_count, 1)

        model.eval()
        with torch.inference_mode():
            val_pred = model(val_x)
            val_loss = float(criterion(val_pred, val_y).item())

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )

        if val_loss < best_score:
            best_score = val_loss
            best_state = clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.nn_patience:
            break

    model.load_state_dict(best_state)
    stats = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return model, stats, history


def predict_residual_network(
    model: ResidualMLP,
    stats: dict[str, np.ndarray],
    features: np.ndarray,
    device_name: str,
) -> np.ndarray:
    device = torch.device(device_name)
    features_norm = ((features - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
    model.eval()
    with torch.inference_mode():
        pred_norm = model(torch.from_numpy(features_norm).to(device)).cpu().numpy()
    return pred_norm * stats["y_std"] + stats["y_mean"]


def compute_metrics(
    y_true_rtn: np.ndarray,
    y_pred_rtn: np.ndarray,
    frames_eci_to_rtn: np.ndarray,
    r_sgp4_eci: np.ndarray,
    r_hpop_eci: np.ndarray,
) -> dict[str, float]:
    residual_error_rtn = y_pred_rtn - y_true_rtn
    rmse_r = float(np.sqrt(np.mean(residual_error_rtn[:, 0] ** 2)))
    rmse_t = float(np.sqrt(np.mean(residual_error_rtn[:, 1] ** 2)))
    rmse_n = float(np.sqrt(np.mean(residual_error_rtn[:, 2] ** 2)))
    total_rtn_rmse = float(np.sqrt(np.mean(np.sum(residual_error_rtn**2, axis=1))))

    delta_r_pred_eci = np.einsum("nij,nj->ni", np.transpose(frames_eci_to_rtn, (0, 2, 1)), y_pred_rtn)
    r_corrected_eci = r_sgp4_eci + delta_r_pred_eci
    corrected_error_eci = r_corrected_eci - r_hpop_eci
    corrected_position_rmse_eci_3d = float(np.sqrt(np.mean(np.sum(corrected_error_eci**2, axis=1))))

    return {
        "rmse_r_m": rmse_r,
        "rmse_t_m": rmse_t,
        "rmse_n_m": rmse_n,
        "total_rtn_rmse_m": total_rtn_rmse,
        "corrected_position_rmse_eci_3d_m": corrected_position_rmse_eci_3d,
    }


def evaluate_by_split(bundle: dict[str, Any], prediction_rtn_m: np.ndarray, split_indices: np.ndarray) -> dict[str, float]:
    return compute_metrics(
        y_true_rtn=bundle["residual_rtn_pos_m"][split_indices],
        y_pred_rtn=prediction_rtn_m[split_indices],
        frames_eci_to_rtn=bundle["frames_eci_to_rtn"][split_indices],
        r_sgp4_eci=bundle["sgp4_eci_pos_m"][split_indices],
        r_hpop_eci=bundle["hpop_eci_pos_m"][split_indices],
    )


def save_full_segment_plot(
    figure_path: Path,
    time_seconds: np.ndarray,
    y_true_rtn: np.ndarray,
    y_harmonic_rtn: np.ndarray,
    y_hybrid_rtn: np.ndarray,
    first_pass_indices: np.ndarray,
    gap_indices: np.ndarray,
    second_pass_indices: np.ndarray,
    train_end_time_sec: float,
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
            linewidth=1.8,
            label="True RTN Residual" if axis_idx == 0 else None,
        )
        axis.plot(
            time_seconds,
            y_harmonic_rtn[:, axis_idx],
            color="tab:orange",
            linewidth=1.2,
            linestyle="--",
            label="Harmonic Baseline" if axis_idx == 0 else None,
        )
        axis.plot(
            time_seconds,
            y_hybrid_rtn[:, axis_idx],
            color="tab:red",
            linewidth=1.2,
            label="Harmonic + Residual NN" if axis_idx == 0 else None,
        )
        axis.axvline(
            train_end_time_sec,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="First-Pass Train End" if axis_idx == 0 else None,
        )
        axis.set_ylabel(f"{component_names[axis_idx]} (m)")
        axis.grid(True, alpha=0.25)

    axes[0].set_title("Full-Segment RTN Residual Prediction: Harmonic Baseline vs Hybrid")
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Time Since Segment Start (s)")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def save_training_curve(figure_path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return

    epochs = np.asarray([row["epoch"] for row in history], dtype=np.float64)
    train_loss = np.asarray([row["train_loss"] for row in history], dtype=np.float64)
    val_loss = np.asarray([row["val_loss"] for row in history], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(epochs, train_loss, label="Train Loss", color="tab:blue", linewidth=1.5)
    axis.plot(epochs, val_loss, label="Val Loss", color="tab:red", linewidth=1.5)
    axis.set_title("Residual NN Training Curve")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Huber Loss")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Two-stage RTN residual predictor: harmonic baseline plus direct-query residual MLP."
    )
    parser.add_argument("--npz", type=str, required=True, help="Path to a single-satellite .npz file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for plots, metrics, and predictions.")
    parser.add_argument("--harmonics", type=str, default="1,2", help="Comma-separated harmonic orders, e.g. '1,2'.")
    parser.add_argument("--drift-order", type=int, default=1, help="Polynomial drift order in hours.")
    parser.add_argument("--ridge", type=float, default=1.0e-6, help="Ridge penalty for the harmonic fit.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Tail fraction of first pass for validation.")
    parser.add_argument("--disable-nn", action="store_true", help="Skip the residual neural network stage.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Torch device.")
    parser.add_argument("--nn-hidden-width", type=int, default=64, help="Residual MLP hidden width.")
    parser.add_argument("--nn-hidden-layers", type=int, default=2, help="Residual MLP hidden layers.")
    parser.add_argument("--nn-dropout", type=float, default=0.05, help="Residual MLP dropout.")
    parser.add_argument("--nn-lr", type=float, default=1.0e-3, help="Residual MLP learning rate.")
    parser.add_argument("--nn-weight-decay", type=float, default=1.0e-5, help="Residual MLP weight decay.")
    parser.add_argument("--nn-batch-size", type=int, default=512, help="Residual MLP batch size.")
    parser.add_argument("--nn-epochs", type=int, default=300, help="Residual MLP max epochs.")
    parser.add_argument("--nn-patience", type=int, default=40, help="Residual MLP early-stop patience.")
    parser.add_argument("--nn-huber-delta", type=float, default=1.0, help="Residual MLP Huber delta.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    bundle = prepare_bundle(args.npz)
    harmonic_orders = parse_harmonic_orders(args.harmonics)
    train_first_pass_indices, val_first_pass_indices = split_first_pass(bundle["first_pass_indices"], args.val_fraction)

    phase_rad = estimate_orbital_phase_rad(bundle["sgp4_eci_pos_m"], bundle["sgp4_eci_vel_mps"])
    design_matrix, design_names = build_harmonic_design_matrix(
        time_seconds=bundle["time_seconds"],
        phase_rad=phase_rad,
        phase_origin_rad=phase_rad[int(bundle["first_pass_indices"][0])],
        drift_order=args.drift_order,
        harmonic_orders=harmonic_orders,
    )

    harmonic_coefficients = fit_ridge_regression(
        design_matrix[train_first_pass_indices],
        bundle["residual_rtn_pos_m"][train_first_pass_indices],
        ridge=args.ridge,
    )
    harmonic_prediction_rtn_m = design_matrix @ harmonic_coefficients
    residual_after_harmonic_rtn_m = bundle["residual_rtn_pos_m"] - harmonic_prediction_rtn_m

    residual_nn_prediction_rtn_m = np.zeros_like(harmonic_prediction_rtn_m)
    nn_history: list[dict[str, float]] = []
    residual_nn_stats: dict[str, np.ndarray] | None = None

    if not args.disable_nn:
        direct_query_features, direct_query_feature_names = build_direct_query_features(
            bundle=bundle,
            phase_rad=phase_rad,
            harmonic_prediction_rtn_m=harmonic_prediction_rtn_m,
            harmonic_orders=harmonic_orders,
        )
        residual_model, residual_nn_stats, nn_history = train_residual_network(
            features=direct_query_features,
            targets=residual_after_harmonic_rtn_m,
            train_indices=train_first_pass_indices,
            val_indices=val_first_pass_indices,
            args=args,
        )
        residual_nn_prediction_rtn_m = predict_residual_network(
            model=residual_model,
            stats=residual_nn_stats,
            features=direct_query_features,
            device_name=args.device,
        )
    else:
        direct_query_feature_names = []

    hybrid_prediction_rtn_m = harmonic_prediction_rtn_m + residual_nn_prediction_rtn_m

    split_map = {
        "first_pass_train": train_first_pass_indices,
        "first_pass_validation": val_first_pass_indices if val_first_pass_indices.size > 0 else train_first_pass_indices,
        "gap": bundle["gap_indices"],
        "second_pass": bundle["second_pass_indices"],
        "full_segment": np.arange(bundle["time_seconds"].shape[0], dtype=np.int64),
    }

    harmonic_metrics = {
        split_name: evaluate_by_split(bundle, harmonic_prediction_rtn_m, split_indices)
        for split_name, split_indices in split_map.items()
    }
    hybrid_metrics = {
        split_name: evaluate_by_split(bundle, hybrid_prediction_rtn_m, split_indices)
        for split_name, split_indices in split_map.items()
    }

    predictions_path = output_dir / "predictions_full_segment.npz"
    np.savez_compressed(
        predictions_path,
        time_seconds=bundle["time_seconds"],
        y_true_rtn=bundle["residual_rtn_pos_m"],
        y_harmonic_rtn=harmonic_prediction_rtn_m,
        y_hybrid_rtn=hybrid_prediction_rtn_m,
        y_residual_nn_rtn=residual_nn_prediction_rtn_m,
        orbital_phase_rad=phase_rad,
    )

    plot_path = output_dir / "full_segment_rtn_residual_comparison.png"
    save_full_segment_plot(
        figure_path=plot_path,
        time_seconds=bundle["time_seconds"],
        y_true_rtn=bundle["residual_rtn_pos_m"],
        y_harmonic_rtn=harmonic_prediction_rtn_m,
        y_hybrid_rtn=hybrid_prediction_rtn_m,
        first_pass_indices=bundle["first_pass_indices"],
        gap_indices=bundle["gap_indices"],
        second_pass_indices=bundle["second_pass_indices"],
        train_end_time_sec=bundle["time_seconds"][train_first_pass_indices[-1]],
    )

    training_curve_path = output_dir / "residual_nn_training_curve.png"
    save_training_curve(training_curve_path, nn_history)

    harmonic_coefficients_dict = {
        "feature_names": design_names,
        "R": harmonic_coefficients[:, 0].tolist(),
        "T": harmonic_coefficients[:, 1].tolist(),
        "N": harmonic_coefficients[:, 2].tolist(),
    }

    artifacts = {
        "predictions_full_segment": str(predictions_path),
        "full_segment_rtn_residual_comparison": str(plot_path),
    }
    if nn_history:
        artifacts["residual_nn_training_curve"] = str(training_curve_path)

    result = {
        "config": {
            "npz": str(Path(args.npz).resolve()),
            "output_dir": str(output_dir),
            "harmonics": list(harmonic_orders),
            "drift_order": int(args.drift_order),
            "ridge": float(args.ridge),
            "val_fraction": float(args.val_fraction),
            "disable_nn": bool(args.disable_nn),
            "device": args.device,
            "nn_hidden_width": int(args.nn_hidden_width),
            "nn_hidden_layers": int(args.nn_hidden_layers),
            "nn_dropout": float(args.nn_dropout),
            "nn_lr": float(args.nn_lr),
            "nn_weight_decay": float(args.nn_weight_decay),
            "nn_batch_size": int(args.nn_batch_size),
            "nn_epochs": int(args.nn_epochs),
            "nn_patience": int(args.nn_patience),
            "nn_huber_delta": float(args.nn_huber_delta),
        },
        "split_sizes": {
            "first_pass_points": int(bundle["first_pass_indices"].size),
            "train_first_pass_points": int(train_first_pass_indices.size),
            "val_first_pass_points": int(val_first_pass_indices.size),
            "gap_points": int(bundle["gap_indices"].size),
            "second_pass_points": int(bundle["second_pass_indices"].size),
            "full_segment_points": int(bundle["time_seconds"].shape[0]),
        },
        "harmonic_coefficients": harmonic_coefficients_dict,
        "direct_query_feature_names": direct_query_feature_names,
        "training_history": nn_history,
        "metrics": {
            "harmonic_only": harmonic_metrics,
            "harmonic_plus_residual_nn": hybrid_metrics,
        },
        "artifacts": artifacts,
    }

    if residual_nn_stats is not None:
        result["residual_nn_stats"] = {
            key: value.tolist() for key, value in residual_nn_stats.items()
        }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"]["harmonic_plus_residual_nn"]["second_pass"], indent=2))
    print(f"Saved metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
