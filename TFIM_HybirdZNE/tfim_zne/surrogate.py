"""Fourier surrogate and full-batch training.

``best_model.pth`` is the historical filename. It stores the final epoch,
because the original early-stopping code was commented out.
"""

import hashlib
import json
import math
import os
import pickle
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


class FourierProductLayer(nn.Module):
    def __init__(self, input_dim, hidden_units, max_freq, use_geometric_mean=True):
        super().__init__()
        self.d = input_dim
        self.H = hidden_units
        self.Lambda = max_freq
        self.use_geometric_mean = use_geometric_mean

        self.a_coeffs = nn.Parameter(torch.empty(hidden_units, input_dim, max_freq))
        self.b_coeffs = nn.Parameter(torch.empty(hidden_units, input_dim, max_freq))
        self.c_coeffs = nn.Parameter(torch.empty(hidden_units, input_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.a_coeffs, mean=0.0, std=0.001)
        nn.init.normal_(self.b_coeffs, mean=0.0, std=0.001)
        nn.init.constant_(self.c_coeffs, 1.0)

    def forward(self, x):
        batch_size = x.shape[0]
        k_vec = torch.arange(1, self.Lambda + 1, device=x.device).float()
        x_expanded = x.unsqueeze(1).unsqueeze(-1)
        kx = x_expanded * k_vec

        term_cos = self.a_coeffs * torch.cos(kx)
        term_sin = self.b_coeffs * torch.sin(kx)
        fourier_sum = torch.sum(term_cos + term_sin, dim=-1)
        phi = self.c_coeffs + fourier_sum

        y_hidden = torch.prod(phi, dim=2)
        return y_hidden


class QLoFTSModel(nn.Module):
    def __init__(self, input_dim, hidden_units=32, max_freq=4, dropout_rate=0.2):
        super().__init__()
        self.fourier_layer = FourierProductLayer(
            input_dim, hidden_units, max_freq,
            use_geometric_mean=(input_dim > 10)
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_head = nn.Linear(hidden_units, 1)
        nn.init.normal_(self.output_head.weight, std=0.1)

    def forward(self, x):
        feat_fourier = self.fourier_layer(x)
        combined = feat_fourier
        combined = self.dropout(combined)
        out = self.output_head(combined)
        return out


def total_param_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


OUTPUT_FILENAMES = (
    "best_model.pth", "result.pkl", "final_prediction.pkl", "test_metrics.json",
    "test_predictions.pkl", "history.json", "metadata.json",
)
OBSOLETE_OUTPUT_FILENAMES = ("result_ZNE.pkl", "final_prediction_ZNE.pkl", "final_prediction_train.pkl")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output(output_root, noise_level, seed, overwrite=False):
    """Keep preserved inputs and previously generated results safe by default."""
    package_root = Path(__file__).resolve().parents[1]
    output_root = Path(output_root).resolve()
    if package_root not in output_root.parents:
        raise ValueError("Training outputs must stay inside this submission folder.")
    protected_roots = [package_root / name for name in ("frozen", "config", "tfim_zne")]
    for protected_root in protected_roots:
        protected_root = protected_root.resolve()
        try:
            within_protected = os.path.commonpath([str(output_root), str(protected_root)]) == str(protected_root)
        except ValueError:
            within_protected = False
        if within_protected:
            raise ValueError(f"Training output must be outside preserved source/data directories: {protected_root}")
    run_dir = (output_root / f"training_{noise_level}" / f"run_{seed}").resolve()
    if output_root not in run_dir.parents:
        raise ValueError("The run directory must not resolve outside the selected output root.")
    for name in OUTPUT_FILENAMES + OBSOLETE_OUTPUT_FILENAMES:
        if run_dir not in (run_dir / name).resolve().parents:
            raise ValueError("Generated training files must not resolve outside the run directory.")
    conflicts = [str(run_dir / name) for name in OUTPUT_FILENAMES + OBSOLETE_OUTPUT_FILENAMES if (run_dir / name).exists()]
    if conflicts and not overwrite:
        raise FileExistsError(
            "Training outputs already exist; select another --output-root or explicitly use "
            "--overwrite:\n" + "\n".join(conflicts)
        )
    return run_dir


def _load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _save_pickle(value, path):
    with Path(path).open("wb") as handle:
        pickle.dump(value, handle)


def validated_training_config(config):
    """Validate explicit ordered training, validation, and untouched test slices."""
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a JSON object.")
    result = dict(config)
    defaults = {"input_dim": 19, "train_count": 4000, "val_start": 4000, "val_count": 500,
                "test_start": 4500, "test_count": 500, "hidden_units": 128, "max_freq": 1,
                "dropout_rate": 0.0}
    for name, default in defaults.items():
        result.setdefault(name, default)
    for name in ("input_dim", "train_count", "val_count", "test_count", "hidden_units", "max_freq"):
        if type(result[name]) is not int or result[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name in ("val_start", "test_start"):
        if type(result[name]) is not int or result[name] < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    dropout = result["dropout_rate"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("dropout_rate must be finite and in [0, 1).")
    intervals = [(0, result["train_count"]),
                 (result["val_start"], result["val_start"] + result["val_count"]),
                 (result["test_start"], result["test_start"] + result["test_count"])]
    if intervals[0][1] > intervals[1][0] or intervals[1][1] > intervals[2][0]:
        raise ValueError("Training, validation, and test slices must be ordered and must not overlap; split starts are never moved automatically.")
    for name in ("num_samples", "evaluate_count"):
        if name in result:
            if type(result[name]) is not int or result[name] < 1:
                raise ValueError(f"{name} must be a positive integer.")
            if intervals[-1][1] > result[name]:
                raise ValueError(f"Requested split end {intervals[-1][1]} exceeds configured {name}={result[name]}.")
    return result


def load_training_splits(data_root, noise_level, seed, config):
    """Return train/validation/test tensors without shuffling or sampling draws."""
    config = validated_training_config(config)
    if noise_level not in (1, 3, 5, 7, 9):
        raise ValueError("Surrogate labels must use folding factor 1, 3, 5, 7, or 9.")
    data_root = Path(data_root)
    params_path = data_root / "training_1" / f"run_{seed}" / "params.pkl"
    labels_path = data_root / f"training_{noise_level}" / f"run_{seed}" / "energy_ZNE_local_dp_co_tm.pkl"
    samples = _load_pickle(params_path)
    labels = _load_pickle(labels_path)
    input_dim = config["input_dim"]
    train_count = config["train_count"]
    val_start, val_count = config["val_start"], config["val_count"]
    test_start, test_count = config["test_start"], config["test_count"]
    required_rows = test_start + test_count
    if len(samples) < required_rows or len(labels) < required_rows:
        raise ValueError(f"Seed {seed}, level {noise_level} needs {required_rows} rows; "
                         f"found {len(samples)} samples and {len(labels)} labels.")

    def make_split(start, count):
        # Keep the original np.array -> first 19 columns -> float32 conversion.
        x = np.array(samples[start:start + count])
        if x.ndim != 2 or x.shape[1] < input_dim:
            raise ValueError(f"Sample matrix must have at least {input_dim} columns, got {x.shape}.")
        x = x[:, :input_dim]
        y = np.array(labels[start:start + count])
        if y.size != count:
            raise ValueError("Expected one scalar energy label per sample.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Samples and labels must contain finite values.")
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32).reshape(-1, 1))

    train_x, train_y = make_split(0, train_count)
    val_x, val_y = make_split(val_start, val_count)
    test_x, test_y = make_split(test_start, test_count)
    return (train_x, train_y, val_x, val_y, test_x, test_y), (params_path, labels_path)


def train_surrogate(data_root, output_root, config, noise_level, seed, epochs,
                    config_path=None, overwrite=False, log_every=100):
    """Train on CPU in the original full-batch operation and RNG order."""
    config = validated_training_config(config)
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if type(log_every) is not int or log_every < 0:
        raise ValueError("log_every must be a non-negative integer.")
    run_dir = validate_output(output_root, noise_level, seed, overwrite)
    set_seed(seed)
    splits, data_paths = load_training_splits(data_root, noise_level, seed, config)
    train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = splits
    model = QLoFTSModel(
        input_dim=train_samples.shape[1],
        hidden_units=int(config.get("hidden_units", 128)),
        max_freq=int(config.get("max_freq", 1)),
        dropout_rate=float(config.get("dropout_rate", 0)),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100, verbose=True
    )
    loss_fn = nn.MSELoss()
    history = {'train_loss': [], 'val_loss': [], 'grad_norm': []}

    for step in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(train_samples)
        loss = loss_fn(prediction, train_labels)
        loss.backward()
        g_norm = total_param_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            prediction_val = model(val_samples)
            loss_val = loss_fn(prediction_val, val_labels)

        scheduler.step(loss_val)
        history['train_loss'].append(loss.item())
        history['val_loss'].append(loss_val.item())
        history['grad_norm'].append(g_norm)
        if log_every and ((step + 1) % log_every == 0 or step + 1 == epochs):
            print(f"Level {noise_level}, seed {seed}, step {step + 1}: "
                  f"Train Loss = {loss.item():.6f}, Validation Loss = {loss_val.item():.6f}")

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "best_model.pth")
    model.eval()
    with torch.no_grad():
        test_predictions = model(test_samples)
        test_mse = float(loss_fn(test_predictions, test_labels).item())
    _save_pickle(test_predictions, run_dir / "test_predictions.pkl")
    _save_pickle(test_predictions, run_dir / "final_prediction.pkl")
    _save_pickle(np.array([test_mse]), run_dir / "result.pkl")
    if overwrite:
        for name in OBSOLETE_OUTPUT_FILENAMES:
            obsolete = run_dir / name
            if obsolete.is_file():
                obsolete.unlink()
    test_metrics = {"metric": "MSE", "mse": test_mse, "test_num": config["test_count"],
                    "test_slice": [config["test_start"], config["test_start"] + config["test_count"]],
                    "test_forward_evaluations": 1, "evaluation_stage": "after_final_checkpoint_saved",
                    "targets": "energy labels at the same folding factor and parameter rows",
                    "noise_level": noise_level, "checkpoint_sha256": sha256_file(run_dir / "best_model.pth")}
    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n", encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    package_root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__).resolve(), package_root / "03_train_surrogate.py"]
    metadata = {
        "noise_level": noise_level, "seed": seed, "epochs": epochs,
        "train_num": int(config.get("train_count", 4000)),
        "val_num": int(config.get("val_count", 500)),
        "test_num": config["test_count"],
        "rank": model.fourier_layer.H,
        "model_architecture": {"input_dim": model.fourier_layer.d, "hidden_units": model.fourier_layer.H,
                               "rank": model.fourier_layer.H, "max_freq": model.fourier_layer.Lambda,
                               "dropout_rate": model.dropout.p},
        "completed_epochs": len(history["train_loss"]),
        "slices": {"train": [0, int(config.get("train_count", 4000))],
                   "validation": [config["val_start"], config["val_start"] + config["val_count"]],
                   "test": [config["test_start"], config["test_start"] + config["test_count"]]},
        "training_protocol": "ordered_train_validation_test",
        "validation_role": "ReduceLROnPlateau scheduler monitor only",
        "test_role": "One evaluation after final-epoch training and checkpoint saving; never used by optimizer, scheduler, or epoch selection",
        "test_forward_evaluations": 1,
        "compatibility_outputs": {"result.pkl": "final test MSE as a one-element array", "final_prediction.pkl": "alias of post-training test_predictions.pkl"},
        "frozen_checkpoint_protocol_compliance": "Not asserted; this metadata describes this newly trained output",
        "checkpoint_selection": "final_epoch", "checkpoint_filename": "best_model.pth",
        "early_stopping_enabled": False, "shuffle": False, "full_batch": True,
        "device": "cpu", "config": config,
        "config_sha256": sha256_file(config_path) if config_path else None,
        "source_sha256": {str(path.relative_to(package_root)): sha256_file(path)
                          for path in source_paths if path.is_file()},
        "input_sha256": {str(path.resolve()): sha256_file(path) for path in data_paths},
        "final_train_loss": history["train_loss"][-1],
        "final_validation_loss": history["val_loss"][-1],
        "final_test_mse": test_mse,
        "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 1e-4},
        "scheduler": {"name": "ReduceLROnPlateau", "mode": "min", "factor": 0.5, "patience": 100},
        "gradient_clip_max_norm": 1.0,
        "runtime": {"python": sys.version, "platform": platform.platform(),
                    "numpy": np.__version__, "torch": torch.__version__,
                    "torch_threads": torch.get_num_threads(),
                    "cuda_available": torch.cuda.is_available()},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved final-epoch surrogate to {run_dir}")
    return model, history, metadata
