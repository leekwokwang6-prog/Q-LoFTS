"""Full-batch noise-level-one surrogate training with original RNG ordering."""

import hashlib
import json
import os
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .model import QLoFTSModel, set_seed, total_param_norm


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_CONFIG = {
    "noise_level": 1,
    "input_dim": 19,
    "train_seeds": [0],
    "num_samples": 5000,
    "train_count": 4000,
    "val_start": 4000,
    "val_count": 500,
    "test_start": 4500,
    "test_count": 500,
    "hidden_units": 128,
    "max_freq": 1,
    "dropout_rate": 0,
    "epochs": 3500,
}


def package_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PACKAGE_ROOT / path).resolve()


def check_output_root(value):
    """Limit generated files to writable locations in this submission."""
    path = package_path(value)
    try:
        relative = path.relative_to(PACKAGE_ROOT)
    except ValueError as exc:
        raise ValueError("Output paths must remain inside the submission folder.") from exc
    if not relative.parts or relative.parts[0].lower() in {
        "frozen", "surrogate_vqe", "config"
    }:
        raise ValueError("The package root, frozen/, surrogate_vqe/, and config/ are protected.")
    return path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_info():
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def normalized_training_config(config):
    result = dict(DEFAULT_TRAINING_CONFIG)
    result.update(config)
    if result["noise_level"] != 1:
        raise ValueError("This package trains only noise level 1.")
    for key in ("input_dim", "num_samples", "train_count", "val_count", "test_count", "hidden_units", "max_freq", "epochs"):
        if not isinstance(result[key], int) or isinstance(result[key], bool) or result[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("val_start", "test_start"):
        if not isinstance(result[key], int) or isinstance(result[key], bool) or result[key] < 0:
            raise ValueError(f"{key} must be a non-negative integer.")
    intervals = list(training_slices(result).values())
    if any(end > result["num_samples"] for start, end in intervals):
        raise ValueError("Requested slices exceed num_samples; change explicit bounds without shifting slices automatically.")
    if any(max(a[0], b[0]) < min(a[1], b[1]) for index, a in enumerate(intervals) for b in intervals[index + 1:]):
        raise ValueError("Training, validation, and held-out test slices must be disjoint.")
    if not 0 <= result["dropout_rate"] < 1:
        raise ValueError("dropout_rate must be in [0, 1).")
    seeds = result["train_seeds"]
    if not isinstance(seeds, list) or not seeds or any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds
    ) or len(set(seeds)) != len(seeds):
        raise ValueError("train_seeds must be a non-empty list of distinct non-negative integers.")
    return result


def training_slices(config):
    return {"train": [0, config["train_count"]],
            "validation": [config["val_start"], config["val_start"] + config["val_count"]],
            "test": [config["test_start"], config["test_start"] + config["test_count"]]}


def load_training_tensors(data_root, seed, config):
    config = normalized_training_config(config)
    data_directory = package_path(data_root) / "training_1" / f"run_{seed}"
    samples_path = data_directory / "params.pkl"
    labels_path = data_directory / "energy_ZNE_local_dp_co_tm.pkl"
    with samples_path.open("rb") as stream:
        samples = pickle.load(stream)
    with labels_path.open("rb") as stream:
        labels = pickle.load(stream)
    intervals = training_slices(config)
    required_rows = max(end for start, end in intervals.values())
    if min(len(samples), len(labels)) < required_rows:
        raise ValueError("The input files do not contain the requested train/validation/test slices.")
    tensors = []
    for start, end in intervals.values():
        x = np.array(samples[start:end])
        y = np.array(labels[start:end])
        if x.ndim != 2 or x.shape[1] < config["input_dim"]:
            raise ValueError("Each parameter sample must contain at least input_dim entries.")
        if y.size != end - start:
            raise ValueError("Each energy label must contain one scalar.")
        split = (torch.tensor(x[:, :config["input_dim"]], dtype=torch.float32),
                 torch.tensor(y, dtype=torch.float32).reshape(-1, 1))
        if not all(bool(torch.isfinite(tensor).all()) for tensor in split):
            raise ValueError("Training, validation, or test data contain non-finite values.")
        tensors.extend(split)
    return tuple(tensors), (samples_path, labels_path)


def train_model(train_samples, train_labels, validation_samples, validation_labels,
                architecture, epochs, log_every=100):
    """Train without reseeding, reshuffling, minibatches, or early stopping."""
    model = QLoFTSModel(**architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, verbose=True
    )
    loss_fn = nn.MSELoss()
    history = {"train_loss": [], "val_loss": [], "grad_norm": []}
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
            prediction_val = model(validation_samples)
            loss_val = loss_fn(prediction_val, validation_labels)
        scheduler.step(loss_val)
        history["train_loss"].append(loss.item())
        history["val_loss"].append(loss_val.item())
        history["grad_norm"].append(g_norm)
        if log_every and ((step + 1) % log_every == 0 or step + 1 == epochs):
            print(f"Step {step + 1}: Train Loss = {loss.item():.6f}, "
                  f" Validation Loss = {loss_val.item():.6f}")
    return model, history


def train_surrogates(config, data_root="frozen/data", output_root="outputs/training",
                     overwrite=False, log_every=100):
    config = normalized_training_config(config)
    output_root = check_output_root(output_root)
    if not isinstance(log_every, int) or log_every < 0:
        raise ValueError("log_every must be a non-negative integer.")
    run_directories = [
        output_root / "training_1" / f"run_{seed}" for seed in config["train_seeds"]
    ]
    for directory in run_directories:
        check_output_root(directory)
        for filename in ("best_model.pth", "history.json", "metadata.json", "test_metrics.json", "test_predictions.pkl"):
            check_output_root(directory / filename)
        if not overwrite and directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Output already exists: {directory}. Use --overwrite explicitly.")
    architecture = {
        "input_dim": config["input_dim"],
        "hidden_units": config["hidden_units"],
        "max_freq": config["max_freq"],
        "dropout_rate": config["dropout_rate"],
    }
    results = []
    for seed, directory in zip(config["train_seeds"], run_directories):
        started = datetime.now(timezone.utc).isoformat()
        started_clock = time.perf_counter()
        set_seed(seed)
        tensors, input_paths = load_training_tensors(data_root, seed, config)
        model, history = train_model(*tensors[:4], architecture=architecture,
                                     epochs=config["epochs"], log_every=log_every)
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = directory / "best_model.pth"
        torch.save(model.state_dict(), checkpoint_path)
        (directory / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        model.eval()
        with torch.no_grad():
            test_prediction = model(tensors[4])
            test_mse = nn.MSELoss()(test_prediction, tensors[5]).item()
        predictions_path = directory / "test_predictions.pkl"
        with predictions_path.open("wb") as stream:
            pickle.dump(test_prediction, stream)
        test_metrics = {
            "metric": "MSE(prediction, same-level held-out test label)", "mse": test_mse,
            "count": config["test_count"], "slice": training_slices(config)["test"],
            "evaluation_count": 1, "checkpoint_selection": "final_epoch",
            "used_for_optimizer_scheduler_or_checkpoint_selection": False,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "predictions_sha256": sha256_file(predictions_path),
        }
        (directory / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n", encoding="utf-8")
        source_paths = [Path(__file__), Path(__file__).with_name("model.py"),
                        PACKAGE_ROOT / "03_train_surrogate.py"]
        metadata = {
            "noise_level": 1,
            "seed": seed,
            "hyperparameters": {
                "train_num": config["train_count"], "val_num": config["val_count"], "test_num": config["test_count"],
                "rank": config["hidden_units"], "epochs": config["epochs"],
            },
            "architecture": architecture,
            "slices": training_slices(config),
            "validation_role": "ReduceLROnPlateau scheduler monitor; not an untouched holdout",
            "test_role": "held out from optimizer, scheduler, and checkpoint selection; evaluated once after the final checkpoint",
            "sample_budget": config["num_samples"],
            "test_evaluation": test_metrics,
            "training": {"batch": "full", "dtype": "float32", "device": "cpu",
                         "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
                         "gradient_clip_max_norm": 1.0, "scheduler": "ReduceLROnPlateau",
                         "scheduler_factor": 0.5, "scheduler_patience": 100,
                         "early_stopping": False, "checkpoint_epoch": config["epochs"],
                          "checkpoint_selection": "final epoch"},
            "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in input_paths],
            "sources": [{"path": str(path.relative_to(PACKAGE_ROOT)), "sha256": sha256_file(path)}
                        for path in source_paths],
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "actual_runtime": runtime_info(),
            "started_utc": started,
            "elapsed_seconds": time.perf_counter() - started_clock,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Saved final-epoch model to {checkpoint_path}")
        results.append(directory)
    return results
