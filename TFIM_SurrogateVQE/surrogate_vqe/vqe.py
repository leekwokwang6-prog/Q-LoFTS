"""Optimize the noise-level-one surrogate with the original COBYLA settings."""

import csv
import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.optimize as optimize
import tensorflow as tf
import torch

from .model import load_model as load_surrogate
from .common import guard_output, resolve_path


def set_run_seed(seed):
    """Preserve the original NumPy/Torch seed calls, including NumPy's double reset."""
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optimize_surrogate(model, single_circuit, parameter_seed, max_iteration=2000, tol=0.0001):
    """Return the original surrogate-only objective and every evaluated parameter vector."""
    np.random.seed(parameter_seed)
    tf.random.set_seed(parameter_seed)
    structure, depth = single_circuit[0], single_circuit[1]
    param_num = sum(gate[0] in ("rx", "ry", "rz") for gate in structure)
    if param_num != model.fourier_layer.d:
        raise ValueError(
            "Circuit parameter count does not match the surrogate input dimension: "
            f"{param_num} != {model.fourier_layer.d}"
        )
    initial_parameters = np.random.normal(
        loc=0, scale=1 / (8 * (depth + 2)), size=param_num
    )
    trajectory = []

    def cost_wrapper(p_numpy):
        p_tensor = torch.tensor(p_numpy, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction = model(p_tensor)
            energy = prediction.item()
        trajectory.append((np.array(p_numpy, copy=True), float(energy)))
        return energy

    result = optimize.minimize(
        cost_wrapper,
        initial_parameters,
        method="COBYLA",
        options={"maxiter": int(max_iteration), "tol": float(tol)},
    )
    return result, initial_parameters, trajectory


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_pickle(value, path):
    with Path(path).open("wb") as handle:
        pickle.dump(value, handle)


def run_vqe(config, checkpoint_seeds=None, parameter_seeds=None, max_iteration=None,
            model_root=None, sample_root=None, output_root=None, overwrite=False):
    """Run parameter seeds outside model seeds, matching the original 25-run order."""
    if config.get("noise_level") != 1:
        raise ValueError("This workflow supports noise level 1 only")
    settings = config["vqe"]
    checkpoint_seeds = list(settings["checkpoint_seeds"] if checkpoint_seeds is None else checkpoint_seeds)
    parameter_seeds = list(settings["parameter_seeds"] if parameter_seeds is None else parameter_seeds)
    max_iteration = settings["max_iteration"] if max_iteration is None else max_iteration
    if int(max_iteration) < 1:
        raise ValueError("max_iteration must be positive")
    if not checkpoint_seeds or not parameter_seeds:
        raise ValueError("At least one checkpoint seed and parameter seed are required")
    if len(set(checkpoint_seeds)) != len(checkpoint_seeds) or len(set(parameter_seeds)) != len(parameter_seeds):
        raise ValueError("Seed lists must not contain duplicates")
    model_root = resolve_path(settings["model_root"] if model_root is None else model_root)
    sample_root = resolve_path(settings["sample_root"] if sample_root is None else sample_root)
    output_root = resolve_path(settings["output_root"] if output_root is None else output_root)
    tolerance = settings.get("tolerance", 0.0001)
    tasks = []
    for parameter_seed in parameter_seeds:
        for checkpoint_seed in checkpoint_seeds:
            destination = output_root / f"parameter_seed_{parameter_seed}" / f"model_seed_{checkpoint_seed}"
            checkpoint = model_root / "training_1" / f"run_{checkpoint_seed}" / "best_model.pth"
            sample = sample_root / "training_1" / f"run_{checkpoint_seed}" / "samples.pkl"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if not sample.is_file():
                raise FileNotFoundError(sample)
            destination = guard_output(destination, overwrite=overwrite)
            tasks.append((parameter_seed, checkpoint_seed, checkpoint, sample, destination))

    summaries = []
    for parameter_seed, checkpoint_seed, checkpoint, sample_path, destination in tasks:
        start_time = time.time()
        set_run_seed(checkpoint_seed)
        model, architecture = load_surrogate(checkpoint)
        with sample_path.open("rb") as handle:
            samples = pickle.load(handle)
        if not samples:
            raise ValueError(f"No circuit samples in {sample_path}")
        result, initial_parameters, trajectory = optimize_surrogate(
            model, samples[0], parameter_seed, max_iteration, tolerance
        )
        destination.mkdir(parents=True, exist_ok=True)
        trace = [energy for _, energy in trajectory]
        _save_pickle([float(result.fun)], destination / "energy.pkl")
        _save_pickle([result.x], destination / "param.pkl")
        _save_pickle([trace], destination / "energy_trace.pkl")
        with (destination / "trajectory.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["evaluation", "surrogate_energy"] + [f"theta_{i}" for i in range(len(result.x))])
            for index, (parameters, energy) in enumerate(trajectory, 1):
                writer.writerow([index, energy] + parameters.tolist())
        model_metadata_path = checkpoint.parent / "metadata.json"
        model_metadata = None
        if model_metadata_path.is_file():
            model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
        metadata = {
            "schema_version": 2,
            "method": "surrogate",
            "polynomial_degree": None,
            "ansatz": {"family": "tfim_single_layer", "num_qubits": config["num_qubits"],
                       "num_layers": config["num_layers"], "input_dim": config["input_dim"]},
            "hamiltonian": config["hamiltonian"],
            "noise": config["noise"],
            "algorithm": "surrogate-only VQE",
            "noise_level": 1,
            "energy_definition": "Prediction of the noise-level-one surrogate at the returned parameters",
            "parameter_seed": int(parameter_seed),
            "checkpoint_seed": int(checkpoint_seed),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "samples": str(sample_path),
            "samples_sha256": _sha256(sample_path),
            "sample_index": 0,
            "architecture": architecture,
            "model_training_metadata": model_metadata,
            "optimizer": {"method": "COBYLA", "maxiter": int(max_iteration), "tol": float(tolerance)},
            "max_iteration": int(max_iteration),
            "tolerance": float(tolerance),
            "optimizer_parameter_dtype": str(result.x.dtype),
            "surrogate_input_dtype": "torch.float32",
            "initial_parameters": initial_parameters.tolist(),
            "final_parameters": result.x.tolist(),
            "final_surrogate_energy": float(result.fun),
            "final_reported_energy": float(result.fun),
            "seed_roles": {"parameter_seed": "shared VQE initialization",
                           "checkpoint_seed": "training-data/model run, reused across initializations"},
            "paper_metric": "abs(final_reported_energy - paired noiseless-VQE optimized reference)",
            "evaluations": int(result.nfev),
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "elapsed_seconds": time.time() - start_time,
            "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__, "tensorflow": tf.__version__},
        }
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        summaries.append(metadata)
        print(
            f"parameter_seed={parameter_seed}, model_seed={checkpoint_seed}, "
            f"surrogate_energy={float(result.fun):.10f}, evaluations={int(result.nfev)}"
        )
    return summaries
