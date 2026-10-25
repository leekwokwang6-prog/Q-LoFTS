"""Optimize hybrid QLoFTS VQE with a quadratic zero-noise energy objective.

Circuit energies at folds 1 and 3 and QLoFTS predictions at 5, 7 and 9
supply the quadratic intercept at each current parameter vector.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import importlib
import json
import pickle
import platform
import re
import sys
import time
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOISE = {
    "two_qubit_channel_depolarizing_p": 0.01,
    "single_qubit_channel_depolarizing_p": 0.0005,
    "bit_flip_p": 0.01,
    "two_qubit_channel_T1": 150000,
    "two_qubit_channel_T2": 120000,
    "two_qubit_channel_gateTime": 100,
    "two_qubit_channel_esp": 0.01,
    "single_qubit_channel_T1": 150000,
    "single_qubit_channel_T2": 120000,
    "single_qubit_channel_gateTime": 32,
    "single_qubit_channel_esp": 0.01,
}
HAMILTONIAN = {
    "pbc": False, "hzz": 1, "hxx": 0, "hyy": 0,
    "hx": 1, "hy": 0, "hz": 0, "sparse": False,
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (PACKAGE_ROOT / path).resolve()


def _arguments(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    initial, _ = pre.parse_known_args(argv)
    config_path = _resolve(initial.config) if initial.config else PACKAGE_ROOT / "config" / "experiment.json"
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    common = config.get("common", {})
    defaults = {**common, **config.get("vqe", {})}
    # Also accept a standalone, flat VQE JSON file.
    defaults = {**{k: v for k, v in config.items() if not isinstance(v, dict)}, **defaults}
    defaults = {key.replace("-", "_"): value for key, value in defaults.items()}
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="JSON file; defaults to config/experiment.json; optional common and vqe sections")
    parser.add_argument("--mode", choices=("hybrid",), default="hybrid")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, help="One global Torch/run/checkpoint seed (original default: 0)")
    seeds.add_argument("--seeds", nargs="+", type=int, help="Explicit run/checkpoint seeds; defaults to vqe.seeds or [0]")
    parameters = parser.add_mutually_exclusive_group()
    parameters.add_argument("--parameter-seed", type=int, help="One NumPy/TF initialization seed; original restart uses 0")
    parameters.add_argument("--parameter-seeds", nargs="+", type=int, help="Shared initialization seeds, independently of checkpoint seeds")
    parser.add_argument("--paper-grid", action="store_true", help="Use parameter seeds 0-4 and QLoFTS model seeds 0-4")
    parser.add_argument("--degree", type=int, choices=(2,), default=2, help="Quadratic extrapolation degree (2)")
    parser.add_argument("--max-iteration", type=int, default=2000, help="COBYLA objective-evaluation budget; original maxiter: 2000")
    parser.add_argument("--model-root", default="frozen/models")
    parser.add_argument("--sample-root", default="outputs/sampling")
    parser.add_argument("--sample-fallback-root", help="Explicit alternative sample root, used only when a primary sample file is missing")
    parser.add_argument("--output-root", default="outputs/vqe")
    parser.add_argument("--check-inputs", "--plan", dest="check_inputs", action="store_true", help="Print the run plan and check file paths without importing TensorFlow/Torch")
    parser.add_argument("--overwrite", action="store_true", help="Replace files in the selected run directories")
    known = {action.dest for action in parser._actions}
    parser.set_defaults(**{key: value for key, value in defaults.items() if key in known})
    args = parser.parse_args(argv)
    args.seeds = [args.seed] if args.seed is not None else (args.seeds if args.seeds is not None else [0])
    args.parameter_seeds = (args.parameter_seeds if args.parameter_seeds is not None
                            else [0 if args.parameter_seed is None else args.parameter_seed])
    if args.paper_grid:
        args.parameter_seeds = list(range(5))
        args.seeds = list(range(5))
    for name in ("seeds", "parameter_seeds"):
        values = getattr(args, name)
        if not isinstance(values, (list, tuple)) or not values or any(type(value) is not int for value in values):
            parser.error(f"{name.replace('_', '-')} must be a nonempty list of integers")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be a nonempty list of distinct integers")
    if type(args.max_iteration) is not int or args.max_iteration < 1:
        parser.error("max-iteration must be a positive integer")
    if args.mode != "hybrid":
        parser.error("mode must be hybrid")
    if type(args.degree) is not int or args.degree != 2:
        parser.error("degree must be 2")
    if not args.parameter_seeds or len(set(args.parameter_seeds)) != len(args.parameter_seeds):
        parser.error("parameter-seeds must be a nonempty list of distinct integers")
    if any(seed < 0 for seed in args.seeds + args.parameter_seeds):
        parser.error("seed and parameter-seed must be nonnegative")
    if config.get("num_qubits", 10) != 10 or config.get("num_layers", 1) != 1:
        parser.error("The preserved VQE experiment requires num_qubits=10 and num_layers=1")
    if config.get("hamiltonian", HAMILTONIAN) != HAMILTONIAN:
        parser.error("The preserved VQE experiment requires the original TFIM Hamiltonian")
    noise = {**DEFAULT_NOISE, **config.get("noise", {}), **config.get("vqe", {}).get("noise", {})}
    args.hamiltonian = dict(config.get("hamiltonian", HAMILTONIAN))
    return args, noise


class _TorchPaths:
    """A module-local checkpoint proxy; the Torch module remains unmodified."""

    def __init__(self, torch_module, model_paths):
        self._torch = torch_module
        self._model_paths = model_paths
        self._prefetched_states = {}

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def checkpoint_state(self, level):
        if level not in self._prefetched_states:
            self._prefetched_states[level] = self._torch.load(self._model_paths[level])
        return self._prefetched_states[level]

    def load(self, file, *args, **kwargs):
        match = re.search(r"training_(5|7|9)[/\\]run_\d+[/\\]best_model\.pth$", str(file))
        if not match:
            raise ValueError(f"Unexpected backend checkpoint path: {file}")
        level = int(match.group(1))
        # The backend call has no options, matching the architecture prefetch.
        if not args and not kwargs and level in self._prefetched_states:
            return self._prefetched_states.pop(level)
        # Preserve any explicitly supplied torch.load options.
        return self._torch.load(self._model_paths[level], *args, **kwargs)


class _QLoFTSModelFactory:
    """Infer each checkpoint shape while retaining the original constructor order."""

    def __init__(self, constructor, torch_paths):
        self._constructor = constructor
        self._torch_paths = torch_paths
        self._levels = iter((5, 7, 9))
        self.metadata = {}

    def __call__(self, input_dim, hidden_units=32, max_freq=4, dropout_rate=0.2):
        try:
            level = next(self._levels)
        except StopIteration:
            raise RuntimeError("Hybrid VQE must construct exactly three surrogate models") from None
        path = self._torch_paths._model_paths[level]
        state = self._torch_paths.checkpoint_state(level)
        names = ("fourier_layer.a_coeffs", "fourier_layer.b_coeffs", "fourier_layer.c_coeffs",
                 "output_head.weight", "output_head.bias")
        try:
            shapes = {name: tuple(state[name].shape) for name in names}
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"Invalid surrogate state dict: {path}") from error
        a_shape = shapes[names[0]]
        if len(a_shape) != 3 or any(dimension < 1 for dimension in a_shape):
            raise ValueError(f"Invalid Fourier coefficient shape in {path}: {a_shape}")
        rank, checkpoint_input_dim, checkpoint_max_freq = map(int, a_shape)
        expected = (a_shape, a_shape, (rank, checkpoint_input_dim), (1, rank), (1,))
        if any(shapes[name] != shape for name, shape in zip(names, expected)):
            raise ValueError(f"Inconsistent surrogate parameter shapes: {path}")
        if checkpoint_input_dim != input_dim:
            raise ValueError(f"Surrogate {path} expects {checkpoint_input_dim} inputs; VQE supplies {input_dim}")
        architecture = {"input_dim": checkpoint_input_dim, "hidden_units": rank,
                        "rank": rank, "max_freq": checkpoint_max_freq,
                        "dropout_rate": dropout_rate}
        details = {"model_architecture": architecture, "architecture_source": "checkpoint_shapes"}
        metadata_path = path.with_name("metadata.json")
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as stream:
                training = json.load(stream)
            declared = training.get("model_architecture", {})
            for name in ("input_dim", "hidden_units", "rank", "max_freq"):
                if name in declared and int(declared[name]) != architecture[name]:
                    raise ValueError(f"Checkpoint architecture disagrees with {metadata_path}: {name}")
            architecture["dropout_rate"] = float(declared.get("dropout_rate", dropout_rate))
            details["training_metadata_path"] = str(metadata_path)
            details["training_metadata_sha256"] = _sha256(metadata_path)
            details["training_hyperparameters"] = {
                name: training[name] for name in ("train_num", "val_num", "rank", "epochs", "completed_epochs")
                if name in training
            }
        self.metadata[level] = details
        # Prefetching state dicts performs no model initialization or random calls.
        return self._constructor(
            input_dim=checkpoint_input_dim, hidden_units=rank, max_freq=checkpoint_max_freq,
            dropout_rate=architecture["dropout_rate"],
        )


class _OptimizationRecorder:
    """Record existing calls; never evaluate an objective for bookkeeping."""

    def __init__(self, scipy_optimize, mode="hybrid"):
        if mode != "hybrid":
            raise ValueError("Only hybrid QLoFTS is available")
        self._optimize = scipy_optimize
        self.mode = mode
        self.stages = []
        self.trajectory = []
        self.current_evaluation = None

    def __getattr__(self, name):
        return getattr(self._optimize, name)

    def record_levels(self, levels, values):
        if self.current_evaluation is None:
            return
        self.current_evaluation["noise_levels"] = {
            str(int(level)): {"energy": float(value), "source": "surrogate" if level in (5, 7, 9) else "real"}
            for level, value in zip(levels, values)
        }

    def minimize(self, objective, initial, *args, **kwargs):
        if self.stages:
            raise RuntimeError("Each VQE run must use exactly one optimization stage")
        stage = {"stage": "main", "evaluations": [], "initial_parameters": initial.tolist()}
        self.stages.append(stage)

        def recorded(parameters):
            evaluation = {
                "evaluation": len(self.trajectory) + 1,
                "stage_evaluation": len(stage["evaluations"]) + 1,
                "stage": stage["stage"], "parameters": parameters.tolist(),
            }
            self.current_evaluation = evaluation
            try:
                value = objective(parameters)
            finally:
                self.current_evaluation = None
            evaluation["objective_energy"] = float(value)
            evaluation["mitigated_energy"] = float(value)
            if "noise_levels" not in evaluation:
                evaluation["noise_levels"] = {"1": {"energy": float(value), "source": "real"}}
            self.trajectory.append(evaluation)
            stage["evaluations"].append(float(value))
            return value

        result = self._optimize.minimize(recorded, initial, *args, **kwargs)
        stage["result"] = {
            "fun": float(result.fun), "x": result.x.tolist(),
            "success": bool(result.success), "message": str(result.message),
            "nfev": int(result.nfev),
        }
        return result


class _NumpyRecorder:
    """Capture the existing fit inputs without changing NumPy or random calls."""

    def __init__(self, numpy_module, recorder):
        self._numpy = numpy_module
        self._recorder = recorder

    def __getattr__(self, name):
        return getattr(self._numpy, name)

    def polyfit(self, levels, values, degree, *args, **kwargs):
        if degree != 2:
            raise ValueError("Hybrid QLoFTS supports quadratic extrapolation only")
        coefficients = self._numpy.polyfit(levels, values, degree, *args, **kwargs)
        self._recorder.record_levels(levels, values)
        return coefficients


def _load_backend(mode="hybrid"):
    """Load a fresh hybrid backend so recording proxies do not accumulate."""
    if mode != "hybrid":
        raise ValueError("Only hybrid QLoFTS is available")
    name = f"{__package__}.hybrid"
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def _run_inputs(args, seed):
    sample_path = _resolve(args.sample_root) / "training_1" / f"run_{seed}" / "samples.pkl"
    if not sample_path.is_file() and args.sample_fallback_root:
        sample_path = _resolve(args.sample_fallback_root) / "training_1" / f"run_{seed}" / "samples.pkl"
    model_paths = {
        level: _resolve(args.model_root) / f"training_{level}" / f"run_{seed}" / "best_model.pth"
        for level in (5, 7, 9)
    }
    required = [sample_path, *model_paths.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing inputs:\n" + "\n".join(missing))
    return sample_path, model_paths


def _output_path(args, checkpoint_seed, parameter_seed, validate=True):
    output = _resolve(args.output_root) / "hybrid" / "degree_2"
    output = output / f"parameter_seed_{parameter_seed}"
    output = output / f"model_seed_{checkpoint_seed}"
    output = output.resolve()
    if PACKAGE_ROOT not in output.parents:
        raise SystemExit("VQE outputs must be inside the submission folder")
    protected = [PACKAGE_ROOT / name for name in ("frozen", "config", "tfim_zne")]
    if any(path == output or path in output.parents for path in protected):
        raise SystemExit("VQE outputs must not overwrite package sources or frozen inputs")
    if validate and output.exists():
        if not output.is_dir() or (any(output.iterdir()) and not args.overwrite):
            raise SystemExit(f"Existing run: {output}. Choose another --output-root or use --overwrite.")
    if validate:
        names = ["duration.pkl", "optimization_stages.json", "trajectory.json", "trajectory.csv", "final_parameters.json", "metadata.json"]
        names += [f"VQE/{name}" for name in ("energy.pkl", "param.pkl", "energy_epoch.pkl", "raw_energy_epoch.pkl")]
        if any(output not in (output / name).resolve().parents for name in names):
            raise SystemExit("Run output files must not resolve outside the selected run directory")
    return output


def _plan_runs(args, validate_outputs=True):
    """Cross each parameter initialization with each QLoFTS model seed."""
    plans = []
    model_seeds = args.seeds
    for parameter_seed in args.parameter_seeds:
        for checkpoint_seed in model_seeds:
            global_seed = checkpoint_seed
            sample_path, model_paths = _run_inputs(args, global_seed)
            output = _output_path(args, checkpoint_seed, parameter_seed, validate_outputs)
            plans.append({"parameter_seed": parameter_seed, "checkpoint_seed": checkpoint_seed,
                          "global_seed": global_seed, "sample_path": sample_path,
                          "model_paths": model_paths, "output": output})
    return plans


def _write_trajectory(output, recorder):
    (output / "trajectory.json").write_text(json.dumps(recorder.trajectory, indent=2), encoding="utf-8")
    columns = ["evaluation", "stage_evaluation", "stage", "objective_energy", "mitigated_energy"]
    for level in (1, 3, 5, 7, 9):
        columns.extend([f"energy_lambda_{level}", f"source_lambda_{level}"])
    columns.append("parameters_json")
    with (output / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in recorder.trajectory:
            row = {name: item[name] for name in columns[:5]}
            for level, observation in item["noise_levels"].items():
                row[f"energy_lambda_{level}"] = observation["energy"]
                row[f"source_lambda_{level}"] = observation["source"]
            row["parameters_json"] = json.dumps(item["parameters"])
            writer.writerow(row)


def _run(args, noise, seed, sample_path, model_paths, output):
    # The first sample supplies the fixed ansatz topology, not a test example.
    with sample_path.open("rb") as stream:
        samples = pickle.load(stream)
    if len(samples) == 0:
        raise SystemExit(f"No circuits in {sample_path}")
    sample = samples[0]
    if not isinstance(sample, (list, tuple)) or len(sample) < 2:
        raise SystemExit("The first sample must contain [gate_structure, depth, ...]")
    parameter_count = sum(gate[0] in ("rx", "ry", "rz") for gate in sample[0])
    if parameter_count != 19:
        raise SystemExit(f"The single-layer TFIM protocol expects 19 parameters; found {parameter_count}")

    backend = _load_backend(args.mode)
    backend.torch = _TorchPaths(backend.torch, model_paths)
    qlofts_models = _QLoFTSModelFactory(backend.QLoFTSModel, backend.torch)
    backend.QLoFTSModel = qlofts_models
    recorder = _OptimizationRecorder(backend.optimize, args.mode)
    backend.optimize = recorder
    backend.np = _NumpyRecorder(backend.np, recorder)
    trainer = backend.VqeTrainerNew(
        n_cir_parallel=1, n_runs=1, max_iteration=args.max_iteration,
        n_qubit=10, hamiltonian=args.hamiltonian,
        noise_param=noise,
    )
    # Match the original main order: construct trainer, set global seed, train.
    backend.set_seed(seed)
    started = time.perf_counter()
    work_item = [sample, args.parameter_seed]
    result = trainer.train_circuit(work_item, seed)
    elapsed = time.perf_counter() - started
    energy, parameters, energy_epoch = result[:3]
    payloads = {
        "energy.pkl": [energy], "param.pkl": [parameters], "energy_epoch.pkl": [energy_epoch],
    }
    payloads["raw_energy_epoch.pkl"] = [result[3]]
    vqe_output = output / "VQE"
    vqe_output.mkdir(parents=True, exist_ok=True)
    for filename, value in payloads.items():
        with (vqe_output / filename).open("wb") as stream:
            pickle.dump(value, stream)
    with (output / "duration.pkl").open("wb") as stream:
        pickle.dump(elapsed, stream)
    (output / "optimization_stages.json").write_text(json.dumps(recorder.stages, indent=2), encoding="utf-8")
    _write_trajectory(output, recorder)
    (output / "final_parameters.json").write_text(json.dumps(parameters.tolist(), indent=2), encoding="utf-8")
    dependencies = {}
    for name in ("numpy", "tensorflow", "tensorcircuit", "torch", "scipy"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "unknown"
    dependencies["tensorflow"] = backend.tf.__version__
    dependencies["torch"] = backend.torch.__version__
    checkpoint_seed = seed
    models = [{"folding_factor": level, "path": str(path), "sha256": _sha256(path),
               **qlofts_models.metadata[level]} for level, path in model_paths.items()]
    metadata = {
        "schema_version": 2, "method": args.mode,
        "mode": args.mode, "global_seed": seed, "parameter_seed": args.parameter_seed,
        "checkpoint_seed": checkpoint_seed,
        "checkpoint_run": seed,
        "seed_roles": {"parameter_seed": "shared NumPy/TensorFlow circuit initialization",
                       "checkpoint_seed": "selected checkpoint training-run seed",
                       "global_seed": "backend NumPy/Torch seed before per-initialization reset",
                       "checkpoint_reuse": "The same selected model seeds are reused across initialization seeds"},
        "original_parameter_seed": 0, "max_iteration": args.max_iteration,
        "hybrid_real_levels": [1, 3],
        "hybrid_surrogate_levels": [5, 7, 9],
        "polynomial_degree": 2,
        "real_levels": [1, 3],
        "surrogate_levels": [5, 7, 9],
        "dtype": "complex128", "backend": "tensorflow", "n_qubit": 10,
        "ansatz_topology_index": 0, "parameter_count": parameter_count, "n_runs": 1, "hamiltonian": args.hamiltonian,
        "ansatz": {"family": "tfim_single_layer", "num_qubits": 10, "num_layers": 1, "input_dim": 19},
        "optimizer": {"method": "COBYLA", "maxiter": args.max_iteration, "tol": 0.0001},
        "initial_parameters": recorder.stages[0]["initial_parameters"],
        "noise": noise,
        "sample_path": str(sample_path), "sample_sha256": _sha256(sample_path),
        "models": models,
        "backend_module": "hybrid",
        "backend_sha256": _sha256(PACKAGE_ROOT / "tfim_zne" / "hybrid.py"),
        "duration_seconds": elapsed, "python": sys.version, "platform": platform.platform(),
        "dependencies": dependencies, "final_energy": float(energy), "final_reported_energy": float(energy),
        "final_parameters": parameters.tolist(),
        "objective_evaluations": len(recorder.trajectory),
        "energy_epoch_note": "Entries are objective evaluations, not training epochs. trajectory.csv includes every executed stage.",
        "reproducibility_note": "Initialization, per-gate random calls and the main optimizer ordering are retained. Re-running also requires the same dependency versions and hardware/backend behavior.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.mode} VQE: {output}")
    print(f"Final energy: {float(energy)}")


def main(argv=None):
    args, noise = _arguments(argv)
    plans = _plan_runs(args, validate_outputs=not args.check_inputs)
    if args.check_inputs:
        print(json.dumps({"schema_version": 2, "method": args.mode,
                         "polynomial_degree": 2,
                         "run_count": len(plans), "runs": [
            {"parameter_seed": task["parameter_seed"], "checkpoint_seed": task["checkpoint_seed"],
             "global_seed": task["global_seed"], "output": str(task["output"]),
             "inputs": [str(task["sample_path"]), *[str(path) for path in task["model_paths"].values()]]}
            for task in plans
        ]}, indent=2))
        return
    for task in plans:
        run_args = argparse.Namespace(**vars(args))
        run_args.parameter_seed = task["parameter_seed"]
        _run(run_args, noise, task["global_seed"], task["sample_path"], task["model_paths"], task["output"])


if __name__ == "__main__":
    main()
