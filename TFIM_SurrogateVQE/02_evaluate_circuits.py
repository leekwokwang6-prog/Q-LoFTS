"""Replay archived level-one labels or collect new labels using the original noise logic."""

import argparse
import multiprocessing
import shutil
import time

import numpy as np

from surrogate_vqe.common import (
    guard_output, load_config, load_pickle, resolve_path, save_pickle, sha256, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--mode", choices=("replay", "regenerate"), default="replay")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--sample-root", default="outputs/sampling")
    parser.add_argument("--archive-root", default="frozen/data")
    parser.add_argument("--output-root", default="outputs/evaluation")
    parser.add_argument("--count", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = config["sample_seeds"] if args.seeds is None else args.seeds
    if len(set(seeds)) != len(seeds):
        parser.error("Seed list must not contain duplicates")
    count = config["evaluate_count"] if args.count is None else args.count
    workers = config["evaluation_workers"] if args.workers is None else args.workers
    if count < 1 or workers < 1:
        parser.error("Count and workers must be positive")
    if args.mode == "replay" and count != config["evaluate_count"]:
        parser.error("Replay preserves the complete archived label file; do not change --count")
    sample_root = resolve_path(args.sample_root)
    archive_root = resolve_path(args.archive_root)
    output_root = guard_output(args.output_root, args.overwrite)
    evaluator = None
    if args.mode == "regenerate":
        from surrogate_vqe.evaluator import VqeEvaluator
        evaluator = VqeEvaluator(workers, config["num_qubits"], config["hamiltonian"], config["noise"])
    for seed in seeds:
        start = time.time()
        relative = f"training_1/run_{seed}"
        archive = archive_root / relative
        sample_dir = sample_root / relative
        destination = output_root / relative
        if args.mode == "replay":
            origin = archive
            for name in ("params.pkl", "samples.pkl", "energy_ZNE_local_dp_co_tm.pkl"):
                if not (origin / name).is_file():
                    raise FileNotFoundError(origin / name)
            archived_parameters = load_pickle(archive / "params.pkl")
            archived_samples = load_pickle(archive / "samples.pkl")
            labels = load_pickle(archive / "energy_ZNE_local_dp_co_tm.pkl")
            if len(labels) != count or len(archived_parameters) != len(archived_samples) or count > len(archived_samples):
                raise ValueError("Archive label count or paired circuit/parameter count is inconsistent")
            values = np.asarray(labels)
            if values.size != count or not np.isfinite(values).all():
                raise ValueError("Archive must contain one finite energy per evaluated circuit")
            if sample_dir.exists():
                for name in ("params.pkl", "samples.pkl"):
                    if not (sample_dir / name).is_file():
                        raise FileNotFoundError(sample_dir / name)
                    generated = load_pickle(sample_dir / name)
                    frozen = archived_parameters if name == "params.pkl" else archived_samples
                    if len(generated) < count:
                        raise ValueError(f"Insufficient sampled rows: {sample_dir / name}")
                    if name == "params.pkl":
                        matches = np.array_equal(np.asarray(generated[:count]), np.asarray(frozen[:count]))
                    else:
                        matches = generated[:count] == frozen[:count]
                    if not matches:
                        raise ValueError(f"Sampling differs from the archived run: {sample_dir / name}")
            destination.mkdir(parents=True, exist_ok=True)
            save_pickle(archived_parameters[:count], destination / "params.pkl")
            save_pickle(archived_samples[:count], destination / "samples.pkl")
            shutil.copy2(origin / "energy_ZNE_local_dp_co_tm.pkl", destination / "energy_ZNE_local_dp_co_tm.pkl")
            note = "Archived labels copied byte for byte with their first 5000 paired parameter/circuit rows; no noise draws consumed."
        else:
            origin = sample_dir
            samples = load_pickle(origin / "samples.pkl")
            parameters = load_pickle(origin / "params.pkl")
            if len(samples) < count or len(parameters) != len(samples):
                raise ValueError("Insufficient circuits or mismatched parameter count")
            pool = multiprocessing.Pool(processes=workers)
            try:
                labels = pool.map(evaluator.evaluate_single, samples[:count])
            finally:
                pool.close()
                pool.join()
            destination.mkdir(parents=True, exist_ok=True)
            save_pickle(parameters[:count], destination / "params.pkl")
            save_pickle(samples[:count], destination / "samples.pkl")
            save_pickle(labels, destination / "energy_ZNE_local_dp_co_tm.pkl")
            note = (
                "Original unseeded multiprocessing noise policy retained. New labels do not "
                "reproduce the unsaved historical worker random states."
            )
        write_json({
            "mode": args.mode, "noise_level": 1, "seed_directory": seed,
            "label_count": len(labels), "input_directory": str(origin),
            "active_parameter_rows": count,
            "data_slices": {"train": [0, config["train_count"]],
                            "validation": [config["val_start"], config["val_start"] + config["val_count"]],
                            "test": [config["test_start"], config["test_start"] + config["test_count"]]},
            "applied_noise_parameters": config["noise"] if args.mode == "regenerate" else None,
            "configured_fresh_noise_parameters": config["noise"],
            "configured_hamiltonian": config["hamiltonian"],
            "historical_noise_invocation_verified": False,
            "workers": workers if args.mode == "regenerate" else 0,
            "randomness_note": note,
            "sha256": {name: sha256(destination / name) for name in (
                "params.pkl", "samples.pkl", "energy_ZNE_local_dp_co_tm.pkl")},
            "duration_seconds": time.time() - start,
        }, destination / "evaluation_metadata.json")
        print(f"seed={seed}: {args.mode}, {len(labels)} level-1 labels -> {destination}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
