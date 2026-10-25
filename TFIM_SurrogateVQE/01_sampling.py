"""Sample parameterized circuits at noise level 1."""

import argparse
import random
import time

import numpy as np

from surrogate_vqe.common import guard_output, load_config, save_pickle, sha256, write_json
from surrogate_vqe.sampling import sample_circuits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--sample-num", "--sample_num", type=int)
    parser.add_argument("--output-root", default="outputs/sampling")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = config["sample_seeds"] if args.seeds is None else args.seeds
    count = config["num_samples"] if args.sample_num is None else args.sample_num
    if len(set(seeds)) != len(seeds):
        parser.error("Seed list must not contain duplicates")
    root = guard_output(args.output_root, args.overwrite)
    for seed in seeds:
        start = time.time()
        samples, parameters = sample_circuits(seed, count, config["num_qubits"], config["num_layers"])
        destination = root / "training_1" / f"run_{seed}"
        save_pickle(samples, destination / "samples.pkl")
        save_pickle(parameters, destination / "params.pkl")
        save_pickle({"python": random.getstate(), "numpy": np.random.get_state()}, destination / "rng_after_sampling.pkl")
        write_json({
            "noise_level": 1, "seed": seed, "sample_count": count,
            "num_qubits": config["num_qubits"], "num_layers": config["num_layers"],
            "parameter_count": len(parameters[0]), "circuit_metadata": samples[0][1:],
            "parameter_distribution": "NumPy uniform(-pi, pi), scalar draws in original order",
            "params_sha256": sha256(destination / "params.pkl"),
            "samples_sha256": sha256(destination / "samples.pkl"),
            "duration_seconds": time.time() - start,
        }, destination / "sampling_metadata.json")
        print(f"seed={seed}: saved {count} level-1 circuits to {destination}")


if __name__ == "__main__":
    main()
