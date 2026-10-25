"""Generate parameterized TFIM samples with the configured seed/draw order."""
import argparse
import random
from pathlib import Path

import numpy as np

from tfim_zne.common import ROOT, load_config, output_guard, save_pickle, write_json
from tfim_zne.sampling import generate_hea_sample_theta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--sample-num", type=int)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "sampling")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg["sample_seeds"]
    count = args.sample_num if args.sample_num is not None else cfg["num_samples"]
    if count <= 0 or cfg["num_qubits"] != 10 or cfg["num_layers"] != 1:
        parser.error("This preserved experiment requires 10 qubits, 1 layer and a positive sample count.")
    root = output_guard(args.output_root, args.overwrite)
    levels = [1, 3, 5, 7, 9]
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        samples = {level: [] for level in levels}
        params = []
        for index in range(count):
            result = generate_hea_sample_theta(10, 1, 1)
            for level, sample in zip(levels, result[:5]):
                samples[level].append(sample)
            params.append(result[5])
        for level in levels:
            save_pickle(samples[level], root / f"training_{level}/run_{seed}/samples.pkl")
        save_pickle(params, root / f"training_1/run_{seed}/params.pkl")
        # Save the post-sampling RNG states without drawing any new randomness.
        save_pickle({"python": random.getstate(), "numpy": np.random.get_state()}, root / f"rng_after_run_{seed}.pkl")
        print(f"run_{seed}: {count} samples, 19 parameters, levels={levels}")
    write_json({"seeds": seeds, "samples_per_seed": count, "levels": levels,
                "rng": "random.seed(run_id), np.random.seed(run_id); 19 uniform draws per sample",
                "config": cfg}, root / "metadata.json")


if __name__ == "__main__":
    main()
