"""Replay supplied energy labels or regenerate them with circuit simulation."""
import argparse
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from tfim_zne.common import ROOT, load_config, load_pickle, output_guard, save_pickle, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--mode", choices=["replay", "regenerate"], default="replay")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--levels", nargs="+", type=int,
                        help="Supported folding factors: 1,3,5,7,9.")
    parser.add_argument("--sample-root", type=Path, default=ROOT / "outputs" / "sampling")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "evaluation")
    parser.add_argument("--processes", type=int, default=10)
    parser.add_argument("--count", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg["sample_seeds"]
    levels = args.levels if args.levels is not None else cfg["noise_levels"]
    if not set(levels) <= {1, 3, 5, 7, 9} or args.processes <= 0:
        parser.error("Levels must be 1,3,5,7,9; processes must be positive. Level-zero labels are not part of this workflow.")
    root = output_guard(args.output_root, args.overwrite)
    records = []
    if args.mode == "replay":
        if args.count is not None:
            parser.error("Replay preserves complete historical files; --count applies only to regeneration.")
        for seed in seeds:
            files = [Path(f"training_1/run_{seed}/params.pkl")]
            files += [Path(f"training_{level}/run_{seed}/energy_ZNE_local_dp_co_tm.pkl") for level in levels]
            for relative in files:
                source = ROOT / "frozen/data" / relative
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if relative.name == "params.pkl":
                    params = load_pickle(source)
                    if len(params) < cfg["evaluate_count"]:
                        raise ValueError("Insufficient archived parameters for the active dataset.")
                    sampled_path = args.sample_root / relative
                    if sampled_path.is_file():
                        sampled = load_pickle(sampled_path)
                        if len(sampled) < cfg["evaluate_count"] or not np.array_equal(
                            sampled[:cfg["evaluate_count"]], params[:cfg["evaluate_count"]]
                        ):
                            raise ValueError("Sampled parameters differ from the archived label prefix; regenerate labels explicitly.")
                    save_pickle(params[:cfg["evaluate_count"]], target)
                else:
                    shutil.copyfile(source, target)
                records.append({"file": relative.as_posix(), "sha256": sha256(target)})
        print("Replayed historical labels byte for byte. No circuit execution was performed.")
    else:
        count = args.count if args.count is not None else cfg["evaluate_count"]
        if count <= 0:
            parser.error("--count must be positive.")
        from tfim_zne.evaluator import VqeEvaluator
        print("Original unseeded noise and Pool behavior retained: fresh noisy labels are not bitwise replayable.")
        for seed in seeds:
            relative = Path(f"training_1/run_{seed}/params.pkl")
            params = load_pickle(args.sample_root / relative)
            if len(params) < count:
                raise ValueError(f"Only {len(params)} parameter rows; requested {count}.")
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            save_pickle(params[:count], root / relative)
            for level in levels:
                samples = load_pickle(args.sample_root / f"training_{level}/run_{seed}/samples.pkl")
                if len(samples) < count:
                    raise ValueError("Insufficient sampled circuits.")
                evaluator = VqeEvaluator(args.processes, 10, cfg["hamiltonian"], cfg["noise"])
                # No new seed policy: preserve the original Pool/map implementation and gate noise draws.
                with Pool(processes=args.processes) as pool:
                    values = pool.map(evaluator.evaluate_single, samples[:count])
                destination = root / f"training_{level}/run_{seed}/energy_ZNE_local_dp_co_tm.pkl"
                save_pickle(values, destination)
                records.append({"file": destination.relative_to(root).as_posix(), "sha256": sha256(destination)})
                print(f"run_{seed}, lambda={level}: {len(values)} energies")
    write_json({"mode": args.mode, "seeds": seeds, "levels": levels, "config": cfg,
                "active_parameter_rows": cfg["evaluate_count"] if args.mode == "replay" else count,
                "data_slices": {"train": [0, cfg["train_count"]],
                                "validation": [cfg["val_start"], cfg["val_start"] + cfg["val_count"]],
                                "test": [cfg["test_start"], cfg["test_start"] + cfg["test_count"]]},
                "fresh_noise_has_original_unseeded_rng": args.mode == "regenerate", "files": records}, root / "metadata.json")


if __name__ == "__main__":
    main()
