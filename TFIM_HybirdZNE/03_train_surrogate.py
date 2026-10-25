"""Train QLoFTS models with the configured RNG and update order."""

import argparse
import json
from pathlib import Path

from tfim_zne.surrogate import train_surrogate, validate_output, validated_training_config


PACKAGE_ROOT = Path(__file__).resolve().parent


def package_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PACKAGE_ROOT / path).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--data-root", help="Defaults to config data_root, or frozen/data.")
    parser.add_argument("--output-root", default="outputs/training")
    parser.add_argument("--seeds", nargs="+", type=int, help="Explicit override of train_seeds.")
    parser.add_argument("--levels", nargs="+", type=int, help="Explicit override of proxy_levels.")
    parser.add_argument("--train-num", "--train_num", dest="train_num", type=int, help="Training sample count; config train_count defaults to 4000.")
    parser.add_argument("--val-num", "--val_num", dest="val_num", type=int, help="Validation sample count; config val_count defaults to 500.")
    parser.add_argument("--test-num", "--test_num", dest="test_num", type=int, help="Untouched post-training test count; config test_count defaults to 500.")
    parser.add_argument("--rank", type=int, help="Number of Fourier product terms H; config hidden_units defaults to 128.")
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, help="Explicit single epoch count for all selected levels.")
    parser.add_argument("--show-config", action="store_true", help="Print effective hyperparameters and ordered data slices without training.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing generated training files.")
    parser.add_argument("--log-every", type=int, default=100, help="Progress frequency; 0 disables per-epoch logging.")
    args = parser.parse_args()
    config_path = package_path(args.config)
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    for argument, key in ((args.train_num, "train_count"), (args.val_num, "val_count"),
                          (args.test_num, "test_count"), (args.rank, "hidden_units")):
        if argument is not None:
            if argument < 1:
                parser.error("train-num, val-num, test-num and rank must be positive integers.")
            config[key] = argument
    try:
        config = validated_training_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    seeds = args.seeds if args.seeds is not None else config.get("train_seeds", [0])
    levels = args.levels if args.levels is not None else config.get("proxy_levels", [5, 7, 9])
    if (not isinstance(seeds, list) or not isinstance(levels, list) or not seeds or not levels
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or any(type(level) is not int for level in levels)):
        parser.error("Seeds and levels must be nonempty, and seeds must be nonnegative.")
    if len(set(seeds)) != len(seeds) or len(set(levels)) != len(levels):
        parser.error("Duplicate seeds or levels are not allowed.")
    allowed_levels = config.get("noise_levels", [1, 3, 5, 7, 9])
    if any(level not in allowed_levels or level not in (1, 3, 5, 7, 9) for level in levels):
        parser.error(f"All selected levels must be in noise_levels: {allowed_levels}")
    default_epochs = {"1": 3500, "3": 2200, "5": 1600, "7": 1800, "9": 1200}
    epoch_map = config.get("epochs", default_epochs)
    try:
        jobs = [(level, seed, args.epochs if args.epochs is not None else epoch_map[str(level)])
                for level in levels for seed in seeds]
    except KeyError as exc:
        parser.error(f"Missing epoch setting for folding factor {exc}")
    if any(type(epochs) is not int or epochs < 1 for _, _, epochs in jobs) or args.log_every < 0:
        parser.error("Epochs must be positive, and log-every must be nonnegative.")
    config["epochs"] = dict(epoch_map)
    for level, _, epochs in jobs:
        config["epochs"][str(level)] = epochs
    intervals = [(0, config["train_count"]),
                 (config["val_start"], config["val_start"] + config["val_count"]),
                 (config["test_start"], config["test_start"] + config["test_count"])]
    if args.show_config:
        print(json.dumps({"train_num": config["train_count"], "val_num": config["val_count"], "test_num": config["test_count"],
                          "rank": config.get("hidden_units", 128), "max_freq": config.get("max_freq", 1),
                          "epochs": {str(level): config["epochs"][str(level)] for level in levels}, "seeds": seeds,
                          "slices": {"train": list(intervals[0]), "validation": list(intervals[1]), "test": list(intervals[2])},
                          "validation_role": "scheduler monitor only", "test_role": "evaluated once after final checkpoint"}, indent=2))
        return
    data_root = package_path(args.data_root or config.get("data_root", "frozen/data"))
    output_root = package_path(args.output_root)
    if PACKAGE_ROOT not in output_root.parents:
        parser.error("Training outputs must stay inside the submission folder.")
    # Check every destination before starting a potentially long sequence of jobs.
    for level, seed, _ in jobs:
        validate_output(output_root, level, seed, args.overwrite)
    for level, seed, epochs in jobs:
        train_surrogate(data_root, output_root, config, level, seed, epochs,
                        config_path=config_path, overwrite=args.overwrite,
                        log_every=args.log_every)


if __name__ == "__main__":
    main()
