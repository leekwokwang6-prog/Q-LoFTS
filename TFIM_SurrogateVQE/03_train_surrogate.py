"""Train the noise-level-one Fourier surrogate for VQE."""

import argparse
import json

from surrogate_vqe.training import (
    normalized_training_config, package_path, train_surrogates, training_slices,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--data-root", default="frozen/data")
    parser.add_argument("--output-root", default="outputs/training")
    parser.add_argument("--seeds", type=int, nargs="+", help="Training seeds; default: 0.")
    parser.add_argument("--train-num", "--train_num", dest="train_count", type=int)
    parser.add_argument("--val-num", "--val_num", dest="val_count", type=int)
    parser.add_argument("--test-num", "--test_num", dest="test_count", type=int)
    parser.add_argument("--rank", dest="hidden_units", type=int)
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int)
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=100, help="0 disables epoch logging.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = json.loads(package_path(args.config).read_text(encoding="utf-8-sig"))
        for name in ("train_count", "val_count", "test_count", "hidden_units", "epochs"):
            value = getattr(args, name)
            if value is not None:
                config[name] = value
        if args.seeds is not None:
            config["train_seeds"] = args.seeds
        config = normalized_training_config(config)
        if args.show_config:
            print(json.dumps({**config, "slices": training_slices(config)}, indent=2))
            return
        train_surrogates(config, data_root=args.data_root, output_root=args.output_root,
                         overwrite=args.overwrite, log_every=args.log_every)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
