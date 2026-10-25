"""Run noise-level-one surrogate VQE with the original initialization and COBYLA."""

import argparse
import json
from pathlib import Path

from surrogate_vqe.vqe import resolve_path, run_vqe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config" / "experiment.json")
    parser.add_argument("--checkpoint-seeds", "--seeds", type=int, nargs="+", help="Checkpoint/model seeds; default: the configured five seeds")
    parser.add_argument("--parameter-seeds", type=int, nargs="+", help="Circuit initialization seeds; default: the configured five seeds")
    parser.add_argument("--max-iteration", type=int, help="COBYLA function-evaluation budget (original default: 2000)")
    parser.add_argument("--model-root", type=Path, help="Root containing training_1/run_N/best_model.pth")
    parser.add_argument("--sample-root", type=Path, help="Root containing training_1/run_N/samples.pkl")
    parser.add_argument("--output-root", type=Path, help="Directory for optimization outputs")
    parser.add_argument("--show-config", action="store_true", help="Print the effective VQE settings and exit")
    parser.add_argument("--overwrite", action="store_true", help="Replace output files for the selected seed combinations")
    args = parser.parse_args()
    try:
        config = json.loads(resolve_path(args.config).read_text(encoding="utf-8-sig"))
        if args.show_config:
            settings = dict(config["vqe"])
            for key in ("checkpoint_seeds", "parameter_seeds", "max_iteration", "model_root", "sample_root", "output_root"):
                value = getattr(args, key)
                if value is not None:
                    settings[key] = str(value) if isinstance(value, Path) else value
            print(json.dumps(settings, indent=2))
            return
        run_vqe(
            config,
            checkpoint_seeds=args.checkpoint_seeds,
            parameter_seeds=args.parameter_seeds,
            max_iteration=args.max_iteration,
            model_root=args.model_root,
            sample_root=args.sample_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
