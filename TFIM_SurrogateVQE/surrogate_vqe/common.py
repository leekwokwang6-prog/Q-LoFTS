"""File utilities shared by sampling and circuit evaluation."""

import hashlib
import json
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path=None):
    path = ROOT / "config" / "experiment.json" if path is None else resolve_path(path)
    with path.open(encoding="utf-8") as handle:
        settings = json.load(handle)
    if settings["noise_level"] != 1:
        raise ValueError("This workflow supports noise level 1 only")
    return settings


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_pickle(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def guard_output(path, overwrite=False):
    path = resolve_path(path)
    if ROOT not in path.parents:
        raise ValueError("Output must be a subdirectory of this submission folder")
    relative = path.relative_to(ROOT)
    if relative.parts[0].lower() in {"frozen", "surrogate_vqe", "config"}:
        raise ValueError("Output cannot replace packaged sources or frozen assets")
    if path.exists() and (not path.is_dir() or any(path.iterdir())) and not overwrite:
        raise FileExistsError(f"Output is not empty: {path}. Choose a new path or use --overwrite.")
    return path
