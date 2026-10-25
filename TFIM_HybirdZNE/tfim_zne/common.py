import hashlib
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else ROOT / "config" / "experiment.json"
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_pickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def save_pickle(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(data, stream)


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def output_guard(root, overwrite=False):
    root = Path(root).resolve()
    if ROOT not in root.parents:
        raise ValueError("Outputs must stay inside the new submission folder.")
    protected = [ROOT / "frozen", ROOT / "config", ROOT / "tfim_zne"]
    if any(root == p or p in root.parents for p in protected):
        raise ValueError("Outputs must not overwrite frozen inputs, configuration, or source modules.")
    if root.exists() and not root.is_dir():
        raise FileExistsError(f"Output is a file: {root}")
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"Output exists: {root}. Choose another directory or use --overwrite.")
    root.mkdir(parents=True, exist_ok=True)
    return root
