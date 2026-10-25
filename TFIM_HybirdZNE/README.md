# QLoFTS Hybrid VQE-ZNE

Ten-qubit TFIM optimization: direct circuit energies at folds 1/3 and surrogate predictions at folds 5/7/9, followed by quadratic extrapolation.

## Files

| File or directory | Purpose |
| --- | --- |
| `01_sampling.py` | Sample parameters and construct folded circuits. |
| `02_evaluate_circuits.py` | Prepare energy labels. |
| `03_train_surrogate.py` | Train the QLoFTS surrogate models. |
| `04_vqe_zne.py` | Optimize the VQE-ZNE objective with COBYLA. |
| `tfim_zne/` | Circuit, noise, model, training, and optimization implementations. |
| `config/experiment.json` | Seeds, data splits, hyperparameters, noise, and paths. |
| `frozen/data/`, `frozen/models/` | Supplied datasets and checkpoints. |

## Run

Run inside this directory:

```powershell
conda env create -f environment.yml
conda activate qlofts
python 01_sampling.py
python 02_evaluate_circuits.py
python 03_train_surrogate.py --data-root outputs/evaluation --seeds 0 1 2 3 4
python 04_vqe_zne.py --model-root outputs/training
```

To use supplied checkpoints, run stages 1 and 4, omitting `--model-root outputs/training`. Stage 2 replays supplied labels by default; `--mode regenerate` performs fresh noisy simulation. Generated files go to `outputs/`; use `--help` for options.

Each seed uses 5000 rows: train `[0:4000]`, validation `[4000:4500]`, test `[4500:5000]`. Only validation controls learning-rate scheduling; test evaluation follows training. Rank is 128; default epochs for folds 5/7/9 are 1600/1800/1200. COBYLA uses 2000 evaluations and tolerance `1e-4`.

Supplied checkpoints have not been fully retrained under this split and do not establish untouched-test accuracy. Regenerated noisy labels can differ because historical worker RNG states were not recorded.
