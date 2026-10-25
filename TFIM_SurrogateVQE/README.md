# QLoFTS Surrogate VQE

Ten-qubit TFIM optimization using a noise-level-1 QLoFTS surrogate as the objective. COBYLA optimizes 19 circuit parameters.

## Files

| File or directory | Purpose |
| --- | --- |
| `01_sampling.py` | Sample parameters and construct level-1 circuits. |
| `02_evaluate_circuits.py` | Prepare energy labels. |
| `03_train_surrogate.py` | Train the QLoFTS surrogate. |
| `04_surrogate_vqe.py` | Optimize circuit parameters using surrogate energies. |
| `surrogate_vqe/` | Sampling, simulation, model, training, and optimization implementations. |
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
python 04_surrogate_vqe.py --model-root outputs/training
```

To use supplied checkpoints directly, run `python 04_surrogate_vqe.py`. Stage 2 replays supplied labels by default; `--mode regenerate` performs fresh noisy simulation. Generated files go to `outputs/`; use `--help` for options.

Each seed uses 5000 rows: train `[0:4000]`, validation `[4000:4500]`, test `[4500:5000]`. Only validation controls learning-rate scheduling; test evaluation follows training. Rank is 128 and the default is 3500 epochs. VQE crosses five model seeds with five initializations; COBYLA uses 2000 evaluations and tolerance `1e-4`.

Reported VQE energies are surrogate predictions. Supplied checkpoints have not been fully retrained under this split and do not establish untouched-test accuracy. Regenerated noisy labels can differ because historical worker RNG states were not recorded.
