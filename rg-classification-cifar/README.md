# Reference-Guided Training — CIFAR-10N Classification

Official implementation of **Reference-Guided Training (RG)** for image classification under real-world human annotation noise on CIFAR-10N, accompanying the paper *Reference-Guided Training: Adaptive Gradient Scaling via Per-Sample Loss Comparisons* (NeurIPS 2026).

RG extends Empirical Risk Minimization (ERM) by comparing the learner's per-sample losses with those of a fixed reference model. These comparisons induce adaptive, bounded gradient amplification without requiring prediction imitation or architectural modifications to the learner.

This repository provides the CIFAR-10N experiments using CNN, MLP, and Vision Transformer learners with ResNet56 reference models.

## Experimental Setup

Three learner architectures are evaluated:

- **CNN** — Convolutional Neural Network
- **MLP** — Multilayer Perceptron
- **ViT** — Vision Transformer (identified as `Transformer` in the code)

All experiments use cross-entropy as the underlying supervised loss. The comparison includes:

| Method | Description |
|---|---|
| ERM | Standard cross-entropy training without a reference model |
| RG | Reference-guided training using per-sample loss comparisons with a fixed ResNet56 guide |

The experiments use three human-annotation configurations from CIFAR-10N:

| Scenario | Learner training labels | ResNet56 guide training labels |
|---|---|---|
| `independent` | `random1` | `random3` |
| `matched` | `worst` | `worst` |
| `cleaner-guide` | `worst` | `aggre` |

The `worst` ERM baseline is shared by the last two scenarios. Across the three architectures, the experiment suite therefore contains **six unique ERM conditions**, **nine RG conditions**, and **three ResNet56 guide configurations**. Six architecture-search conditions cover the two distinct learner label sets.

The first 45,000 CIFAR-10 training samples are used for learner and guide training under the selected annotation set. The remaining 5,000 training samples form a **clean validation set**, and the standard 10,000-image CIFAR-10 test set is reserved for final evaluation. Random cropping and horizontal flipping are applied during training.

## Installation

Python 3.9 and a CUDA-capable GPU are recommended for reproducing the experiments.

```bash
conda create -n rg-cifar python=3.9 -y
conda activate rg-cifar
python -m pip install -r requirements.txt
```

### Dataset

The training pipeline automatically downloads CIFAR-10 images and CIFAR-10N human annotations into `data/` when needed. An internet connection is required on the first run.

The source archive does not redistribute the image dataset or the human-annotation tensors.

## Training

Inspect the complete experiment plan before starting:

```bash
python train_all.py --plan
```

Run the full experiment suite:

```bash
python train_all.py
```

Resume an interrupted execution using the existing compatible outputs:

```bash
python train_all.py --resume
```

`train_all.py` executes the required stages in dependency order: architecture searches, ResNet56 guide training, and the ERM/RG experiment matrix. It rejects existing result directories for a fresh run unless `--resume` is specified. Individual stages additionally check `protocol.json` to prevent mixing incompatible experimental settings.

**Fresh-run requirement:** For `python train_all.py`, `results/` should be absent or contain no experiment directories. The supplied experimental records are stored separately in `original-results/`; they do not need to be removed.

### Training Protocol

- **Stage 1 — Architecture selection:** 20 Optuna trials per learner architecture and label set, with a fixed learning rate of `0.01`, up to 30 epochs per trial, and early-stopping patience of 15 epochs.
- **Stage 2 — Method selection:** 20 Optuna trials per ERM or RG condition, using validation cross-entropy to select hyperparameters. Learning rates are drawn from `{0.001, 0.005, 0.01}`.
- **RG hyperparameters:** `lambda` in `[0.25, 20]`, `delta` in `[1e-4, 1]`, and `gamma` (implemented as `margin`) in `[0.9, 1.1]`.
- **Final training:** five runs (seeds 43–47), each for 150 epochs, using SGD with momentum `0.9`, weight decay `5e-4`, batch size `256`, and cosine learning-rate scheduling.
- **Checkpoint selection:** Optuna trials are selected using clean validation cross-entropy. Final-run checkpoints are selected using the corresponding method-specific validation objective. The code also saves best-run and median-run checkpoints.
- **Learning-rate policy:** RG uses the selected learning rate directly, **without additional learning-rate scaling, mean-one normalization, or warm-up**.

The protocol constants and search spaces are defined in `core/experiment_protocol.py` and documented in `docs/EXPERIMENT_PROTOCOL.md`.

### Individual Experiments

The `train.py` interface supports individual stages and scenarios. For example:

```bash
# Architecture search for a CNN with random1 labels
python train.py --stage architecture --scenario independent --model CNN

# Train all three ResNet56 reference guides
python train.py --stage guide --scenario all

# RG with worst learner labels and an aggre-trained guide
python train.py --stage run --scenario cleaner-guide --model CNN --method rg

# Inspect the complete ERM/RG matrix without training
python train.py --stage matrix --scenario all --model ResNet56 --dry-run
```

For the last command, `--model ResNet56` is a special matrix-selection option that expands to all three **learner** architectures; ResNet56 itself is used only as a reference guide.

### Output Organization

```text
results/
├── CNN_random1_architecture/                  # Architecture search
├── CNN_worst_architecture/                    # Architecture search
├── ResNet56_random3_erm/                      # ResNet56 reference guide
├── ResNet56_worst_erm/                        # ResNet56 reference guide
├── ResNet56_aggre_erm/                        # ResNet56 reference guide
├── CNN_random1_erm/                           # ERM with random1 labels
├── CNN_random1_rg_guide-ResNet56_random3/     # RG: independent scenario
├── CNN_worst_rg_guide-ResNet56_worst/         # RG: matched scenario
└── CNN_worst_rg_guide-ResNet56_aggre/         # RG: cleaner-guide scenario
```

Equivalent condition directories are created for MLP and Transformer learners. Depending on the stage, experiment directories contain Optuna studies (`optuna_study.db`), hyperparameter selections, `protocol.json`, provenance metadata, CSV results, Parquet prediction records, and trained checkpoints (`best-model_r*.pt`, `best-model.pt`, and `median-model.pt`).

## Evaluation

**`eval.py` is the results-consolidation and statistics entry point.** It reads saved experiment CSVs; it does not rerun inference or evaluate missing checkpoints.

### 1. Results Consolidation and Statistics

For newly generated experiments under `results/`:

```bash
python eval.py --stats
```

By default, the consolidated CSV and per-condition statistics are written to:

```text
results/processed_results/
├── experiments_results_summary.csv
└── experiments_detailed_summary.csv
```

To evaluate the **included** records in `original-results/` without retraining:

```bash
python eval.py --stats --results-root original-results --out exports/experiments_results_summary.csv --stats-out exports/experiments_detailed_summary.csv
```

This produces per-run records and condition-level accuracy statistics, including the mean and sample standard deviation across available runs. The `--detailed` flag provides the statistics workflow as an alternative to `--stats`.

## Project Structure

```text
rg-classification-cifar-repo/
├── core/                       # RG loss, dataset processing, experiment planning, and training
│   ├── data/                  # CIFAR-10N dataset and guide-prediction handling
│   └── train/                 # Shared training components and loss functions
├── models/                     # CNN, MLP, ViT, and ResNet56 implementations
├── data/                       # CIFAR-10 and CIFAR-10N download location
├── original-results/           # Included Optuna studies and experimental CSV/Parquet records
├── results/                    # Outputs from new experiment executions
├── docs/                       # Experimental protocol and CLI architecture
├── train.py                    # Individual stages and experiment scenarios
├── train_all.py                # Complete dependency-ordered experiment runner
├── eval.py                     # CSV consolidation and statistics
├── make_zip.py                 # Compact source and small-results packaging
├── requirements.txt           # Python dependencies
└── README.md                   # Project documentation
```

`models/resnet20.py` is retained as an implementation dependency of ResNet56; ResNet20 is not a separately evaluated guide in this experiment suite.

## Reproducibility Notes

- **Training objective:** cross-entropy for ERM; RG uses reference-relative per-sample cross-entropy comparisons.
- **Human annotation noise:** CIFAR-10N `random1`, `random3`, `worst`, and `aggre` labels, with each set used according to the scenario table.
- **Validation:** clean CIFAR-10 labels determine trial selection; the test set is used only for final evaluation.
- **Optimization:** SGD with momentum and cosine learning-rate scheduling; RG uses no additional LR scaling.
- **Search and evaluation:** architecture selection and method-level hyperparameter selection are separate stages; final results are based on five seeds.
- **Result provenance:** `protocol.json`, `dependencies.json`, and selected-hyperparameter records guard against incompatible reruns.
- **Experimental records:** `original-results/` contains the experiments CSVs, Optuna databases, protocol files, and small Parquet files.

## Licenses and External Datasets

- **Project source code:** MIT License - see [LICENSE.txt](LICENSE.txt).
- **CIFAR-10N human annotations:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) - see [docs/CIFAR_10N_LICENSE.txt](docs/CIFAR_10N_LICENSE.txt) and the [CIFAR-10N source repository](https://github.com/UCSC-REAL/cifar-10-100n).
- **CIFAR-10 image dataset:** downloaded separately from the [official CIFAR datasets](https://www.cs.toronto.edu/~kriz/cifar.html). This dataset is not redistributed here.
- **Hugging Face Transformers:** Apache License 2.0 - see [docs/TRANSFORMERS_LIB_LICENSE.txt](docs/TRANSFORMERS_LIB_LICENSE.txt).

## Citation

If you use this implementation, please cite:

[Reference-Guided Training: Adaptive Gradient Scaling via Per-Sample Loss Comparisons.](https://openreview.net/forum?id=y7zQMhy4Gr), NeurIPS 2026.

For the other experimental domains, see the Fashion-MNIST classification and M4 regression implementations.