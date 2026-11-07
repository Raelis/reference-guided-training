# Reference-Guided Training — Fashion-MNIST Classification

Official implementation of **Reference-Guided Training (RG)** for image classification under synthetic label noise on Fashion-MNIST, accompanying the paper *Reference-Guided Training: Adaptive Gradient Scaling via Per-Sample Loss Comparisons* (NeurIPS 2026).

RG extends Empirical Risk Minimization (ERM) by comparing the learner's per-sample losses with those of a fixed reference model. These comparisons induce adaptive, bounded gradient amplification without requiring prediction imitation or architectural modifications to the learner.

This repository provides the Fashion-MNIST experiments, optimization controls, guide-quality studies, cross-fitting experiments, and diagnostics reported in the paper.

## Experimental Setup

Three learner architectures are evaluated:

- **CNN** — Convolutional Neural Network
- **MLP** — Multilayer Perceptron
- **ViT** — Vision Transformer (identified as `Transformer` in the code)

Training uses cross-entropy loss with symmetric label corruption applied only to the training labels at **0%, 40%, and 80%**. The clean validation and test sets contain 6,000 and 10,000 samples, respectively; the training split contains 54,000 samples. Fixed noise caches preserve the same corruption patterns across methods, with the 40% corrupted-sample indices contained within the 80% set.

The main comparison includes:

| Method | Description |
|---|---|
| ERM | Standard cross-entropy training |
| RG | Reference-guided training with per-sample loss comparisons |
| KD | Knowledge Distillation using a ResNet20 guide |
| SPL | Self-Paced Learning |
| MentorNet-PD | Sample weighting using a separately pretrained, frozen MentorNet |

For the main RG and KD comparisons, the **ResNet20** guide is trained under the same label-noise condition as the learner. A separate **MLP guide** is used in a guide-quality ablation.

Additional experiments examine ERM at the RG-selected learning rate, mean-one normalization of RG weights, guide quality under different noise levels, two-fold cross-fitting, and guide memorization and weight-allocation diagnostics.

## Installation

Python 3.9 and a CUDA-capable GPU are recommended for reproducing the original experiments.

```bash
conda create -n rg-fmnist python=3.9 -y
conda activate rg-fmnist
python -m pip install -r requirements.txt
```

### Dataset

Fashion-MNIST is downloaded automatically by the training data loader into `data/` when needed. The source archive does not include the image dataset, but it includes the small deterministic label caches under `noise_cache/`.

## Training

Inspect the complete experiment plan before starting:

```bash
python train_all.py --plan
```

Run the full experiment suite:

```bash
python train_all.py
```

Resume after interrupted or failed steps:

```bash
python train_all.py --resume
```

`train_all.py` coordinates architecture selection, reference-model training, the main comparisons, controls, ablations, cross-fitting, and evaluation steps. It records the status of each step and its dependencies, allowing independent experiments to proceed even if another step fails.

**Fresh-run requirement:** `results/` must be empty or absent before running `python train_all.py` without `--resume`. The supplied experimental records are kept separately in `original-results/` and do not need to be deleted.

### Training Protocol

- **Stage 1 — Architecture selection:** 20 Optuna trials per learner architecture, up to 30 epochs per trial, using clean labels and a fixed learning rate of `0.005`.
- **Stage 2 — Method selection:** learning rates selected from `{0.01, 0.005, 0.001}` using validation cross-entropy. ERM evaluates the three distinct candidates; RG, KD, SPL, and MentorNet-PD use up to 20 Optuna trials per experimental condition.
- **Final training:** five runs (seeds 43–47), each completing 300 epochs with SGD (momentum `0.9`, weight decay `5e-4`, batch size `256`) and cosine learning-rate scheduling.
- **Checkpoint selection:** the search selects hyperparameters using validation cross-entropy; final checkpoints are selected using the validation objective associated with each method.
- **Reference models:** ResNet20 and the MLP guide are trained separately at a fixed learning rate of `0.005`.

The complete runner uses the standard learning-rate grid **without additional RG learning-rate scaling**. Advanced individual conditions are available through `train.py`; the complete predefined experiment plan should be run using `train_all.py`.

### Output Organization

```text
results/
├── experiment_manifest.json                   # Step status and execution configuration
├── execution_logs/                            # Per-step logs
├── CNN_n0.00_architecture/                    # Architecture-search Optuna study
├── ResNet20_n0.40_erm/                        # Reference-model training
├── CNN_n0.40_erm/                             # ERM at 40% label noise
├── CNN_n0.40_rg_guide-ResNet20_n0.40/         # RG with matched-noise guide
├── CNN_n0.40_kd_guide-ResNet20_n0.40/         # Knowledge Distillation
└── processed_results/                         # Consolidated evaluation outputs
```

Depending on the experiment, condition directories contain `optuna_study.db`, `experiment-results.csv`, `experiment_data.parquet`, and model checkpoints (`best-model_r*.pt`).

## Evaluation

**`eval.py` is the single evaluation entry point.** It consolidates training records and provides statistics, checkpoint-based analyses, diagnostics, and paper figures.

### 1. Results Consolidation and Statistics

For newly generated experiments in `results/`:

```bash
python eval.py --stats
```

To summarize the **included** CSV records in `original-results/` without starting a new training run:

```bash
python eval.py --stats --results-root original-results --out exports/experiments_results_summary.csv --stats-out exports/experiments_detailed_summary.csv
```

These commands consolidate the per-run CSVs and calculate per-condition statistics; they do not rerun inference.

### 2. Diagnostic Analyses

After generating or restoring the required model checkpoints:

```bash
python eval.py --guide-memorization
python eval.py --rg-weight-allocation
python eval.py --selected-hyperparameters
```

The guide-memorization and RG weight-allocation analyses write outputs under `results/diagnostics/`. Selected Optuna parameters are exported to `results/processed_results/`.

### 3. Error Distributions and Paper Figures

For checkpoint-based per-sample error distributions:

```bash
python eval.py --detailed --architecture Transformer --noise-ratio 0.4 --device cuda
```

To regenerate the CNN noise and guide-quality sweep figures from the included CSV records:

```bash
python eval.py --paper-figures --results-root original-results --results-dir original-results --out exports/experiments_results_summary.csv
```

The checkpoint-based analyses require model checkpoints. The CNN sweep figures can be generated from the included final-run CSV records without retraining.

## Project Structure

```text
.
├── core/                       # RG objective, data processing, training, and evaluation modules
│   └── experiment_scenarios/   # Guide-quality, cross-fitting, and ablation definitions
├── models/                     # CNN, MLP, ViT, ResNet20, MLP guide, and MentorNet-PD
├── data/                       # Fashion-MNIST download location
├── noise_cache/                # Deterministic label-corruption caches
├── original-results/           # Included Optuna studies and per-run experimental records
├── results/                    # Outputs from new executions
├── figures/                    # Saved paper figures and figure-level statistics
├── docs/                       # CLI, output layout, and MentorNet-PD documentation
├── train.py                    # Individual training-stage and scenario CLI
├── train_all.py                # Full experiment orchestration and resume support
├── eval.py                     # Results consolidation, diagnostics, and figures
├── make_zip.py                 # Compact source and small-results packaging
├── requirements.txt           # Python dependencies
└── README.md                   # Project documentation
```

## Reproducibility Notes

- **Training loss:** cross-entropy for classification. RG uses reference-relative per-sample loss comparisons.
- **Noise:** deterministic symmetric corruption applied exclusively to the training labels.
- **Search:** architecture selection and method-level hyperparameter selection are separate stages.
- **Final evaluation:** five runs per condition, with mean and sample standard deviation reported in the paper.
- **Model selection:** validation cross-entropy for hyperparameter search. Method-specific validation objectives for final checkpoints.
- **Optimization:** SGD with momentum and cosine learning-rate scheduling.
- **Experimental records:** the original CSVs, Parquet metadata, Optuna databases, and figures under `original-results/` and `figures/`.

The full Fashion-MNIST image dataset is not included. The training loader downloads it automatically on first use when it is missing (internet access required).

## Licenses and Third-Party Assets

- **Project source code:** MIT License - see [LICENSE.txt](LICENSE.txt).
- **Fashion-MNIST dataset and bundled label caches:** the original Zalando MIT notice is preserved in [docs/FASHION_MNIST_LICENSE.txt](docs/FASHION_MNIST_LICENSE.txt). Dataset source: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).
- **MentorNet-PD implementation:** adapted from [google/mentornet](https://github.com/google/mentornet). The original Apache License 2.0 notice is preserved in [docs/MENTORNET_LICENSE.txt](docs/MENTORNET_LICENSE.txt).
- **Hugging Face Transformers:** Apache License 2.0; see [docs/TRANSFORMERS_LIB_LICENSE.txt](docs/TRANSFORMERS_LIB_LICENSE.txt).

## Citation

If you use this implementation, please cite:

[Reference-Guided Training: Adaptive Gradient Scaling via Per-Sample Loss Comparisons.](https://openreview.net/forum?id=y7zQMhy4Gr), NeurIPS 2026.

For the other experimental domains, see the CIFAR-10N classification and M4 regression implementations.
