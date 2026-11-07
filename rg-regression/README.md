# Reference-Guided Training — M4 Regression

Official implementation of **Reference-Guided Training (RG)** for time-series forecasting on the M4 benchmark, accompanying the paper *Reference-Guided Training: Adaptive Gradient Scaling via Per-Sample Loss Comparisons* (NeurIPS 2026).

RG extends Empirical Risk Minimization (ERM) by comparing the learner's per-sample losses with those of a fixed reference model. These comparisons induce adaptive, bounded gradient amplification without requiring prediction imitation or architectural modifications to the learner.

This repository evaluates RG on the M4 forecasting benchmark using neural forecasting models and statistical reference methods.

## Experimental Setup

The experiments cover three neural architectures:

- **MLP** — Multilayer Perceptron
- **GRU** — Gated Recurrent Unit
- **TCN** — Temporal Convolutional Network

All models use Mean Squared Error (MSE) as the underlying supervised loss.

The training configurations include:

| Method | Description |
|---|---|
| ERM | Standard MSE-based training without a reference model |
| RG + NAR-AOLS | Reference-guided training using a polynomial nonlinear autoregressive guide fitted with Accelerated Orthogonal Least Squares |
| RG + Seasonal Naive | Reference-guided training using seasonal naive forecasts |

The implementation uses `NARMAX` and `seasonal-naive` as the corresponding internal guide identifiers.

Experiments cover all six M4 frequency subsets: Yearly, Quarterly, Monthly, Weekly, Daily, and Hourly.

## Installation

Python 3.9 and a CUDA-capable GPU are recommended for reproducing the original experiments.

```bash
conda create -n rg-regression python=3.9 -y
conda activate rg-regression
python -m pip install -r requirements.txt
```

### Dataset

The experiments use the M4 dataset, consisting of 100,000 time series across six frequency subsets.

The dataset must be available under `data/M4/`, as expected by the `datasetsforecast` data loader and the evaluation scripts.

Dataset files are not included in the compact source archive.

## Training

Run the complete experiment suite from the project root:

```bash
python train.py
```

The configurations are defined in `experiments_stack` in `train.py`, using the tuple:

```python
(model_name, guide_method, m4_subset)
```

The supported values are:

- `model_name`: `MLP`, `GRU`, `TCN`
- `guide_method`: `no`, `NARMAX`, `seasonal-naive`
- `m4_subset`: `Yearly`, `Quarterly`, `Monthly`, `Weekly`, `Daily`, `Hourly`

For `guide_method="no"`, the script executes the ERM architecture search and baseline training. For guided configurations, it executes RG training with the selected statistical reference method.

The experiment runner checks for existing completed checkpoints and skips corresponding training stages. To rerun a completed configuration, back up or rename its experiment directory before execution.

### Output Organization

Experiment artifacts are stored under `results/` using the following naming conventions:

```text
results/
├── MLP_Yearly_mse/                         # ERM architecture search
├── MLP_Yearly_mse_baseline/                # Additional ERM baseline training
├── MLP_Yearly_baseline-mse-NARMAX/         # RG with NAR-AOLS
├── MLP_Yearly_baseline-mse-seasonal-naive/ # RG with Seasonal Naive
└── Baselines/                              # Cached statistical reference forecasts
```

Depending on the training stage, experiment directories contain Optuna studies, training records, validation-selected checkpoints, and prediction artifacts.

## Evaluation

Two evaluation scripts serve different purposes.

### 1. Detailed Evaluation

```bash
python eval-detailed.py
```

This is the **primary evaluation script for the regression results reported in the paper**.

It loads saved checkpoints for ERM and RG with the NAR-AOLS guide, performs inference, and computes:

- M4 forecasting metrics, including sMAPE, MASE, and OWA
- ERM–guide and RG–guide error-ranking correlations
- Relative OWA improvements
- Aggregated metrics for the M4 frequency groups
- Results tables and the error-ranking alignment figure

The `Others` group combines Weekly, Daily, and Hourly. Its OWA is computed by aggregating sMAPE and MASE using the corresponding numbers of series before normalization by the aggregated Naive2 benchmark. It is not computed as a direct weighted average of subset-level OWA values.

Outputs include:

```text
processed_results/
├── m4_results.csv
└── guide_alignment.csv
```

**Requirements:** This evaluation needs the M4 dataset, the corresponding experiment records, saved model checkpoints (`best-model_r*.pt`), and any reference forecasts required by the data loader.

The compact source archive excludes model checkpoints and large prediction caches. These artifacts must be restored or regenerated to rerun the complete evaluation.

### 2. Results Consolidation

```bash
python eval.py
```

This script collects `experiment-results.csv` files from the experiment directories and concatenates them into:

```text
processed_results/results.csv
```

It does **not** independently recalculate OWA, rerun inference, or reproduce the detailed M4 evaluation.

## Project Structure

```text
rg-regression-repo/
├── core/                       # RG objective, dataset processing, training, and statistical guides
├── models/                     # Neural architectures, including MLP, GRU, and TCN
├── data/                       # M4 dataset location (not included in the compact archive)
├── results/                    # Experiment records, Optuna studies, and training artifacts
│   └── Baselines/              # Cached statistical reference forecasts
├── processed_results/          # Consolidated and aggregated evaluation results
├── train.py                    # Main experiment runner
├── eval.py                     # CSV results consolidation
├── eval-detailed.py            # Paper-level M4 evaluation
├── make_zip.py                 # Compact source and small-results packaging
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Reproducibility Notes

- **Training loss:** Mean Squared Error (MSE).
- **Random seed:** The main experiment runner sets `SEED = 42`.
- **Model selection:** Training includes Optuna-based hyperparameter search and validation-based checkpoint selection.
- **Optimization:** Neural forecasting models use AdamW and learning-rate scheduling.
- **Early stopping:** Training uses validation-based early stopping.
- **Statistical guides:** Implemented in `core/workers.py`.
- **Evaluation:** The detailed evaluator implements the M4 metric aggregation used for the reported results.
- **Hardware:** The experiments were developed for GPU-accelerated training with PyTorch.

The compact repository distribution retains source code and small experimental artifacts but omits the M4 dataset, model checkpoints, and large prediction arrays. Consequently, reproducing the full evaluation requires regenerating or separately obtaining these artifacts.

## Licenses and External Data

- **Project source code:** MIT License - see [LICENSE.txt](LICENSE.txt).
- **M4 dataset:** obtained separately from the [official M4 competition resources](https://github.com/Mcompetitions/M4-methods) through the data-loading pipeline. The dataset is not bundled with this repository and its use remains subject to its source's terms.
- **Python dependencies:** installed separately from [requirements.txt](requirements.txt) and subject to their respective upstream licenses.

## Citation

If you use this implementation, please cite:

[Reference-Guided Training: Adaptive Gradient Scaling via Per-Sample Loss Comparisons.](https://openreview.net/forum?id=y7zQMhy4Gr), NeurIPS 2026.

For the other experimental domains, see the Fashion-MNIST and CIFAR-10N implementations in the main project repository.