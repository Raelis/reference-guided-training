# Experiment output layout

All experiment paths are constructed by `core/experiment_paths.py`. The
training runner writes checkpoints, studies and metrics; dataset builders
only prepare input tensors and reference logits.

## Examples

| Experiment | Directory under `results/` |
|---|---|
| Stage-1 CNN architecture | `CNN_n0.00_architecture/` |
| ERM, 40% noise | `CNN_n0.40_erm/` |
| RG with a 40% noise guide | `CNN_n0.40_rg_guide-ResNet20_n0.40/` |
| RG with a clean guide | `CNN_n0.40_rg_guide-ResNet20_n0.00/` |
| KD | `CNN_n0.40_kd_guide-ResNet20_n0.40/` |
| MentorNet-PD | `CNN_n0.40_mentornet-pd/` |
| ResNet20 guide | `ResNet20_n0.40_erm/` |
| Cross-fit fold guide | `ResNet20_n0.40_erm-crossfit_fold0/` |
| Cross-fit RG learner | `CNN_n0.40_rg-crossfit_guide-ResNet20_n0.40/` |
| Ablation 7 | `CNN_n0.40_rg-ablation7_guide-MLP-guide_n0.00/` |

A condition directory contains `optuna_study.db` when tuning is applicable,
`experiment-results.csv`, `experiment_data.parquet`, and the final run
checkpoints `best-model_rN.pt`. The best and median aliases are also saved.
CSV rows include `run_id`, `seed`, the optimizer `lr`, applicable RG
`lambda`/`delta`/`gamma`, source `selected_trial`, and one-based `best_epoch`.
MentorNet-PD rows also record the selected loss percentile, burn-in percentage,
and moving-average decay, while its Optuna study uses a distinct study name
inside the same `optuna_study.db` file.

For cross-fitting, `core/crossfit_data.py` keeps fold assignments and logits
in memory. It does not create separate fold-logit files or extra directories 
under `results/`.

## Evaluation modules

`eval.py` is the sole evaluation CLI and owns all argument parsing.
`core/evaluation_summary.py` aggregates CSV and Optuna summaries;
`core/evaluation_diagnostics.py` computes guide and RG diagnostics;
`core/evaluation_distributions.py` loads saved checkpoints to calculate
per-sample CE distributions and produce plots. The three modules do not
parse command-line arguments.

MentorNet-PD pretraining creates `results/mentornet-pd-pretraining/mentornet.pt`
and `training_metadata.json`. The files are required for every MentorNet-PD
StudentNet condition but not for ERM, RG, KD, or SPL.
