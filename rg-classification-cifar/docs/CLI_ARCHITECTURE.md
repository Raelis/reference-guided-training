# Experiment implementation map

| Module | Responsibility |
| --- | --- |
| `train.py` | Public CLI and stage dispatch |
| `train_all.py` | Run all paper experiments in order |
| `core/experiment_plan.py` | Scenario planning and protocol manifests |
| `core/experiment_paths.py` | Canonical path naming |
| `core/experiment_protocol.py` | Experimental constants and search spaces |
| `core/experiment_method.py` | ERM/RG parameter definitions and loss construction |
| `core/experiment_runner.py` | Experiment object and dataset dispatch |
| `core/experiment_studies.py` | Architecture and method Optuna searches |
| `core/experiment_fit.py` | Final runs, checkpoints, resume and provenance |
| `core/data/dataset_builder.py` | CIFAR-10N split, augmentation and guide logits |
| `core/train/trainer.py` | Shared numerical train/validation loop |
| `core/train/loss_factory.py` | Classification CE and reference-guided objective |
| `core/evaluation_metrics.py` | Final test metrics |
| `core/evaluation_summary.py` | Aggregate results for export |

Output directories follow Fashion-MNIST style: `<model>_<labels>_<method>`
with `_guide-ResNet56_<guide-labels>` appended for RG. Architecture studies
are placed in `<model>_<labels>_architecture`.