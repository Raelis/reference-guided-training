# Reference-Guided Training (RG)

**Official implementation of Reference-Guided Training (RG): Adaptive Gradient Scaling via Per-Sample Loss Comparisons.**

**Accepted at NeurIPS 2026.**

**Paper:** [Reference-Guided Training: Adaptive Gradient Scaling via Per-Sample Loss Comparisons (NeurIPS 2026)](https://openreview.net/pdf?id=y7zQMhy4Gr)

## Overview

Reference-Guided Training (RG) is a training method that adaptively scales per-sample gradients by comparing the loss of a model being trained against reference losses derived from a fixed guide model.

Instead of weighting all training samples equally, RG uses these loss comparisons to identify samples for which the current model performs worse than the reference. Their gradient contributions are amplified through a smooth, bounded mechanism.

RG integrates into standard supervised learning objectives without requiring architectural modifications or access to the reference model during training. Only its predictions or precomputed per-sample losses are needed.

### Main Features

- **Reference-guided optimization:** Uses per-sample loss comparisons to guide training.
- **Adaptive gradient scaling:** Dynamically adjusts gradient contributions based on the relative performance of the learner and reference.
- **Bounded amplification:** Gradient scaling factors remain between 1 and \(1+\lambda\).
- **Loss-function flexibility:** Supports any differentiable loss function that provides per-sample losses, including classification and regression objectives.
- **Reference-model independence:** Requires only reference predictions or precomputed per-sample losses, without access to the reference model itself during training.
- **Architecture independence:** Can be applied to different model architectures without modifying their internal structure.

## Method

For each training sample, RG compares the current model's loss with a reference loss obtained from fixed guide predictions.

When the current loss exceeds a specified threshold relative to the reference loss, RG smoothly increases that sample's contribution to the optimization.

Unlike hard sample selection methods, RG retains all training samples and adjusts their gradient contributions continuously.

The reference model does not need to be loaded or executed during training, provided that its predictions or the corresponding per-sample reference losses are available.

For the complete mathematical formulation, theoretical analysis, and experimental evaluation, please refer to the [NeurIPS 2026 paper](https://openreview.net/pdf?id=y7zQMhy4Gr).

## Experiments

The NeurIPS 2026 paper evaluates RG across classification and regression tasks:

| Dataset | Task | Setting |
|---|---|---|
| Fashion-MNIST | Image classification | Synthetic label noise |
| CIFAR-10N | Image classification | Real-world human annotation noise |
| M4 | Time-series forecasting | Regression |

The experiments investigate the effectiveness of reference-guided optimization across different architectures, data conditions, and reference-model configurations.

## Reproducibility

This GitHub repository serves as the main development repository for Reference-Guided Training. The implementation may evolve through improvements, bug fixes, and additional experiments.

The exact experimental version associated with the NeurIPS 2026 publication will be preserved separately on Zenodo.

**Zenodo archive:** To be added.

For reproducing the results reported in the paper, use the archived experimental version once available.

## Citation

If you use Reference-Guided Training (RG) or build upon this work in your research, please cite our NeurIPS 2026 paper:

**[Reference-Guided Training: Adaptive Gradient Scaling via Per-Sample Loss Comparisons](https://openreview.net/pdf?id=y7zQMhy4Gr)**

Israel Rodrigues Soares, Pedro Benedetti, Thanda Shwe, Israel Mendonça dos Santos, and Masayoshi Aritsugi.

*Advances in Neural Information Processing Systems (NeurIPS), 2026.*

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

Third-party datasets and other external assets remain subject to their respective licenses.
