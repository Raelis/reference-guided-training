# CIFAR-10N experimental protocol

## Scenarios

| Scenario | Learner labels | Guide labels |
| --- | --- | --- |
| independent | random1 | random3 |
| matched | worst | worst |
| cleaner-guide | worst | aggre |

Each learner (CNN, MLP, Transformer/ViT) is evaluated with ERM and RG. The ERM
condition for `worst` is shared between the last two scenarios. This gives six
unique ERM and nine RG conditions, in addition to six learner architecture
searches and three ResNet56 guide conditions.

## Split and preprocessing

The first 45,000 CIFAR-10 training examples use the selected human noise label
set. The final 5,000 training examples supply clean validation labels for both
hyperparameter and checkpoint selection. The held-out CIFAR-10 test split is
used only for final evaluation. Random crop and random horizontal flip are
applied to learner training. Reference logits are inferred once using the
non-augmented, normalized images and are aligned by original image index.

## Training and selection

- Architecture search: 20 Optuna trials, 30 epochs each, SGD LR 0.01, early
  stopping patience 15, clean validation cross-entropy.
- ResNet56 guide models: separate runs for random3, worst and aggre.
- ERM and RG tuning: 20 trials per condition, 150 epochs, trial selection by
  minimum clean validation cross-entropy.
- Learning rate is categorical: 0.001, 0.005, 0.01. No LR scaling, mean-one
  normalization, or warm-up is applied to RG.
- RG tuning additionally searches lamb in [0.25, 20] (log), delta in
  [1e-4, 1] (log), margin in [0.9, 1.1] (uniform).
- Final fitting: 5 seeds, 150 epochs, SGD momentum 0.9, weight decay 5e-4,
  fixed batch size 256, cosine scheduler.
- Final checkpoints use the minimum method-specific validation objective.
  The median checkpoint is selected across five final runs by that criterion.
