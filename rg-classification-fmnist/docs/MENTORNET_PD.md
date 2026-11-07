# MentorNet-PD source and implementation

The code in `models/mentornet_pd.py` is a PyTorch adaptation of the MentorNet
implementation supplied in `mentornet-master.zip` (Google Research, 2018),
licensed under Apache License 2.0. Reference:

Jiang et al. (2018), *MentorNet: Learning Data-Driven Curriculum for Very Deep
Neural Networks on Corrupted Labels*, ICML, https://proceedings.mlr.press/v80/jiang18c.html

The implementation follows `code/training_mentornet/models.py` for the
predefined curriculum, `code/training_mentornet/data_generator.py` for the
feature grid, `code/training_mentornet/train.py` for the 30,000-step learning
schedule, `code/utils.py` for the MentorNet architecture and sample selection,
and the MentorNet-PD commands in `commands/train_studentnet_resnet.sh`.

A learned MentorNet is pretrained once from 100,000 synthetic examples and
frozen during StudentNet learning. 

The StudentNet loss percentile uses nearest-observation interpolation, matching
`tf.contrib.distributions.percentile` in the original TensorFlow 1.x code,
rather than PyTorch`s default linear interpolation.

The synthetic target set contains continuous weights, so the official reader
classifies it as non-binary. The mentor pretraining objective is therefore mean
squared error on sigmoid probabilities, not binary cross-entropy.

## StudentNet hyperparameter search

The pretrained MentorNet is fixed, but the StudentNet curriculum has three
validation-selected parameters, in addition to LR:

- `mnpd_loss_p_percentile`: continuous range 0.50 to 0.95; quantile for the
  exponential moving-average threshold.
- `mnpd_burn_in_pct`: integer 0 to 40; percentage of initial optimization steps
  trained with uniform per-sample weights and no sample dropout.
- `mnpd_moving_average_decay`: continuous range 0.10 to 0.90; EMA retention
  coefficient for the loss-percentile threshold.

Twenty Optuna trials use the same StudentNet LR grid as the other methods,
maximum 300 epochs, patience 30 monitoring validation cross-entropy, and
selection by best validation cross-entropy. Each failed/interrupted trial is
restarted from epoch one with its original parameters and seed. The final fits
use five seeds and all 300 epochs, without early stopping. StudentNet PD
searches use their own named Optuna study inside the regular `optuna_study.db`
so an older LR-only study cannot silently replace the tuned curriculum.
The experiment CSV and checkpoint parameters record all three values.

Burn-in does not change the frozen network or the PD pretraining procedure;
it only temporarily sets StudentNet example weights to one. The PD-specific
example-dropout schedule remains 50% for the first 17% of the optimization
steps and 5% for the remainder of training. This is MentorNet-PD, not
MentorNet-DD.
