# MentorNet-PD network, predefined-curriculum data and mentor pretraining.
from pathlib import Path
import json
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


MENTOR_CHECKPOINT = Path('results') / 'mentornet-pd-pretraining' / 'mentornet.pt'
PRETRAIN_SAMPLES = 100_000
PRETRAIN_STEPS = 30_000
PRETRAIN_BATCH_SIZE = 32
PRETRAIN_LR = 0.1
PD_MOVING_AVERAGE_DECAY = 0.5
PD_DROPOUT_SCHEDULE = ((0.5, 17), (0.05, 83))


class MentorNetPD(nn.Module):
    # Predict sample weights from loss, loss difference, label and progress.

    def __init__(self):
        super().__init__()
        self.label_embedding = nn.Embedding(2, 2)
        self.epoch_embedding = nn.Embedding(100, 5)
        self.epoch_embedding.weight.requires_grad_(False)
        self.recurrent = nn.LSTM(input_size=2, hidden_size=1, bidirectional=True,
                                 batch_first=True)
        self.hidden = nn.Linear(9, 20)
        self.output_weight = nn.Parameter(torch.empty(20, 1))
        self.output_offset = nn.Parameter(torch.empty(1))
        for name, parameter in self.recurrent.named_parameters():
            if name.startswith('bias_'):
                nn.init.zeros_(parameter)
            else:
                nn.init.xavier_uniform_(parameter)
        nn.init.xavier_uniform_(self.label_embedding.weight)
        nn.init.xavier_uniform_(self.epoch_embedding.weight)
        nn.init.normal_(self.hidden.weight, 0.0, 1.0)
        nn.init.normal_(self.hidden.bias, 0.0, 1.0)
        nn.init.normal_(self.output_weight, 0.0, 1.0)
        nn.init.normal_(self.output_offset, 0.0, 1.0)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 4:
            raise ValueError('MentorNet features must have shape [batch, 4]')
        labels = features[:, 2].long().clamp(0, 1)
        epochs = features[:, 3].long().clamp(0, 99)
        recurrent_input = features[:, :2].unsqueeze(1)
        _, (_, cell_state) = self.recurrent(recurrent_input)
        state = torch.cat((cell_state[0], cell_state[1]), dim=1)
        combined = torch.cat((self.label_embedding(labels),
                              self.epoch_embedding(epochs), state), dim=1)
        return (torch.tanh(self.hidden(combined)) @
                (self.output_weight + self.output_offset)).squeeze(1)


def predefined_pd_targets(features):
    # Reproduce the vstar_mentornet_pd curriculum defined by Jiang et al.
    differences = features[:, 1]
    epochs = features[:, 3]
    binary = (differences <= 0).to(differences.dtype)
    linear = (1.0 - differences).clamp(0.0, 1.0)
    return torch.where(epochs <= 18, torch.ones_like(binary),
                       torch.where(epochs >= 90, linear, binary))


def synthetic_pd_examples(sample_size=PRETRAIN_SAMPLES, seed=42):
    # Sample without replacement from the reference loss/difference/epoch grid.
    if not 1 < sample_size <= 1_200_000:
        raise ValueError('sample_size must be between 2 and 1,200,000')
    losses = np.concatenate((np.arange(0, 10, .1), np.arange(10, 30, 1)))
    differences = np.arange(-5, 5, .1)
    n_diff = len(differences)
    population = len(losses) * n_diff * 100
    rng = np.random.RandomState(seed)
    indices = rng.choice(population, size=sample_size, replace=False)
    epoch = (indices % 100).astype(np.float32)
    diff = differences[(indices // 100) % n_diff].astype(np.float32)
    loss = losses[indices // (n_diff * 100)].astype(np.float32)
    features = torch.from_numpy(np.stack((loss, diff, np.zeros_like(loss), epoch), axis=1))
    targets = predefined_pd_targets(features)
    split = int(sample_size * .8)
    return (features[:split], targets[:split]), (features[split + 1:], targets[split + 1:])


def pretrain_mentornet_pd(*, device='cpu', steps=PRETRAIN_STEPS,
                            sample_size=PRETRAIN_SAMPLES, seed=42,
                            checkpoint=MENTOR_CHECKPOINT):
    # Train MentorNet on predefined PD targets.
    if steps < 1:
        raise ValueError('steps must be positive')
    checkpoint = Path(checkpoint)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    (features, targets), (validation_x, validation_y) = synthetic_pd_examples(sample_size, seed)
    mentor = MentorNetPD().to(device)
    optimizer = torch.optim.Adam(mentor.parameters(), lr=PRETRAIN_LR)
    rng = np.random.RandomState(seed)
    n = len(features)
    order = rng.permutation(n)
    index = 0
    mentor.train()
    for step in range(steps):
        if index + PRETRAIN_BATCH_SIZE > n:
            order = rng.permutation(n)
            index = 0
        batch_indices = order[index:index + PRETRAIN_BATCH_SIZE]
        index += PRETRAIN_BATCH_SIZE
        x = features[batch_indices].to(device)
        target = targets[batch_indices].to(device)
        optimizer.param_groups[0]['lr'] = PRETRAIN_LR * (0.9 ** (step // 1000))
        optimizer.zero_grad()
        loss = F.mse_loss(torch.sigmoid(mentor(x)), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mentor.parameters(), max_norm=5.0)
        optimizer.step()
        if (step + 1) % 1000 == 0 or step + 1 == steps:
            print(f'MentorNet-PD pretraining: step={step+1}/{steps} loss={loss.item():.6f}', flush=True)
    mentor.eval()
    with torch.inference_mode():
        val_sum = 0.0
        for x, target in zip(validation_x.split(1024), validation_y.split(1024)):
            val_sum += F.mse_loss(torch.sigmoid(mentor(x.to(device))),
                target.to(device), reduction='sum').item()
    val_loss = val_sum / len(validation_x)
    if not np.isfinite(val_loss):
        raise FloatingPointError('MentorNet-PD pretraining produced nonfinite validation MSE')
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    data = {'model_state_dict': mentor.cpu().state_dict(),
            'architecture': 'bidirectional-lstm-1-embeddings-2-5-fc-20-shared-offset',
            'training_steps': steps, 'sample_size': sample_size, 'seed': seed,
            'validation_mse': val_loss}
    temporary = checkpoint.with_suffix('.tmp')
    torch.save(data, temporary)
    os.replace(temporary, checkpoint)
    metadata = checkpoint.with_name('training_metadata.json')
    temporary_metadata = metadata.with_suffix('.tmp')
    temporary_metadata.write_text(json.dumps({k: v for k, v in data.items()
        if k != 'model_state_dict'}, indent=2), encoding='utf-8')
    os.replace(temporary_metadata, metadata)
    print(f'MentorNet-PD saved to {checkpoint}; validation MSE={val_loss:.6f}')
    return checkpoint


def load_mentornet_pd(device='cpu', checkpoint=MENTOR_CHECKPOINT):
    # Load the pre-trained MentorNet used by all StudentNet experiments.
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f'Missing MentorNet-PD checkpoint: {path}. '
                                'Run `python train.py --stage mentornet-pretrain`.')
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('training_steps') != PRETRAIN_STEPS or data.get('sample_size') != PRETRAIN_SAMPLES:
        raise ValueError(f'MentorNet-PD checkpoint does not use the complete pretraining protocol: {path}')
    mentor = MentorNetPD()
    mentor.load_state_dict(data['model_state_dict'], strict=True)
    mentor.to(device).eval()
    mentor.requires_grad_(False)
    return mentor


def student_pd_weights(mentor, losses, progress, moving_average, percentile,
                       dropout_rate, *, decay=PD_MOVING_AVERAGE_DECAY):
    # Predict frozen MentorNet weights and apply original random example dropout.
    with torch.no_grad():
        detached = losses.detach().float()
        # TF 1.x contrib.distributions.percentile defaults to nearest interpolation.
        threshold = torch.quantile(detached, float(percentile),
                                   interpolation='nearest').item()
        moving_average = decay * moving_average + (1.0 - decay) * threshold
        features = torch.stack((detached,
                                detached - moving_average,
                                torch.zeros_like(detached),
                                torch.full_like(detached, min(int(progress), 99))), dim=1)
        weights = torch.sigmoid(mentor(features))
        keep = int(len(weights) * (1.0 - dropout_rate))
        if keep < len(weights):
            indices = np.random.choice(len(weights), keep, replace=False)
            mask = torch.zeros_like(weights)
            mask[torch.as_tensor(indices, dtype=torch.long, device=weights.device)] = 1
            weights = weights * mask
    return weights, moving_average
