"""Checkpoint-based guide memorization and RG weight diagnostics."""
from pathlib import Path
from .experiment_paths import method_dir

def analyze_guide_memorization_condition(noise, device):
    # Compare guide logits against clean and observed training labels.
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision.datasets import FashionMNIST
    from core.dataset_builder import DatasetBuilder

    ds = DatasetBuilder(noise_ratio=noise, baseline_noise_ratio=noise,
                        baseline_method='ResNet20', seed=42, device=device)
    ds.build()
    original = FashionMNIST(root='./data', train=True, download=False)
    indices = np.random.default_rng(42).permutation(len(original))[:54000]
    clean = original.targets[indices].long()
    corrupted = ds.train_dataset.labels.long()
    logits = ds.baseline_train_outputs.float()
    preds = logits.argmax(dim=1)
    changed = clean != corrupted
    observed_corrupted = changed.sum().item()
    n = len(clean)
    loss_noisy = F.cross_entropy(logits, corrupted, reduction='none')
    loss_clean = F.cross_entropy(logits, clean, reduction='none')
    return {
        'noise_ratio': noise, 'train_n': n, 'changed_labels_n': int(observed_corrupted),
        'guide_train_accuracy_on_clean_labels': float((preds == clean).float().mean()),
        'guide_train_accuracy_on_observed_labels': float((preds == corrupted).float().mean()),
        'guide_predicts_corrupted_label_among_changed':
            float((preds[changed] == corrupted[changed]).float().mean()) if observed_corrupted else float('nan'),
        'guide_predicts_original_label_among_changed':
            float((preds[changed] == clean[changed]).float().mean()) if observed_corrupted else float('nan'),
        'guide_ce_observed_among_changed':
            float(loss_noisy[changed].mean()) if observed_corrupted else float('nan'),
        'guide_ce_clean_among_changed':
            float(loss_clean[changed].mean()) if observed_corrupted else float('nan'),
    }


def evaluate_guide_memorization(device=None, *, results_root=Path("results"), analyze=None):
    # Write results/diagnostics/guide_memorization.csv.
    import csv
    import torch

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analyze = analyze or analyze_guide_memorization_condition
    rows = [analyze(noise, device)
            for noise in (0.0, 0.4, 0.8)]
    target = Path(results_root) / 'diagnostics' / 'guide_memorization.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved', target)
    for row in rows:
        print(row)


def analyze_rg_weight_condition(model, noise, device):
    # Compute RG sample-weight statistics using saved checkpoints.
    import numpy as np
    import optuna
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from core.dataset_builder import DatasetBuilder
    from core.experiment_runner import read_architecture, STUDY_NAME
    from core.model_factory import construct

    folder = method_dir(model, noise, 'rg', 'ResNet20', noise)
    db, ckpt_path = folder / 'optuna_study.db', folder / 'best-model.pt'
    study = optuna.load_study(study_name=STUDY_NAME,
        storage=f'sqlite:///{db.resolve()}')
    params = study.best_trial.params
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    student = construct(model, read_architecture(model), device)
    student.load_state_dict(checkpoint['state_dict'], strict=True)
    student.eval()
    ds = DatasetBuilder(noise_ratio=noise, baseline_noise_ratio=noise,
        baseline_method='ResNet20', seed=42, device=device)
    ds.build()
    original = datasets.FashionMNIST(root='./data', train=True, download=False,
        transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))]))
    indices = np.random.default_rng(42).permutation(len(original))[:54000]
    clean_labels = original.targets[indices].long()
    noisy_labels = ds.train_dataset.labels.long()
    guide_logits = ds.baseline_train_outputs.float()
    changed = (clean_labels != noisy_labels).numpy()
    weights = []
    with torch.inference_mode():
        dataset = Subset(original, indices)
        for start, (x, _y) in enumerate(DataLoader(dataset, batch_size=256,
                                                    shuffle=False, num_workers=0)):
            lo = start * 256
            hi = lo + len(x)
            target = noisy_labels[lo:hi].to(device)
            output = student(x.to(device))
            teacher = guide_logits[lo:hi].to(device)
            learner_loss = F.cross_entropy(output, target, reduction='none')
            reference_loss = F.cross_entropy(teacher, target, reduction='none')
            r = learner_loss - float(params['margin']) * reference_loss
            w = 1. + float(params['lamb']) * torch.clamp(
                r / float(params['delta']), min=0.0, max=1.0)
            weights.append(w.cpu().numpy())
    weights = np.concatenate(weights)
    unchanged = ~changed
    return {'model': model, 'noise': noise, 'learner_checkpoint': str(ckpt_path),
            'n_train': len(weights), 'n_changed': int(changed.sum()),
            'best_rg_lr': params['lr'], 'lamb': params['lamb'],
            'delta': params['delta'], 'gamma': params['margin'],
            'mean_w_overall': float(weights.mean()),
            'mean_w_corrupted_labels': float(weights[changed].mean()) if changed.any() else float('nan'),
            'mean_w_unchanged_labels': float(weights[unchanged].mean()),
            'unamplified_share_corrupted': float(np.mean(weights[changed] <= 1+1e-8))
                    if changed.any() else float('nan'),
            'unamplified_share_unchanged': float(np.mean(weights[unchanged] <= 1+1e-8)),
    }


def evaluate_rg_weight_allocation(device=None, *, results_root=Path("results"), analyze=None):
    # Write results/diagnostics/rg_training_weight_allocation.csv.
    import csv
    import torch

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analyze = analyze or analyze_rg_weight_condition
    rows = [analyze(model, noise, device)
            for model in ('CNN', 'MLP', 'Transformer') for noise in (0., .4, .8)]
    out = Path(results_root) / 'diagnostics' / 'rg_training_weight_allocation.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved', out)


