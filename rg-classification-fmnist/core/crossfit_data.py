import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from core.dataset_builder import DatasetBuilder, load_model


def crossfit_folds(n, seed=42, k=2):
    rng = np.random.default_rng(seed + 10_000) 
    return rng.permutation(n) % k


def assemble_out_of_fold(outputs_per_copy, folds):
    k = len(outputs_per_copy)
    assert k == 2, "only admits k=2"
    out = torch.empty_like(outputs_per_copy[0])
    for f in range(k):
        mask = torch.from_numpy(folds == f)
        out[mask] = outputs_per_copy[1 - f][mask]
    return out


class CrossFitDatasetBuilder(DatasetBuilder):

    def __init__(self, guide_ckpt_paths, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guide_ckpt_paths = guide_ckpt_paths

    def generate_baseline_predictions(self, X_train, X_val, X_test, y_test=None, y_val=None):
        k = len(self.guide_ckpt_paths)
        batch_size = 256

        models = [
            load_model(self.MyModel, path, self.device, (1, 28, 28), 10)
            for path in self.guide_ckpt_paths
        ]
        for m in models:
            m.eval()

        def forward_loader(model, X):
            loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
            outs = []
            with torch.no_grad():
                for (xb,) in loader:
                    xb = xb.to(self.device)
                    outs.append(model(xb).cpu())
            return torch.cat(outs, dim=0)

        train_outputs_per_copy = [forward_loader(m, X_train) for m in models]
        val_outputs_per_copy = [forward_loader(m, X_val) for m in models]
        test_outputs_per_copy = [forward_loader(m, X_test) for m in models]

        if y_val is None:
            raise ValueError('Clean validation labels required for crossfit guide check')
        fold_accuracies = {}
        for f, out in enumerate(val_outputs_per_copy):
            probs = torch.softmax(out, dim=1).numpy()
            y_pred = np.argmax(probs, axis=1)
            acc = accuracy_score(y_val.numpy(), y_pred)
            fold_accuracies[f"fold{f}_validation_accuracy"] = acc
            print(f"[Crossfit guide fold{f}] Validation Accuracy: {acc:.4f}")

        # Use guide predictions from the complementary training fold.
        fold_train = crossfit_folds(X_train.shape[0], seed=self.seed, k=k)
        fold_val = crossfit_folds(X_val.shape[0], seed=self.seed, k=k)
        fold_test = crossfit_folds(X_test.shape[0], seed=self.seed, k=k)

        self.baseline_train_outputs = assemble_out_of_fold(train_outputs_per_copy, fold_train)
        self.baseline_val_outputs = assemble_out_of_fold(val_outputs_per_copy, fold_val)
        self.baseline_test_outputs = assemble_out_of_fold(test_outputs_per_copy, fold_test)

        probs = torch.softmax(self.baseline_val_outputs, dim=1).numpy()
        y_pred = np.argmax(probs, axis=1)
        acc = accuracy_score(y_val.numpy(), y_pred)
        fold_accuracies["assembled_validation_accuracy"] = acc
        print(f"[Crossfit assembled guide] Validation Accuracy: {acc:.4f}")
        if acc < 0.3:
            raise ValueError("Crossfit guide validation accuracy < 0.3")

        self.fold_accuracies = fold_accuracies
