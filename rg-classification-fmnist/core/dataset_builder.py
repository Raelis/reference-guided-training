from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import datasets
import os, copy
import urllib.request
from torchvision import transforms
from torch.utils.data import Subset

def load_model(model_class, path: str, device: str, window_size, horizon):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_params = checkpoint["model_params"]
    model_params, other_params = model_class.prepare_model_params(model_params)
    model_params["num_features"] = window_size
    model_params["output_dim"] = horizon
    model = model_class(**model_params).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model_class.model_postprocess(model)
    return model

class FashionMNISTWithGuide(torch.utils.data.Dataset):

    def __init__(self, subset, labels, baseline_outputs):
        self.subset = subset
        self.labels = labels
        self.baseline_outputs = baseline_outputs

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, _ = self.subset[idx]
        y = self.labels[idx]
        guide = self.baseline_outputs[idx]
        return x, y, guide


class DatasetBuilder:

    def __init__(
        self,
        noise_ratio=0,
        baseline_noise_ratio=0,
        baseline_method="no",
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):

        self.noise_ratio = noise_ratio
        self.guide_noise_ratio = baseline_noise_ratio
        self.noise_type = f"fmnist-sym_{int(noise_ratio * 100)}" if noise_ratio > 0 else "clean"
        self.guide_noise_type = f"fmnist-sym_{int(baseline_noise_ratio * 100)}" if baseline_noise_ratio > 0 else "clean"

        self.baseline_method = baseline_method
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)


        self.noise_cache_dir = "./noise_cache"
        os.makedirs(self.noise_cache_dir, exist_ok=True)




        if baseline_method == "MLP":
            from models.mlp_model import MyModel
        elif baseline_method == "CNN":
            from models.cnn2d import MyModel
        elif baseline_method == "Transformer":
            from models.transformer import MyModel
        elif baseline_method == "ResNet20":
            from models.resnet20 import MyModel
        elif baseline_method == "MLP-guide":
            from models.mlp_model_guide import MyModel
        elif baseline_method == "no":
            MyModel = None
        else:
            raise ValueError("Unknown guide model")

        self.MyModel = MyModel
        self.device = device





    def _noise_path(self):
        return os.path.join(
            self.noise_cache_dir,
            f"{self.noise_type}_seed{self.seed}_fashionmnist.pt"
        )

    def load_or_create_noise(self, y_clean):

        path = self._noise_path()

        if os.path.exists(path):
            return torch.load(path).long()

        if self.noise_type.startswith("fmnist-sym_"):
            rate = int(self.noise_type.split("_")[1]) / 100
            y_noisy = self.symmetric_noise(y_clean, rate)

        elif self.noise_type.startswith("fmnist-asym_"):
            rate = int(self.noise_type.split("_")[1]) / 100
            y_noisy = self.asymmetric_noise(y_clean, rate)

        elif self.noise_type == "clean":
            y_noisy = y_clean.clone()

        else:
            raise ValueError(self.noise_type)

        torch.save(y_noisy.to(torch.int64), path)
        return y_noisy





    def symmetric_noise(self, y, noise_rate):

        rng = np.random.default_rng(self.seed)
        y_noisy = y.clone().numpy()

        n = len(y_noisy)
        n_noisy = int(noise_rate * n)

        noisy_idx = rng.choice(n, n_noisy, replace=False)

        for idx in noisy_idx:
            original = y_noisy[idx]
            new_label = rng.integers(0, 10)

            while new_label == original:
                new_label = rng.integers(0, 10)

            y_noisy[idx] = new_label

        return torch.tensor(y_noisy, dtype=torch.long)

    def asymmetric_noise(self, y, noise_rate):

        rng = np.random.default_rng(self.seed)

        transition = {
            0: 6,
            2: 4,
            3: 5,
            5: 3,
            7: 9
        }

        y_noisy = y.clone().numpy()

        for cls in transition:
            cls_idx = np.where(y_noisy == cls)[0]
            n_noisy = int(noise_rate * len(cls_idx))
            noisy_subset = rng.choice(cls_idx, n_noisy, replace=False)
            y_noisy[noisy_subset] = transition[cls]

        return torch.tensor(y_noisy, dtype=torch.long)





    def generate_baseline_predictions(self, X_train, X_val, X_test, y_test=None, y_val=None):

        is_pretrained = self.MyModel.model_name.endswith("-pre")

        if not is_pretrained:
            from .experiment_paths import method_dir
            standard_experiment_dir = str(method_dir(self.MyModel.model_name, self.guide_noise_ratio, 'erm'))
            batch_size = 256
            model_path = os.path.join(standard_experiment_dir, "best-model.pt")

        else:
            batch_size = 256
            model_path = None

        model = load_model(
            self.MyModel,
            model_path,
            self.device,
            (1, 28, 28),
            10
        )
        model.eval()

        def forward_loader(X):
            loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
            outs = []
            with torch.no_grad():
                for (xb,) in loader:
                    xb = xb.to(self.device)
                    outs.append(model(xb).cpu())
            return torch.cat(outs, dim=0)

        self.baseline_train_outputs = forward_loader(X_train)
        self.baseline_val_outputs = forward_loader(X_val)
        self.baseline_test_outputs = forward_loader(X_test)



        if y_val is None:
            raise ValueError('Clean validation labels required for reference sanity check')
        probs = torch.softmax(self.baseline_val_outputs, dim=1).numpy()
        y_pred = np.argmax(probs, axis=1)
        acc = accuracy_score(y_val.numpy(), y_pred)
        print(f"[Baseline sanity check] Validation Accuracy: {acc:.4f}")
        if acc < 0.3:
            raise ValueError("Guide validation accuracy < 0.3")





    def build(self):

        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

        train_base = datasets.FashionMNIST(
            root="./data",
            train=True,
            transform=test_transform,
            download=True
        )

        train_aug = datasets.FashionMNIST(
            root="./data",
            train=True,
            transform=train_transform,
            download=False
        )

        test_full = datasets.FashionMNIST(
            root="./data",
            train=False,
            transform=test_transform,
            download=True
        )

        y_clean = train_base.targets.detach().clone()
        y_test_clean = test_full.targets.detach().clone()

        train_label = self.load_or_create_noise(y_clean)


        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(train_base))

        train_idx = idx[:54000]
        val_idx = idx[54000:]

        train_subset = Subset(train_aug, train_idx)
        val_subset = Subset(train_base, val_idx)
        test_subset = test_full

        y_train = train_label[train_idx]
        y_val = y_clean[val_idx]
        y_test = y_test_clean


        if self.baseline_method != "no":

            def extract(dataset):
                loader = DataLoader(dataset, batch_size=256, shuffle=False)
                xs = []
                for x, _ in loader:
                    xs.append(x)
                return torch.cat(xs)

            X_train = extract(Subset(train_base, train_idx))
            X_val = extract(Subset(train_base, val_idx))
            X_test = extract(test_full)

            self.generate_baseline_predictions(
                X_train, X_val, X_test, y_test=y_test, y_val=y_val
            )

        else:
            self.baseline_train_outputs = y_train.float()
            self.baseline_val_outputs = y_val.float()
            self.baseline_test_outputs = y_test.float()

        self.train_dataset = FashionMNISTWithGuide(
            train_subset, y_train.long(), self.baseline_train_outputs
        )

        self.val_dataset = FashionMNISTWithGuide(
            val_subset, y_val.long(), self.baseline_val_outputs
        )

        self.test_dataset = FashionMNISTWithGuide(
            test_subset, y_test.long(), self.baseline_test_outputs
        )










    def build_loaders(self, batch_size=256):
        workers = min(max((os.cpu_count() or 1) - 1, 0), 4)
        return (
            DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                       num_workers=workers, pin_memory=True, persistent_workers=workers > 0),
            DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False,
                       num_workers=workers, pin_memory=True, persistent_workers=workers > 0),
            DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                       num_workers=workers, pin_memory=True, persistent_workers=workers > 0),
        )