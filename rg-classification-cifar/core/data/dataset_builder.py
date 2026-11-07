import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from torchvision import datasets, transforms
import os
from core.utils import load_model
from core.experiment_paths import guide_dir
import urllib.request


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR10N_KEYS = {
    "clean": "clean_label",
    "aggre": "aggre_label",
    "worst": "worse_label",
    "random1": "random_label1",
    "random2": "random_label2",
    "random3": "random_label3",
}


TRAIN_SPLIT_END = 45000
VAL_SPLIT_END = 50000


class CIFARWithGuide(torch.utils.data.Dataset):
    """Pairs each image with its (noisy) label and the guide logits for that index.

    The subset carries the torchvision transform, so augmentation is applied
    lazily per epoch on the training split while the guide logits, computed once
    from non-augmented images, stay aligned by index.
    """

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
    def __init__(self, noise_type="worst",  guide_noise_type="clean", baseline_method="no", seed=42, device = "cuda" if torch.cuda.is_available() else "cpu"):
        self.noise_type = noise_type
        self.guide_noise_type = guide_noise_type
        self.baseline_method = baseline_method
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

        if baseline_method == "ResNet56":
            from models.resnet56 import MyModel
        elif baseline_method == "no":
            MyModel = None
        else:
            raise ValueError("CIFAR-10N supports only ResNet56 or no guide")

        self.MyModel = MyModel
        self.device = device


    def load_cifar10n_labels(self):
        os.makedirs("./data", exist_ok=True)
        cifar10n_path = "./data/CIFAR-10_human.pt"

        if not os.path.exists(cifar10n_path):
            url = "https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/data/CIFAR-10_human.pt"
            urllib.request.urlretrieve(url, cifar10n_path)

        noise_file = torch.load(cifar10n_path, weights_only=False)

        return {
            name: torch.tensor(noise_file[key])
            for name, key in CIFAR10N_KEYS.items()
        }


    def generate_baseline_predictions(self, X_train, X_val, n_test, y_val=None):
        model_path = guide_dir(self.guide_noise_type) / "best-model.pt"
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing ResNet56 guide checkpoint: {model_path}")

        batch_size = 256
        model = load_model(self.MyModel, str(model_path), self.device, (3, 32, 32), 10)

        model.eval()

        def forward_loader(X, batch_size):
            loader = DataLoader(
                TensorDataset(X),
                batch_size=batch_size,
                shuffle=False
            )

            outputs = []
            with torch.no_grad():
                for (xb,) in loader:
                    xb = xb.to(self.device)
                    outputs.append(model(xb).cpu())

            return torch.cat(outputs, dim=0)

        self.baseline_train_outputs = forward_loader(X_train, batch_size)
        self.baseline_val_outputs = forward_loader(X_val, batch_size)

        # Only train and validation require reference logits.
        self.baseline_test_outputs = torch.zeros(n_test, 10)

        if y_val is not None:
            val_pred = self.baseline_val_outputs.argmax(dim=1)
            val_acc = (val_pred == y_val).float().mean().item()
            print(f"[Guide validation diagnostic] Clean validation accuracy: {val_acc:.4f}")

    def build(self):


        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])


        train_base = datasets.CIFAR10(
            root="./data", train=True, transform=test_transform, download=True
        )
        train_augmented = datasets.CIFAR10(
            root="./data", train=True, transform=train_transform, download=False
        )
        test_full = datasets.CIFAR10(
            root="./data", train=False, transform=test_transform, download=True
        )

        y_test_clean = torch.tensor(test_full.targets)


        noise_map = self.load_cifar10n_labels()

        if self.noise_type not in noise_map:
            raise ValueError(f"Invalid noise_type: {self.noise_type}")

        train_label = noise_map[self.noise_type]


        train_idx = np.arange(0, TRAIN_SPLIT_END)
        val_idx   = np.arange(TRAIN_SPLIT_END, VAL_SPLIT_END)

        train_subset = Subset(train_augmented, train_idx)
        val_subset   = Subset(train_base, val_idx)
        test_subset  = test_full

        y_train = train_label[train_idx]


        # Clean labels drive selection; noisy labels are diagnostic only.
        y_val = noise_map["clean"][val_idx]
        self.val_labels_noisy = train_label[val_idx].long()
        self.val_labels_clean = y_val.long()

        y_test_final = y_test_clean

        self.train_labels_for_report = y_train


        if self.baseline_method != "no":

            print("Generating baseline outputs...")

            def extract_tensor(dataset):
                loader = DataLoader(dataset, batch_size=256, shuffle=False)
                xs = []
                for x, _ in loader:
                    xs.append(x)
                return torch.cat(xs)


            # Inference is deterministic and retains original sample ordering.
            X_train_base = extract_tensor(Subset(train_base, train_idx))
            X_val_base   = extract_tensor(Subset(train_base, val_idx))
            self.generate_baseline_predictions(
                X_train_base,
                X_val_base,
                len(y_test_final),
                y_val=y_val.long(),
            )

        else:

            self.baseline_train_outputs = y_train.float()
            self.baseline_val_outputs   = y_val.float()
            self.baseline_test_outputs  = y_test_final.float()


        self.train_dataset = CIFARWithGuide(
            train_subset,
            y_train.long(),
            self.baseline_train_outputs
        )

        self.val_dataset = CIFARWithGuide(
            val_subset,
            y_val.long(),
            self.baseline_val_outputs
        )

        self.test_dataset = CIFARWithGuide(
            test_subset,
            y_test_final.long(),
            self.baseline_test_outputs
        )


    def build_loaders(self, batch_size=64):
        workers = max(0, min((os.cpu_count() or 1) - 1, 4))
        loader_options = dict(batch_size=batch_size, num_workers=workers,
                              pin_memory=str(self.device).startswith("cuda"),
                              persistent_workers=workers > 0)
        return (
            DataLoader(self.train_dataset, shuffle=True, **loader_options),
            DataLoader(self.val_dataset, shuffle=False, **loader_options),
            DataLoader(self.test_dataset, shuffle=False, **loader_options),
        )
