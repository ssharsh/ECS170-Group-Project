import argparse
import gzip
import json
import pickle
import random
import struct
from pathlib import Path
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".pgm", ".png", ".ppm", ".tif", ".tiff"}


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        classes: list[str],
    ) -> None:
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels).long()
        self.classes = classes

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def _pickle_splits(path: Path) -> tuple[ImageFolderDataset, ImageFolderDataset]:
    with path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict) or not all(
        split in data for split in ("train", "test")
    ):
        raise ValueError(f"Expected train and test splits in dataset file {path}.")

    raw_labels: list[int] = []
    for split in ("train", "test"):
        records = data[split]
        if not isinstance(records, list) or not records:
            raise ValueError(f"Dataset split {split!r} in {path} must not be empty.")
        for record in records:
            if not isinstance(record, dict) or not {"image", "label"} <= record.keys():
                raise ValueError(
                    f"Invalid image/label record in split {split!r} of {path}."
                )
            label = record["label"]
            if not isinstance(label, (int, np.integer)):
                raise ValueError(f"Invalid class label in split {split!r} of {path}.")
            raw_labels.append(int(label))

    class_values = sorted(set(raw_labels))
    class_to_index = {value: index for index, value in enumerate(class_values)}
    classes = [str(value) for value in class_values]
    datasets = []
    for split in ("train", "test"):
        records = data[split]
        images: list[np.ndarray] = []
        labels: list[int] = []
        expected_shape: tuple[int, ...] | None = None
        for record in records:
            pixels = np.asarray(record["image"], dtype=np.uint8)
            if pixels.ndim == 2:
                pixels = pixels[np.newaxis, ...]
            elif pixels.ndim == 3 and pixels.shape[-1] in (1, 3):
                pixels = pixels.transpose(2, 0, 1)
            elif pixels.ndim != 3 or pixels.shape[0] not in (1, 3):
                raise ValueError(
                    f"Unsupported image shape {pixels.shape} in {path}."
                )
            if expected_shape is None:
                expected_shape = pixels.shape
            elif pixels.shape != expected_shape:
                raise ValueError(
                    f"Inconsistent image dimensions in split {split!r} of {path}."
                )
            images.append(pixels)
            labels.append(class_to_index[int(record["label"])])
        datasets.append(
            ImageFolderDataset(np.stack(images), np.asarray(labels), classes)
        )

    train_data, test_data = datasets
    if train_data.images.shape[1:] != test_data.images.shape[1:]:
        raise ValueError(f"Training and testing image dimensions differ in {path}.")
    return train_data, test_data


def _image_records(
    directory: Path, class_names: list[str] | None = None
) -> tuple[list[np.ndarray], list[int], list[str]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    class_dirs = sorted(path for path in directory.iterdir() if path.is_dir())
    if class_names is None:
        class_names = [path.name for path in class_dirs]
    if not class_names:
        raise ValueError(f"No class subdirectories found in {directory}.")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    images: list[np.ndarray] = []
    labels: list[int] = []
    expected_shape: tuple[int, ...] | None = None
    for class_dir in class_dirs:
        if class_dir.name not in class_to_index:
            raise ValueError(
                f"Unknown class directory {class_dir.name!r} in {directory}."
            )
        image_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for path in image_paths:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if expected_shape is None:
                expected_shape = pixels.shape
            elif pixels.shape != expected_shape:
                raise ValueError(
                    f"Inconsistent image size in {directory}: {path} is "
                    f"{pixels.shape[:2]}, expected {expected_shape[:2]}."
                )
            images.append(pixels.transpose(2, 0, 1))
            labels.append(class_to_index[class_dir.name])
    if not images:
        raise ValueError(f"No supported images found in {directory}.")
    return images, labels, class_names


def _folder_splits(
    root: Path, *, allow_unsplit: bool = False
) -> tuple[ImageFolderDataset, ImageFolderDataset]:
    train_dir, test_dir = root / "train", root / "test"
    if train_dir.is_dir() and test_dir.is_dir():
        train_images, train_labels, classes = _image_records(train_dir)
        test_images, test_labels, _ = _image_records(test_dir, classes)
    elif train_dir.exists() or test_dir.exists():
        raise FileNotFoundError(
            f"Both {train_dir} and {test_dir} are required for a split dataset."
        )
    elif allow_unsplit and root.is_dir():
        class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        classes = [path.name for path in class_dirs]
        train_images: list[np.ndarray] = []
        train_labels: list[int] = []
        test_images: list[np.ndarray] = []
        test_labels: list[int] = []
        for class_index, class_dir in enumerate(class_dirs):
            paths = sorted(
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if len(paths) < 2:
                raise ValueError(
                    f"Class {class_dir.name!r} needs at least two images to split."
                )
            split_at = 5 if len(paths) >= 10 else max(1, int(len(paths) * 0.8))
            for path, image_list, label_list in [
                *((p, train_images, train_labels) for p in paths[:split_at]),
                *((p, test_images, test_labels) for p in paths[split_at:]),
            ]:
                with Image.open(path) as image:
                    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
                image_list.append(pixels.transpose(2, 0, 1))
                label_list.append(class_index)
        if not classes:
            raise ValueError(f"No class subdirectories found in {root}.")
    else:
        raise FileNotFoundError(
            f"Expected {root}/train/<class>/ and {root}/test/<class>/ image "
            "directories."
        )

    train_array = np.stack(train_images)
    test_array = np.stack(test_images)
    if train_array.shape[1:] != test_array.shape[1:]:
        raise ValueError(
            f"Training and testing image dimensions differ in {root}: "
            f"{train_array.shape[1:]} vs {test_array.shape[1:]}."
        )
    return (
        ImageFolderDataset(train_array, np.asarray(train_labels), classes),
        ImageFolderDataset(test_array, np.asarray(test_labels), classes),
    )


def _find_idx_file(root: Path, stem: str) -> Path:
    for candidate in (root / stem, root / f"{stem}.gz"):
        if candidate.is_file():
            return candidate
    matches = sorted(root.rglob(stem)) + sorted(root.rglob(f"{stem}.gz"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find {stem} (or {stem}.gz) under {root}."
    )


def _read_idx(path: Path, expected_magic: int) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        magic = struct.unpack(">I", stream.read(4))[0]
        if magic != expected_magic:
            raise ValueError(f"Invalid IDX file {path}: magic number {magic}.")
        if expected_magic == 2051:
            count, rows, columns = struct.unpack(">III", stream.read(12))
            values = np.frombuffer(stream.read(), dtype=np.uint8)
            if values.size != count * rows * columns:
                raise ValueError(f"Invalid IDX image data length in {path}.")
            return values.reshape(count, 1, rows, columns).copy()
        count = struct.unpack(">I", stream.read(4))[0]
        values = np.frombuffer(stream.read(), dtype=np.uint8)
        if values.size != count:
            raise ValueError(f"Invalid IDX label data length in {path}.")
        return values.copy()


class MNISTDataset:
    @staticmethod
    def load(root: Path) -> tuple[ImageFolderDataset, ImageFolderDataset]:
        if root.is_file():
            return _pickle_splits(root)
        splits = []
        for split in ("train", "t10k"):
            images = _read_idx(
                _find_idx_file(root, f"{split}-images-idx3-ubyte"), 2051
            )
            labels = _read_idx(
                _find_idx_file(root, f"{split}-labels-idx1-ubyte"), 2049
            )
            if len(images) != len(labels):
                raise ValueError(f"MNIST {split} image and label counts differ.")
            splits.append((images, labels))
        classes = [str(index) for index in range(10)]
        return (
            ImageFolderDataset(splits[0][0], splits[0][1], classes),
            ImageFolderDataset(splits[1][0], splits[1][1], classes),
        )


class ORLDataset:
    @staticmethod
    def load(root: Path) -> tuple[ImageFolderDataset, ImageFolderDataset]:
        if root.is_file():
            return _pickle_splits(root)
        return _folder_splits(root, allow_unsplit=True)


class CIFARDataset:
    @staticmethod
    def load(root: Path) -> tuple[ImageFolderDataset, ImageFolderDataset]:
        if root.is_file():
            return _pickle_splits(root)
        return _folder_splits(root)


class CNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        channels: tuple[int, ...],
        kernel_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_channels = input_channels
        for channel_count in channels:
            layers.extend(
                (
                    nn.Conv2d(
                        previous_channels,
                        channel_count,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                )
            )
            previous_channels = channel_count
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(previous_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


EXPERIMENTS = (
    {
        "name": "baseline",
        "channels": (32, 64),
        "kernel_size": 3,
        "hidden_size": 128,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "weighted_loss": False,
    },
    {
        "name": "deeper_adamw",
        "channels": (32, 64, 128),
        "kernel_size": 3,
        "hidden_size": 256,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weighted_loss": False,
    },
    {
        "name": "wide_kernel_weighted",
        "channels": (32, 64),
        "kernel_size": 5,
        "hidden_size": 128,
        "optimizer": "AdamW",
        "learning_rate": 0.0005,
        "weighted_loss": True,
    },
)


def _evaluate(
    model: CNN,
    loader: DataLoader,
    device: torch.device,
    channel_mean: torch.Tensor | None,
    channel_std: torch.Tensor | None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device=device, dtype=torch.float32).div_(255)
            if channel_mean is not None and channel_std is not None:
                images = (images - channel_mean) / channel_std
            outputs = model(images)
            predictions.append(outputs.argmax(dim=1).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(targets), np.concatenate(predictions)


def _train_experiment(
    config: dict,
    train_data: Dataset,
    test_data: Dataset,
    num_classes: int,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    *,
    augment_cifar: bool = False,
) -> tuple[dict, list[float]]:
    torch.manual_seed(seed)
    model = CNN(
        input_channels=train_data[0][0].shape[0],
        num_classes=num_classes,
        channels=config["channels"],
        kernel_size=config["kernel_size"],
        hidden_size=config["hidden_size"],
    ).to(device)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    channel_mean = None
    channel_std = None
    if augment_cifar:
        channel_mean = (
            train_data.images.float().mean(dim=(0, 2, 3)).div(255).to(device)
        )[None, :, None, None]
        channel_std = (
            train_data.images.float().std(dim=(0, 2, 3), unbiased=False)
            .div(255)
            .clamp_min(1e-6)
            .to(device)
        )[None, :, None, None]
    if config["weighted_loss"]:
        labels = train_data.labels.numpy()
        counts = np.bincount(labels, minlength=num_classes)
        if np.any(counts == 0):
            raise ValueError("Every class must have at least one training image.")
        weights = len(labels) / (num_classes * counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
        loss_name = "weighted cross-entropy"
    else:
        criterion = nn.CrossEntropyLoss()
        loss_name = "cross-entropy"
    optimizer_type: Type[torch.optim.Optimizer] = (
        torch.optim.Adam if config["optimizer"] == "Adam" else torch.optim.AdamW
    )
    optimizer = optimizer_type(model.parameters(), lr=config["learning_rate"])

    losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        samples_seen = 0
        for images, labels in train_loader:
            images = images.to(device=device, dtype=torch.float32).div_(255)
            if channel_mean is not None and channel_std is not None:
                images = (images - channel_mean) / channel_std
            if augment_cifar:
                flip_mask = torch.rand(len(images), device=device) < 0.5
                images[flip_mask] = images[flip_mask].flip(-1)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            samples_seen += len(labels)
        losses.append(total_loss / samples_seen)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(
                f"{config['name']}: epoch {epoch + 1}/{epochs}, "
                f"loss={losses[-1]:.4f}"
            )

    targets, predictions = _evaluate(
        model, test_loader, device, channel_mean, channel_std
    )
    weighted = precision_recall_fscore_support(
        targets, predictions, average="weighted", zero_division=0
    )
    macro = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    metrics = {
        "experiment": config["name"],
        "convolution_channels": list(config["channels"]),
        "kernel_size": config["kernel_size"],
        "stride": 1,
        "padding": config["kernel_size"] // 2,
        "pooling": "2x2 max pooling after each convolution",
        "hidden_size": config["hidden_size"],
        "optimizer": config["optimizer"],
        "learning_rate": config["learning_rate"],
        "loss": loss_name,
        "accuracy": accuracy_score(targets, predictions),
        "precision_weighted": weighted[0],
        "recall_weighted": weighted[1],
        "f1_weighted": weighted[2],
        "precision_macro": macro[0],
        "recall_macro": macro[1],
        "f1_macro": macro[2],
    }
    return metrics, losses


def _train_orl_pca_svm(
    train_data: ImageFolderDataset,
    test_data: ImageFolderDataset,
) -> dict:
    train_images = (
        train_data.images.numpy()
        .astype(np.float32)
        .mean(axis=1)
        .reshape(len(train_data), -1)
        / 255
    )
    test_images = (
        test_data.images.numpy()
        .astype(np.float32)
        .mean(axis=1)
        .reshape(len(test_data), -1)
        / 255
    )
    pca = PCA(n_components=40, svd_solver="randomized", random_state=0)
    train_features = pca.fit_transform(train_images)
    test_features = pca.transform(test_images)
    model = SVC(C=10, gamma=0.01, kernel="rbf")
    model.fit(train_features, train_data.labels.numpy())
    targets = test_data.labels.numpy()
    predictions = model.predict(test_features)
    matrix = np.zeros(
        (len(train_data.classes), len(train_data.classes)), dtype=np.int64
    )
    np.add.at(matrix, (targets, predictions), 1)
    true_positive = np.diag(matrix)
    class_support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros_like(true_positive, dtype=float),
        where=predicted_support != 0,
    )
    recall = np.divide(
        true_positive,
        class_support,
        out=np.zeros_like(true_positive, dtype=float),
        where=class_support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    class_weights = class_support / class_support.sum()
    return {
        "experiment": "pca_svm",
        "model": "40-component PCA + RBF SVM",
        "optimizer": "SVC",
        "loss": None,
        "accuracy": float(np.mean(targets == predictions)),
        "precision_weighted": np.dot(precision, class_weights),
        "recall_weighted": np.dot(recall, class_weights),
        "f1_weighted": np.dot(f1, class_weights),
        "precision_macro": precision.mean(),
        "recall_macro": recall.mean(),
        "f1_macro": f1.mean(),
    }


def _dataset_loaders(dataset_name: str, root: Path):
    loaders = {
        "mnist": MNISTDataset,
        "orl": ORLDataset,
        "cifar": CIFARDataset,
    }
    pickle_path = root / dataset_name.upper()
    dataset_root = pickle_path if pickle_path.is_file() else root / dataset_name
    return loaders[dataset_name].load(dataset_root)


def run_dataset(
    dataset_name: str,
    root: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    experiment_name: str,
) -> None:
    train_data, test_data = _dataset_loaders(dataset_name, root)
    if train_data.classes != test_data.classes:
        raise ValueError(f"{dataset_name}: training and testing classes differ.")
    if len(train_data) == 0 or len(test_data) == 0:
        raise ValueError(f"{dataset_name}: train and test sets must not be empty.")
    dataset_dir = output_dir / dataset_name
    plots_dir = dataset_dir / "plots"
    metrics_dir = dataset_dir / "metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\n{dataset_name.upper()}: {len(train_data):,} train, "
        f"{len(test_data):,} test, {len(train_data.classes)} classes, "
        f"image shape {tuple(train_data[0][0].shape)}"
    )

    results = []
    losses_by_experiment = {}
    if experiment_name == "pca_svm" and dataset_name != "orl":
        raise ValueError("The pca_svm experiment is only available for ORL.")
    experiment_configs = EXPERIMENTS if experiment_name == "all" else tuple(
        config for config in EXPERIMENTS if config["name"] == experiment_name
    )
    for config in experiment_configs:
        metrics, losses = _train_experiment(
            config,
            train_data,
            test_data,
            len(train_data.classes),
            epochs,
            batch_size,
            seed,
            device,
            augment_cifar=dataset_name == "cifar",
        )
        results.append(metrics)
        losses_by_experiment[config["name"]] = losses

    if dataset_name == "orl" and experiment_name in ("all", "pca_svm"):
        results.append(_train_orl_pca_svm(train_data, test_data))

    results_frame = pd.DataFrame(results)
    (metrics_dir / "results.json").write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "train_examples": len(train_data),
                "test_examples": len(test_data),
                "classes": train_data.classes,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    results_frame.to_csv(metrics_dir / "results.csv", index=False)

    if losses_by_experiment:
        for name, losses in losses_by_experiment.items():
            plt.plot(range(1, len(losses) + 1), losses, label=name)
        plt.xlabel("Training epoch")
        plt.ylabel("Mean training cross-entropy")
        plt.title(f"{dataset_name.upper()} CNN training loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "training_loss.png", dpi=150)
        plt.close()
    print(results_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved outputs under {dataset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNNs on MNIST, ORL faces, and CIFAR image folders."
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "mnist", "orl", "cifar"),
        default="all",
        help="Dataset to run (default: all three).",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results" / "cnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment",
        choices=(
            "all",
            "pca_svm",
            *(config["name"] for config in EXPERIMENTS),
        ),
        default="all",
        help="Train every configuration or one CNN/ORL PCA-SVM experiment.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    datasets = ("mnist", "orl", "cifar") if args.dataset == "all" else (args.dataset,)
    for dataset_name in datasets:
        run_dataset(
            dataset_name,
            data_dir,
            output_dir,
            args.epochs,
            args.batch_size,
            args.seed,
            device,
            args.experiment,
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
