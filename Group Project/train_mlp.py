import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = Path(__file__).resolve().parent


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: tuple[int, ...], num_classes: int):
        super().__init__()
        layers: list[nn.Module] = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(previous_size, hidden_size), nn.ReLU()))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def load_data(target_column: int, header: str) -> tuple[np.ndarray, ...]:
    header_arg: int | None = 0 if header == "present" else None
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Put train.csv and test.csv beside train_mlp.py."
            )

    train = pd.read_csv(train_path, header=header_arg)
    test = pd.read_csv(test_path, header=header_arg)
    if train.empty or test.empty:
        raise ValueError("train.csv and test.csv must both contain at least one row.")
    if train.shape[1] != test.shape[1]:
        raise ValueError(
            f"Train/test column counts differ ({train.shape[1]} vs {test.shape[1]})."
        )
    if not -train.shape[1] <= target_column < train.shape[1]:
        raise ValueError(
            f"Target column {target_column} is out of range for "
            f"{train.shape[1]} columns."
        )

    train_labels = train.iloc[:, target_column]
    test_labels = test.iloc[:, target_column]
    if train_labels.isna().any() or test_labels.isna().any():
        raise ValueError("Target column contains missing labels.")

    train_features = train.drop(columns=train.columns[target_column])
    test_features = test.drop(columns=test.columns[target_column])
    test_features.columns = train_features.columns

    numeric_columns = train_features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [
        column for column in train_features.columns if column not in numeric_columns
    ]
    transformers: list[tuple[str, Pipeline, list]] = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("No feature columns remain after removing the target column.")

    preprocessor = ColumnTransformer(transformers, sparse_threshold=0)
    x_train = preprocessor.fit_transform(train_features).astype(np.float32)
    x_test = preprocessor.transform(test_features).astype(np.float32)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels).astype(np.int64)
    known_labels = set(label_encoder.classes_)
    unknown_labels = set(test_labels.unique()) - known_labels
    if unknown_labels:
        raise ValueError(
            f"Test set contains labels not present in training data: "
            f"{sorted(map(str, unknown_labels))}"
        )
    y_test = label_encoder.transform(test_labels).astype(np.int64)

    print(f"Training rows: {len(y_train):,}; testing rows: {len(y_test):,}")
    print(f"Input features after preprocessing: {x_train.shape[1]:,}")
    print(f"Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")
    print(f"Target column: {target_column} (zero-based position)")
    return x_train, y_train, x_test, y_test, label_encoder.classes_


def run_experiment(
    name: str,
    hidden_sizes: tuple[int, ...],
    optimizer_name: str,
    learning_rate: float,
    class_weighted: bool,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict, list[float]]:
    model = MLP(x_train.shape[1], hidden_sizes, num_classes).to(device)
    if class_weighted:
        counts = np.bincount(y_train, minlength=num_classes)
        weights = len(y_train) / (num_classes * counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
    else:
        criterion = nn.CrossEntropyLoss()

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    epoch_losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        samples_seen = 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            samples_seen += len(labels)
        epoch_losses.append(total_loss / samples_seen)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"{name}: epoch {epoch + 1}/{epochs}, loss={epoch_losses[-1]:.6f}")

    model.eval()
    with torch.no_grad():
        predictions = (
            model(torch.from_numpy(x_test).to(device)).argmax(dim=1).cpu().numpy()
        )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_test, predictions, average="weighted", zero_division=0
        )
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    metrics = {
        "experiment": name,
        "hidden_layers": list(hidden_sizes),
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "loss": "weighted cross-entropy" if class_weighted else "cross-entropy",
        "accuracy": accuracy_score(y_test, predictions),
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }
    return metrics, epoch_losses


def save_architecture_plot(input_size: int, num_classes: int) -> Path:
    models = {
        "Baseline": [input_size, 64, 32, num_classes],
        "Deeper": [input_size, 128, 64, 32, num_classes],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, (name, dimensions) in zip(axes, models.items()):
        axis.axis("off")
        axis.set_title(name)
        for index, dimension in enumerate(dimensions):
            label = (
                f"Input\n{dimension} features"
                if index == 0
                else f"Output\n{dimension} classes"
                if index == len(dimensions) - 1
                else f"Hidden {index}\n{dimension} units\nReLU"
            )
            y = 0.9 - index * (0.8 / max(len(dimensions) - 1, 1))
            axis.text(
                0.5,
                y,
                label,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.5", "fc": "#e8f1fb", "ec": "#315b7d"},
                transform=axis.transAxes,
            )
            if index < len(dimensions) - 1:
                next_y = 0.9 - (index + 1) * (0.8 / (len(dimensions) - 1))
                axis.annotate(
                    "",
                    xy=(0.5, next_y + 0.055),
                    xytext=(0.5, y - 0.055),
                    xycoords=axis.transAxes,
                    arrowprops={"arrowstyle": "->", "color": "#315b7d"},
                )
    fig.suptitle("Fully connected MLP architectures")
    fig.tight_layout()
    path = DATA_DIR / "model_architecture.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and compare PyTorch MLP models using train.csv and test.csv."
    )
    parser.add_argument(
        "--target-column",
        type=int,
        default=0,
        help="Zero-based target-column position (default: 0, as in the inspection script).",
    )
    parser.add_argument(
        "--header",
        choices=("absent", "present"),
        default="absent",
        help="Whether the CSV files have a header row (default: absent).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    x_train, y_train, x_test, y_test, classes = load_data(
        args.target_column, args.header
    )
    architecture_path = save_architecture_plot(x_train.shape[1], len(classes))

    experiments = [
        ("baseline", (64, 32), "Adam", 1e-3, False),
        ("deeper_adamw", (128, 64, 32), "AdamW", 1e-3, False),
        ("deeper_weighted", (128, 64, 32), "AdamW", 5e-4, True),
    ]
    results = []
    losses_by_experiment: dict[str, list[float]] = {}
    for name, layers, optimizer, learning_rate, weighted in experiments:
        metrics, losses = run_experiment(
            name,
            layers,
            optimizer,
            learning_rate,
            weighted,
            x_train,
            y_train,
            x_test,
            y_test,
            len(classes),
            args.epochs,
            args.batch_size,
            device,
        )
        results.append(metrics)
        losses_by_experiment[name] = losses

    results_path = DATA_DIR / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    results_frame = pd.DataFrame(results).set_index("experiment")
    results_frame.to_csv(DATA_DIR / "results.csv")
    print("\nTest-set performance:")
    print(results_frame.to_string(float_format=lambda value: f"{value:.4f}"))

    for name, losses in losses_by_experiment.items():
        plt.plot(range(1, len(losses) + 1), losses, label=name)
    plt.xlabel("Training epoch")
    plt.ylabel("Training loss")
    plt.title("MLP training convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = DATA_DIR / "training_loss.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved metrics to {results_path} and results.csv")
    print(f"Saved convergence plot to {plot_path}")
    print(f"Saved architecture diagram to {architecture_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
