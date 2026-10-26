# ECS 170 Classification Project

## Setup and run in VS Code

Select this Python interpreter in VS Code:

```text
/home/ssharsh/.venvs/ecs170/bin/python
```

Install the plotting dependency once:

```bash
/home/ssharsh/.venvs/ecs170/bin/python -m pip install matplotlib
```

Place `train.csv` and `test.csv` beside `train_mlp.py`, then run from the
integrated terminal:

```bash
/home/ssharsh/.venvs/ecs170/bin/python train_mlp.py
```

By default, CSVs are treated as having no header and the first column (position
`0`) is the class label, consistent with the supplied inspection script. If
your files have a header, add `--header present`. If the label is in another
column, pass its zero-based position with `--target-column N`. For example:

```bash
/home/ssharsh/.venvs/ecs170/bin/python train_mlp.py --target-column 4 --header present
```

The program uses the instructor-provided train/test partition as-is. It fits
missing-value handling, scaling, categorical encoding, and label encoding from
the training data only. The test set is reserved for final evaluation.

## Stage 2: MLP model and experiments

All models are fully connected multilayer perceptrons. Linear layers use
PyTorch's default parameter initialization, ReLU is applied after each hidden
layer, and the output layer produces one logit per class (no softmax is needed
with cross-entropy loss).

| Experiment | Hidden layers | Optimizer | Learning rate | Loss |
|---|---|---|---:|---|
| baseline | 64, 32 units | Adam | 0.001 | Cross-entropy |
| deeper_adamw | 128, 64, 32 units | AdamW | 0.001 | Cross-entropy |
| deeper_weighted | 128, 64, 32 units | AdamW | 0.0005 | Inverse-frequency weighted cross-entropy |

The default run trains for 100 epochs with a batch size of 64. You can change
the epoch count, batch size, and random seed with `--epochs`, `--batch-size`,
and `--seed`. CUDA is used automatically if available; otherwise training
runs on CPU.

## Outputs and report

After a successful run:

- `model_architecture.png` illustrates the baseline and deeper MLP layouts.
- `training_loss.png` plots training loss by epoch for each experiment.
- `results.csv` and `results.json` contain held-out test metrics.

Accuracy and weighted/macro Precision, Recall, and F1 are reported. Weighted
averages account for class frequency; macro averages weight each class equally.
Use the actual generated outputs to complete the report's dataset description,
performance table, convergence discussion, and ablation-study comparison.

The CSV files were not present in the project folder when this scaffold was
prepared, so no dataset-specific results are included yet.

## Stage 3: CNN image classification

Install dependencies with `python -m pip install -r requirements.txt`. The
instructor-provided pickle files `MNIST`, `ORL`, and `CIFAR` can be used
directly by placing them in one directory:

```text
stage_3_data/
  MNIST
  ORL
  CIFAR
```

The loaders read the supplied train/test records and labels from those files.
They also support MNIST IDX files and ORL/CIFAR image folders. For the folder
format, use `mnist/` with the standard IDX files, `orl/train/<person>/` and
`orl/test/<person>/`, and `cifar/train/<class>/` and `cifar/test/<class>/`.
ORL also accepts unsplit subject folders and makes a deterministic 80/20 split.
ORL and CIFAR image folders support PNG, JPEG, BMP, PGM, PPM, and TIFF files.

Run all datasets or select one to iterate quickly:

```bash
python scripts/train_cnn.py --data-dir /path/to/stage_3_data
python scripts/train_cnn.py --dataset mnist --data-dir /path/to/stage_3_data
```

The default run uses 10 epochs, batch size 64, a fixed seed, and CUDA when
available. Three configurations compare a two-block 3x3 baseline, a deeper
three-block model, and a two-block 5x5 model with weighted cross-entropy. Each
dataset gets its own `results/cnn/<dataset>/metrics/results.csv` and
`results.json`, plus a training-loss plot in `results/cnn/<dataset>/plots/`.
Metrics are held-out test accuracy and weighted/macro Precision, Recall, and
F1. The JSON also records the class names and split sizes. The weighted-loss
experiment changes its optimizer, learning rate, kernel, and loss together, so
it is a combined comparison rather than an isolated loss ablation.

For the supplied files in this repository's sibling `project stage 3/`
directory, pass that directory as `--data-dir`. Training outputs are generated
under `results/cnn/`; use the metrics CSVs and loss plots to fill in
`REPORT_TEMPLATE_CNN.md`.
