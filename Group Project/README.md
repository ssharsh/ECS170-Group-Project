# ECS 170 MLP project

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

## Model and experiments

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

## How to add data
Put your data sets into data folder, then run train_mlp.py
