# Stage 2: PyTorch MLP Classification

> Replace every bracketed item with facts from the supplied dataset and the
> outputs of `train_mlp.py`. Do not report the temporary smoke-test results.
> Keep the submitted report within the five-page limit.

## Section 1: Task Description

This project uses a multilayer perceptron (MLP) to classify [describe the
examples and what each class means]. The goal is to predict [target variable]
from [briefly describe the input features].

## Section 2: Model Description

The program compares a baseline MLP and two deeper-model training settings.
Each MLP consists of an input layer, fully connected hidden layers with ReLU
activations, and an output layer with one logit per class. Cross-entropy loss
operates directly on the logits.

Insert `model_architecture.png` here. The input width is determined by the
training data after preprocessing; the output width is the number of classes.

## Section 3: Experiment Settings

### 3.1 Dataset Description

The dataset contains [number] training examples and [number] testing examples,
provided in `train.csv` and `test.csv`. The target is column [name/position].
There are [number] classes: [class names]. The instructor-provided split was
used without additional splitting or cross-validation.

Numeric features were median-imputed and standardized. Categorical features
were imputed with their most frequent training value and one-hot encoded.
Preprocessing was fitted only on the training features.

### 3.2 Detailed Experimental Setups

All experiments used [number] epochs, batch size [number], and random seed
[number]. The linear layers use PyTorch's default initialization.

| Experiment | Hidden-layer widths | Optimizer | Learning rate | Loss |
|---|---|---|---:|---|
| baseline | 64, 32 | Adam | 0.001 | Cross-entropy |
| deeper_adamw | 128, 64, 32 | AdamW | 0.001 | Cross-entropy |
| deeper_weighted | 128, 64, 32 | AdamW | 0.0005 | Inverse-frequency weighted cross-entropy |

### 3.3 Evaluation Metrics

Accuracy is the fraction of correct predictions. Precision measures the
fraction of predicted class assignments that are correct; recall measures the
fraction of actual examples recovered; F1 is the harmonic mean of precision
and recall. Weighted averages weight each class by its support, while macro
averages give every class equal weight. The results include both weighted and
macro Precision, Recall, and F1 to make class-imbalance effects visible.

### 3.4 Source Code

Code location/link: [add the GitHub or online-drive URL after uploading].

### 3.5 Training Convergence Plot

Insert `training_loss.png` here. The figure plots mean training cross-entropy
loss per epoch for each experiment. Describe whether each curve decreases and
whether it appears to stabilize: [brief observation based on the generated
plot].

### 3.6 Model Performance

Copy the held-out test metrics from `results.csv`. Do not use training metrics
or the temporary smoke-test output.

| Experiment | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| deeper_adamw | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| deeper_weighted | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

### 3.7 Ablation Studies

Compare the baseline with `deeper_adamw` to discuss the effect of adding a
hidden layer and changing Adam to AdamW. Compare `deeper_adamw` with
`deeper_weighted` to discuss the effect of class-weighted loss and the
learning-rate change. Report the observed metric differences and note that
more than one setting changes in the latter comparison: [analysis].
