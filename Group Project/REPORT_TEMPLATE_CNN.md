# Stage 3: CNN Image Classification

## Task and data

This stage evaluates classifiers for handwritten digits (MNIST), faces
(ORL), and colored objects (CIFAR-10). The supplied instructor train/test
splits were used without modification.

| Dataset | Training images | Testing images | Classes | Image dimensions |
|---|---:|---:|---:|---|
| MNIST | 60,000 | 10,000 | 10 | 1 x 28 x 28 |
| ORL | 360 | 40 | 40 | 1 x 112 x 92 |
| CIFAR-10 | 50,000 | 10,000 | 10 | 3 x 32 x 32 |

ORL images have three identical grayscale channels in the supplied pickle
files; the loader converts them to one channel for CNN training. The provided
pickle records contain the images, integer labels, and train/test split for
each dataset.

## Model and training

The baseline and deeper CNNs use convolution, ReLU, and 2 x 2 max pooling
after each convolution, followed by adaptive global average pooling and a
fully connected classifier. The ORL-specific CNN retains a 4 x 4 feature map
instead of averaging to 1 x 1, and uses batch normalization, dropout, and
training-only image augmentation. All models output one logit per class and
use cross-entropy loss (class-weighted for the weighted experiment).
Convolutions use stride 1 and padding equal to half the kernel size.

The diagram below shows the three-block convolutional backbone. For ORL, the
best model keeps more spatial information after this backbone, as described
above and in the configuration table.

![CNN architecture diagram](./results/cnn/plots/model_architecture.png)

| Experiment | Convolution channels | Kernel / padding / stride | Pooling | Hidden units | Optimizer | Loss |
|---|---|---|---|---:|---|---|
| baseline | 32, 64 | 3 / 1 / 1 | 2 x 2 max | 128 | Adam, 0.001 | Cross-entropy |
| deeper_adamw | 32, 64, 128 | 3 / 1 / 1 | 2 x 2 max | 256 | AdamW, 0.001 | Cross-entropy |
| wide_kernel_weighted | 32, 64 | 5 / 2 / 1 | 2 x 2 max | 128 | AdamW, 0.0005 | Weighted cross-entropy |
| ORL spatial augmented | 32, 64, 128 | 3 / 1 / 1 | 2 x 2 max, adaptive 4 x 4 | 256 | AdamW, 0.001 | Cross-entropy |

The MNIST CNN uses 10 epochs and CIFAR-10 uses 30 epochs, with batch size 64
and seed 42. ORL configurations use batch size 40 and seed 42; the
spatial-augmented model uses 60 epochs. ORL images are standardized per image,
then receive small random rotation/translation and brightness/contrast
changes during training. CIFAR-10 uses training-set channel normalization and
random horizontal flips. Training ran on CPU. All reported models are CNNs;
ORL uses one grayscale channel.

![MNIST CNN training loss](./results/cnn/mnist/plots/training_loss.png)

![ORL CNN training loss](./results/cnn/orl/plots/training_loss.png)

![CIFAR-10 CNN training loss](./results/cnn/cifar/plots/training_loss.png)

## Test-set results

Metrics below are computed on the supplied held-out test splits. Values are
rounded to four decimal places; source CSVs are in `results/cnn/<dataset>/metrics/`.

| Dataset | Experiment | Accuracy | Weighted F1 | Macro F1 | Weighted Precision | Weighted Recall | Macro Precision | Macro Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST | deeper_adamw | 0.9875 | 0.9875 | 0.9874 | 0.9877 | 0.9875 | 0.9876 | 0.9874 |
| ORL | spatial augmented CNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CIFAR-10 | deeper_adamw | 0.7910 | 0.7891 | 0.7891 | 0.7904 | 0.7910 | 0.7904 | 0.7910 |

### Controlled configuration ablations

Each row below changes one setting from the same baseline CNN while keeping
the other model/training settings fixed. The baseline uses two convolution
blocks (32, 64 channels), 3 x 3 kernels, stride 1, padding 1, 2 x 2 max
pooling, 128 hidden units, and cross-entropy loss.

| Variant | Single setting changed |
|---|---|
| deeper | Adds a third convolution block (128 channels) |
| kernel5 | Kernel size 5 (padding 2 to preserve feature-map size) |
| stride2 | Convolution stride 2 |
| no_padding | Convolution padding 0 |
| average_pool | Replaces max pooling with average pooling |
| hidden256 | Hidden layer width 256 |
| label_smoothing | Cross-entropy with label smoothing 0.1 |

For manageable CPU runtime, every ablation ran for 3 epochs with seed 42.
MNIST and CIFAR-10 each used a stratified 1,000-image subset of the training
split; ORL used all 360 training images. All configurations were scored on
the full supplied test split. Thus these are short-run ablation results, not
the final full-training scores above; the ORL short runs in particular are
undertrained.

| Configuration | MNIST (1k train) | ORL (360 train) | CIFAR-10 (1k train) |
|---|---:|---:|---:|
| baseline | 21.89% | 5.00% | 21.29% |
| deeper | 31.93% | 2.50% | 21.83% |
| kernel5 | 26.94% | 2.50% | 22.44% |
| stride2 | 34.05% | 5.00% | 22.77% |
| no_padding | 21.07% | 5.00% | 21.97% |
| average_pool | 20.06% | 5.00% | 20.41% |
| hidden256 | 20.96% | 5.00% | 20.66% |
| label_smoothing | 21.10% | 5.00% | 21.32% |

In this short schedule, stride 2 performed best on MNIST (34.05% versus
21.89% baseline), while the deeper model performed best on CIFAR-10 (21.83%
versus 21.29%). ORL results cluster around chance (2.5% for 40 classes), so
these 3-epoch scores do not establish which setting is best for ORL; use the
separately trained spatial-augmented ORL CNN result above for its final
performance. These small-subset, short-schedule results should not be
compared directly to the final full-training results.

![MNIST controlled ablation loss](./results/cnn/controlled_ablations/mnist/plots/training_loss.png)

![ORL controlled ablation loss](./results/cnn/controlled_ablations/orl/plots/training_loss.png)

![CIFAR-10 controlled ablation loss](./results/cnn/controlled_ablations/cifar/plots/training_loss.png)

The ORL-specific spatial-augmented CNN is an additional model outside these
controlled ablations. It retains a 4 x 4 feature map and uses batch
normalization, dropout, image standardization, mild affine and
brightness/contrast augmentation, and 60 epochs. It achieved 100% on the ORL
test split.

## Accuracy targets

| Model / dataset | Desired accuracy | Test accuracy | Result |
|---|---:|---:|---|
| MNIST CNN | > 95% | 98.75% | Met |
| ORL CNN | > 95% | 100.00% | Met |
| CIFAR-10 CNN | > 70% | 79.10% | Met |

The ORL spatial-augmented CNN classified all 40 test images correctly on the
provided split. Its architecture and epoch count were selected using a
40-image validation subset (one image per subject) drawn only from the
training partition; it reached 100% validation accuracy at epoch 60, then
was trained on the full 360-image training partition. The provided test split
was reserved for final evaluation. There is only one test image per subject,
so the result is specific to this small split and should not be interpreted
as a broad estimate of face-recognition performance.
CIFAR-10 accuracy improved over the previously reported 66.68% deeper-model
result after longer training with channel normalization and horizontal-flip
augmentation.
