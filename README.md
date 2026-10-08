# CS6301-Project1
mulitlayer perceptron implemented from first principles with numpy, including manual
backpropagation, gradient checking, and controlled experiments 

# MLP from Scratch for MNIST

This project implements a multilayer perceptron (MLP) from scratch using NumPy and trains it to classify handwritten digits from the MNIST dataset.

The purpose of the project is to implement and verify the core components of neural-network training without using automatic differentiation or neural-network frameworks. The implementation includes forward propagation, manual backpropagation, numerically stable softmax cross-entropy, minibatch SGD, gradient checking, sanity checks, and controlled experiments.

## Project Structure

```text
mlp-from-scratch/
├── src/
│   ├── model.py
│   ├── data.py
│   ├── train.py
│   ├── toy_example.py
│   ├── toy_refactored.py
│   ├── mnist_sanity.py
│   └── sanity_checks.py
│
├── experiments/
│   ├── run_learning_rate.py
│   ├── run_weight_scale.py
│   └── run_batch_size.py
│
├── plots/
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   ├── learning_rate_validation_loss.png
│   ├── learning_rate_validation_accuracy.png
│   ├── weight_scale_validation_loss.png
│   ├── weight_scale_validation_accuracy.png
│   ├── batch_size_validation_loss.png
│   └── batch_size_validation_accuracy.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The MNIST dataset is downloaded using `kagglehub`, so it does not need to be manually stored in the repository.

## Main Training Run

The primary MNIST training implementation is:

```text
src/train.py
```

Run it from the project root with:

```bash
python3 -m src.train
```

This loads MNIST, initializes the network, performs minibatch SGD, reports training and validation loss/accuracy, and generates the baseline training curves.

## Assignment Components

### 1. Affine Layers and ReLU

Located in:

```text
src/model.py
```

The network architecture is

```text
784 inputs -> 128 hidden ReLU units -> 10 output logits
```

The affine transformations and ReLU activation are implemented directly with NumPy.

### 2. Numerically Stable Softmax Cross-Entropy

Located in:

```text
src/model.py
```

The implementation subtracts the maximum logit from each row before exponentiation and uses a log-sum-exp calculation for the cross-entropy loss. This avoids directly exponentiating large logits.

### 3. Forward Propagation

Located in:

```text
src/model.py
```

The complete forward pass computes

```text
X
 ↓
Z1 = X W1^T + b1
 ↓
H1 = ReLU(Z1)
 ↓
A = H1 W2^T + b2
 ↓
Softmax
 ↓
P
```

where `P` contains the predicted class probabilities.

### 4. Manual Backpropagation

Located in:

```text
src/model.py
```

All gradients are derived and implemented manually. No automatic differentiation is used.

The implementation computes gradients for:

```text
W1
b1
W2
b2
```

including propagation of the output error through the hidden layer and ReLU activation.

### 5. Minibatch Stochastic Gradient Descent

Located in:

```text
src/train.py
```

Training examples are shuffled before minibatches are constructed. Parameters are updated after each minibatch using SGD.

### 6. Training and Validation Curves

Located in:

```text
src/train.py
```

Generated plots are saved under:

```text
plots/
```

The baseline curves are:

```text
plots/loss_curve.png
plots/accuracy_curve.png
```

### 7. Tiny-Dataset Overfitting Test

Located in:

```text
src/sanity_checks.py
```

Run:

```bash
python3 src/sanity_checks.py
```

The network is trained on only 20 MNIST examples. It should be capable of reaching 100% training accuracy and a loss below \(10^{-3}\).

### 8. Numerical Gradient Checking

The gradient-checking implementation is demonstrated with the small toy network in:

```text
src/toy_refactored.py
```

Run:

```bash
python3 src/toy_refactored.py
```

The script compares manually computed analytical gradients against finite-difference numerical gradients for:

```text
W1
b1
W2
b2
```

Relative errors should be very small.

### 9. ReLU Derivative Convention

The ReLU derivative is implemented in:

```text
src/model.py
```

The convention used at the nondifferentiable point is

```text
ReLU'(0) = 0
```

and this convention is used consistently during backpropagation.

### 10. Reproducibility

A NumPy random-number generator is initialized from a specified seed. The seed controls both parameter initialization and minibatch shuffling so that experimental runs can be reproduced.

Controlled experiments are repeated over multiple seeds to measure seed-to-seed variation.

## Initial MNIST Sanity Check

Located in:

```text
src/mnist_sanity.py
```

Run:

```bash
python3 src/mnist_sanity.py
```

This verifies the initial network dimensions, probability output, loss, and accuracy before full training.

For an untrained 10-class classifier, the initial cross-entropy loss should be close to

$$
\log(10) \approx 2.3026.
$$

## Controlled Experiments

All controlled experiments use the same MNIST training subset and validation set while changing one experimental variable at a time.

Each configuration is evaluated across multiple random seeds.

### Learning Rate

Located in:

```text
experiments/run_learning_rate.py
```

Run:

```bash
python3 experiments/run_learning_rate.py
```

Compares:

```text
learning rate = 0.01
learning rate = 0.1
learning rate = 1.0
```

Plots are saved as:

```text
plots/learning_rate_validation_loss.png
plots/learning_rate_validation_accuracy.png
```

### Weight Initialization Scale

Located in:

```text
experiments/run_weight_scale.py
```

Run:

```bash
python3 experiments/run_weight_scale.py
```

Compares:

```text
weight scale = 0.001
weight scale = 0.01
weight scale = 0.1
```

Plots are saved as:

```text
plots/weight_scale_validation_loss.png
plots/weight_scale_validation_accuracy.png
```

### Batch Size

Located in:

```text
experiments/run_batch_size.py
```

Run:

```bash
python3 experiments/run_batch_size.py
```

Compares:

```text
batch size = 32
batch size = 64
batch size = 128
```

Plots are saved as:

```text
plots/batch_size_validation_loss.png
plots/batch_size_validation_accuracy.png
```

## Development and Verification

The network was first implemented using a small synthetic dataset in:

```text
src/toy_example.py
```

This version was used to inspect matrix dimensions and manually work through the forward and backward equations.

The implementation was then refactored into reusable functions in:

```text
src/toy_refactored.py
```

The toy network was trained to 100% accuracy and its analytical gradients were checked against numerical finite-difference gradients before the same equations were used for MNIST.

## Dataset

MNIST contains \(28\times28\) grayscale images of handwritten digits from 0 through 9.

Each image is flattened into a vector of 784 features and pixel values are normalized from the integer range `[0, 255]` to the floating-point range `[0, 1]`.

The dataset is divided into:

```text
Training:   50,000 examples
Validation: 10,000 examples
Test:       10,000 examples
```

The test set is kept separate from the controlled experiments and model-selection process.

## Requirements

The project primarily uses:

```text
numpy
matplotlib
kagglehub
```

See `requirements.txt` for the complete environment requirements.
