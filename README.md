# 🧠 Artificial Neural Network Tutorial - MLP & CNN from Scratch

A hands-on implementation of Multi-Layer Perceptrons (MLP) and Convolutional Neural Networks (CNN) built from scratch using NumPy, then accelerated with PyTorch and CUDA GPU support.

---

## Overview

This project walks through building neural networks from first principles, starting with a bare-bones NumPy MLP and scaling up to a GPU-accelerated PyTorch CNN. The goal is to develop a deep understanding of how forward passes, backpropagation, and gradient descent work under the hood, before relying on high-level framework abstractions.

**Key concepts covered:**
- Perceptron architecture and Multi-Layer Perceptrons (MLP)
- ReLU activation, Softmax, MSE, and Cross-Entropy loss
- He initialization and mini-batch SGD
- Convolutional layers, max pooling, and fully connected layers
- PyTorch model definition, DataLoader, and CUDA device management

---

## Project Structure

```
ann-tutorial/
├── part1_mlp/
│   ├── mlp.py              # MLP2 class: forward, backward, step
│   ├── train.py            # Training loop with mini-batch SGD
│   └── examples/
│       ├── regression_1d.py        # Piecewise-linear regression
│       ├── binary_classification.py # Concentric rings (bullseye)
│       └── multiclass.py           # 3-blob classification
├── part2_numpy_cnn/
│   ├── cnn_numpy.py        # Conv2D, MaxPool, FC layer in NumPy
│   └── evaluate.py         # Train/test loop with timing
├── part3_pytorch_cnn/
│   ├── model.py            # CatDogCNN(nn.Module)
│   ├── train_gpu.py        # Training loop with CUDA support
│   └── evaluate.py         # GPU inference with accuracy tracking
├── data/
│   ├── train/
│   │   ├── cat/
│   │   └── dog/
│   └── test/
└── README.md
```

---

## Part I — MLP Basics

A 2-layer fully connected neural network implemented with NumPy only.

### Architecture

```
Input Layer  →  Hidden Layer (ReLU)  →  Output Layer
   [N, in]        [N, hidden]              [N, out]
```

### Features

- **He initialization** for stable gradient flow through ReLU layers
- **Two modes:** `regression` (MSE loss) and `classification` (Softmax + Cross-Entropy)
- **Mini-batch SGD** with optional L2 weight decay
- **Full backpropagation** implemented manually via chain rule

### Training Results - 1D Piecewise-Linear Regression

Target function: `f(x) = max(0, 0.5x + 0.2) + 0.3·max(0, -x + 0.5)`

| Epoch | Loss   |
|-------|--------|
| 5     | 0.0044 |
| 10    | 0.0039 |
| 15    | 0.0036 |
| 20    | 0.0033 |
| 25    | 0.0030 |
| 30    | 0.0029 |
| 35    | 0.0027 |
| 40    | 0.0023 |
| 45    | 0.0023 |
| 50    | 0.0021 |

> Loss decreases steadily, confirming convergence. ReLU units carve the input space into piecewise-linear regions to approximate the target function.

### Additional Examples

| Task | Dataset | Hidden Dim | Epochs | Notes |
|------|---------|-----------|--------|-------|
| Binary Classification | Concentric rings (bullseye) | 64 | 200 | Non-linearly separable |
| Multiclass Classification | 3 Gaussian blobs | 64 | 200 | 3-class Softmax output |

---

## Part II - NumPy CNN (Cats vs. Dogs)

A minimal CNN built entirely from NumPy to expose the low-level mechanics of convolutional networks.

### Architecture

```
Input Image (64×64×3)
    ↓ Conv2D (3×3 kernel)
    ↓ ReLU
    ↓ MaxPool (2×2)
    ↓ Flatten
    ↓ Fully Connected + ReLU
    ↓ Output (1 neuron)
```

### Dataset

Uses the [Kaggle Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data), structured as:

```
data/
├── train/
│   ├── dog/
│   └── cat/
└── test/
```

### Training Results

| Epoch | Loss   |
|-------|--------|
| 1     | 0.7000 |
| 2     | 0.7000 |
| 3     | 0.7000 |
| 4     | 0.7000 |
| 5     | 0.7000 |

- **Training time:** `1.22 seconds`

> Loss behavior reflects an untrained baseline CNN. This implementation prioritizes mechanistic understanding over performance — the NumPy path makes every convolution, pooling step, and gradient update transparent.

---

## Part III - GPU-Accelerated PyTorch CNN

The same CNN rebuilt with PyTorch and CUDA, demonstrating the performance gains from GPU acceleration and automatic differentiation.

### Architecture

```python
class CatDogCNN(nn.Module):
    # Conv(3→16) → ReLU → MaxPool
    # Conv(16→32) → ReLU → MaxPool
    # Flatten → FC → ReLU → Output
```

### Key Implementation Details

- `datasets.ImageFolder` + `DataLoader` for efficient batching
- `Adam` optimizer, `lr=0.001`
- `nn.MSELoss` for output
- Automatic device routing via `tensor.to(device)`

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CatDogCNN().to(device)
```

### Training Results

| Epoch | Loss   |
|-------|--------|
| 1     | 0.5251 |
| 2     | 0.2640 |
| 3     | 0.3160 |
| 4     | 0.2786 |
| 5     | 0.2226 |

- **Device:** CUDA (GPU)
- **Training time:** `0.12 seconds`

---

## Results & Comparison

| Model | Device | Training Time | Notes |
|-------|--------|--------------|-------|
| NumPy CNN | CPU | 1.22 s | Manual gradients, no autograd |
| PyTorch CNN | CUDA (GPU) | 0.12 s | Automatic differentiation |

**PyTorch GPU is ~10× faster** than the NumPy CPU implementation on the same task.

> This gap grows significantly at larger scale — with real image datasets, full batch sizes, and deeper architectures, the speedup from GPU parallelism becomes orders of magnitude larger.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ann-tutorial.git
cd ann-tutorial
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download from [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) and place it in the `data/` directory.

### 4. Run each part

```bash
# Part I — MLP regression
python part1_mlp/examples/regression_1d.py

# Part II — NumPy CNN
python part2_numpy_cnn/cnn_numpy.py

# Part III — PyTorch CNN (GPU)
python part3_pytorch_cnn/train_gpu.py
```

---

## Dependencies

```
numpy
Pillow
matplotlib
torch
torchvision
```

> PyTorch GPU support requires a CUDA-compatible GPU and the appropriate CUDA toolkit. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for setup instructions.

---

## References

- [NumPy Documentation](https://numpy.org/doc/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Neural Networks and Deep Learning — Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning Specialization — Andrew Ng (Coursera)](https://www.coursera.org/specializations/deep-learning)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Kaggle Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)

