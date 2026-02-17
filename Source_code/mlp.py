"""
mlp.py — 2-Layer Multi-Layer Perceptron (NumPy only)
Author: Based on tutorial by Young H. Cho, Ph.D.

Architecture:
    Input → Hidden (ReLU) → Output
    Mode: 'regression' (MSE) or 'classification' (Softmax + Cross-Entropy)
"""

import math
import numpy as np


# ─────────────────────────────────────────────
# Activation Functions
# ─────────────────────────────────────────────

def relu(x):
    """Rectified Linear Unit activation."""
    return np.maximum(0.0, x)


def d_relu(x):
    """Derivative of ReLU — used in backpropagation."""
    return (x > 0).astype(x.dtype)


def softmax(logits):
    """
    Numerically stable softmax.
    Converts raw scores to class probabilities summing to 1.
    """
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y, num_classes):
    """Convert integer class labels to one-hot encoded vectors."""
    Y = np.zeros((y.size, num_classes))
    Y[np.arange(y.size), y] = 1.0
    return Y


# ─────────────────────────────────────────────
# MLP Model
# ─────────────────────────────────────────────

class MLP2:
    """
    2-Layer Fully Connected Neural Network.

    Forward pass:
        z1     = X @ W1 + b1
        h1     = ReLU(z1)
        z2     = h1 @ W2 + b2
        output = z2 (regression) or softmax(z2) (classification)

    Parameters
    ----------
    in_dim      : number of input features
    hidden_dim  : number of hidden neurons
    out_dim     : number of output neurons
    mode        : 'regression' or 'classification'
    seed        : random seed for reproducibility
    """

    def __init__(self, in_dim, hidden_dim, out_dim, mode='regression', seed=0):
        assert mode in ('regression', 'classification'), \
            "mode must be 'regression' or 'classification'"

        self.mode = mode
        rng = np.random.default_rng(seed)

        # He initialization — keeps gradient variance stable for ReLU
        k1 = math.sqrt(2.0 / in_dim)
        k2 = math.sqrt(2.0 / hidden_dim)

        self.W1 = rng.normal(0, k1, (in_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0, k2, (hidden_dim, out_dim))
        self.b2 = np.zeros((1, out_dim))

    def forward(self, X):
        """Compute forward pass and store intermediates for backprop."""
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h1 = relu(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2

        if self.mode == 'classification':
            self.probs = softmax(self.z2)
            return self.probs

        return self.z2  # raw output for regression

    def loss(self, y):
        """
        Compute loss given true labels y.
        - Regression    : 0.5 * MSE
        - Classification: Cross-Entropy
        """
        if self.mode == 'regression':
            diff = self.z2 - y
            return 0.5 * np.mean(np.sum(diff * diff, axis=1))
        else:
            N = y.shape[0]
            logp = -np.log(self.probs[np.arange(N), y] + 1e-12)
            return np.mean(logp)

    def backward(self, y):
        """
        Backpropagation via chain rule.
        Returns gradients for all parameters: (dW1, db1, dW2, db2)
        """
        N = self.X.shape[0]

        # ── Output layer gradient ──
        if self.mode == 'regression':
            dz2 = (self.z2 - y) / N
        else:
            Y = one_hot(y, self.z2.shape[1])
            dz2 = (self.probs - Y) / N

        # ── Layer 2 gradients ──
        dW2 = self.h1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)

        # ── Backprop through ReLU ──
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * d_relu(self.z1)

        # ── Layer 1 gradients ──
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def step(self, grads, lr=1e-2, weight_decay=0.0):
        """
        SGD weight update with optional L2 regularization.

        Parameters
        ----------
        grads        : tuple of (dW1, db1, dW2, db2)
        lr           : learning rate
        weight_decay : L2 regularization coefficient (0 = disabled)
        """
        dW1, db1, dW2, db2 = grads

        if weight_decay > 0:
            dW1 += weight_decay * self.W1
            dW2 += weight_decay * self.W2

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2


# ─────────────────────────────────────────────
# Mini-Batch Iterator
# ─────────────────────────────────────────────

def batch_iter(X, y, batch_size, shuffle=True, seed=0):
    """
    Yield (X_batch, y_batch) mini-batches.
    Shuffles indices each call when shuffle=True.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        j = idx[i:i + batch_size]
        yield X[j], y[j]


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train(model, X, y, epochs=200, lr=1e-2, batch_size=64,
          weight_decay=0.0, verbose=True):
    """
    Train MLP2 using mini-batch SGD.

    Parameters
    ----------
    model        : MLP2 instance
    X            : input features, shape (N, in_dim)
    y            : labels — float array for regression, int array for classification
    epochs       : number of full passes over the data
    lr           : learning rate
    batch_size   : samples per mini-batch
    weight_decay : L2 regularization coefficient
    verbose      : print loss every 10% of epochs

    Returns
    -------
    losses : list of full-batch loss values per epoch
    """
    losses = []

    for e in range(1, epochs + 1):
        for Xb, yb in batch_iter(X, y, batch_size, shuffle=True, seed=e):
            model.forward(Xb)
            grads = model.backward(yb)
            model.step(grads, lr=lr, weight_decay=weight_decay)

        # Track full-batch loss after each epoch
        model.forward(X)
        L = model.loss(y)
        losses.append(L)

        if verbose and (e % max(1, epochs // 10) == 0):
            print(f"Epoch {e:4d}/{epochs}  loss={L:.4f}")

    return losses
