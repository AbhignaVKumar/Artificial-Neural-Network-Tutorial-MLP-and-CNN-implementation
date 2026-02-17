"""
cnn_numpy.py — Simple CNN built with NumPy only
Author: Based on tutorial by Young H. Cho, Ph.D.

Architecture:
    Input (64×64×3)
        → Conv2D (3×3) → ReLU → MaxPool (2×2)
        → Flatten
        → Fully Connected → ReLU
        → Output (1 neuron, binary classification)

Usage:
    python cnn_numpy.py --data_dir data/train --epochs 10
"""

import os
import time
import argparse
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_data(data_dir, img_size=64):
    """
    Load images from data_dir/cat and data_dir/dog folders.
    Returns normalized float arrays and integer labels (0=cat, 1=dog).
    """
    X, y = [], []
    for label, folder in enumerate(['cat', 'dog']):
        path = os.path.join(data_dir, folder)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder not found: {path}")
        files = [f for f in os.listdir(path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for file in files:
            img = (Image.open(os.path.join(path, file))
                   .convert('RGB')
                   .resize((img_size, img_size)))
            X.append(np.array(img) / 255.0)
            y.append(label)
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


# ─────────────────────────────────────────────
# CNN Layers
# ─────────────────────────────────────────────

def conv2d(img, kernel):
    """
    2D convolution (valid padding, stride 1).
    img    : (H, W, C)
    kernel : (kH, kW, C)
    returns: (H-kH+1, W-kW+1)
    """
    h, w, _ = img.shape
    kh, kw, _ = kernel.shape
    out_h, out_w = h - kh + 1, w - kw + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(img[i:i+kh, j:j+kw, :] * kernel)
    return out


def maxpool(img, size=2):
    """
    2D max pooling with non-overlapping windows of given size.
    img  : (H, W)
    size : pooling window size
    returns: (H//size, W//size)
    """
    h, w = img.shape
    new_h, new_w = h // size, w // size
    pooled = np.zeros((new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            pooled[i, j] = np.max(
                img[i*size:(i+1)*size, j*size:(j+1)*size]
            )
    return pooled


# ─────────────────────────────────────────────
# CNN Class
# ─────────────────────────────────────────────

class NumpyCNN:
    """
    Minimal single-filter CNN.
    Only the fully connected layer weights are trained.
    """

    def __init__(self, img_size=64, seed=0):
        rng = np.random.default_rng(seed)
        self.conv_filter = rng.normal(0, 0.1, (3, 3, 3))
        pooled_size = (img_size - 2) // 2  # after conv (valid) + pool
        self.W_fc = rng.normal(0, 0.01, (pooled_size * pooled_size * 3, 1))
        self.b_fc = np.zeros((1,))

    def forward(self, img):
        """
        Run forward pass on a single image.
        Returns (output, flat, pooled, relu_out) for backprop.
        """
        conv_out  = conv2d(img, self.conv_filter)
        relu_out  = relu(conv_out)
        pooled    = maxpool(relu_out)
        flat      = pooled.flatten()

        # Pad or trim flat to match W_fc if sizes differ
        flat = flat[:self.W_fc.shape[0]]

        fc_out    = relu(np.dot(flat, self.W_fc) + self.b_fc)
        return fc_out, flat, pooled, relu_out

    def train(self, X_train, y_train, lr=0.001, epochs=10):
        """Train the FC layer using MSE loss and manual gradient descent."""
        history = []

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            t0 = time.time()

            for i in range(len(X_train)):
                img   = X_train[i]
                label = np.array([y_train[i]], dtype=float)

                y_pred, flat, _, _ = self.forward(img)
                loss = mse_loss(label, y_pred)
                total_loss += loss

                # Backprop through FC + ReLU
                grad_y  = mse_grad(label, y_pred) * relu_derivative(y_pred)
                grad_W  = np.outer(flat, grad_y)
                grad_b  = grad_y

                self.W_fc -= lr * grad_W
                self.b_fc -= lr * grad_b

            elapsed = time.time() - t0
            avg_loss = total_loss / len(X_train)
            history.append(avg_loss)
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Time: {elapsed:.2f}s")

        return history

    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evaluate accuracy on test set."""
        correct = 0
        t0 = time.time()

        for i in range(len(X_test)):
            y_pred, _, _, _ = self.forward(X_test[i])
            prediction = int(y_pred[0] > threshold)
            if prediction == y_test[i]:
                correct += 1

        elapsed = time.time() - t0
        accuracy = correct / len(X_test)
        print(f"\nTest Accuracy : {accuracy * 100:.2f}%")
        print(f"Inference Time: {elapsed:.2f}s ({len(X_test)} images)")
        return accuracy


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NumPy CNN — Cats vs Dogs')
    parser.add_argument('--data_dir', type=str, default='data/train')
    parser.add_argument('--epochs',   type=int, default=10)
    parser.add_argument('--lr',       type=float, default=0.001)
    parser.add_argument('--split',    type=float, default=0.8)
    args = parser.parse_args()

    print("Loading data...")
    X, y = load_data(args.data_dir)
    print(f"Loaded {len(X)} images. Shape: {X.shape}")

    split = int(args.split * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = NumpyCNN(img_size=64, seed=42)

    print(f"\nTraining for {args.epochs} epochs...")
    t_start = time.time()
    model.train(X_train, y_train, lr=args.lr, epochs=args.epochs)
    print(f"\nTotal Training Time: {time.time() - t_start:.2f}s")

    print("\nEvaluating...")
    model.evaluate(X_test, y_test)
