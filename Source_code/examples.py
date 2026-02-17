"""
examples.py — MLP application examples
Author: Based on tutorial by Young H. Cho, Ph.D.

Demonstrates:
  1. 1D piecewise-linear regression
  2. Binary classification on concentric rings (bullseye)
  3. Multiclass classification on 3 Gaussian blobs

Run all examples:
    python examples.py

Run a specific example:
    python examples.py --task regression
    python examples.py --task binary
    python examples.py --task multiclass
"""

import argparse
import numpy as np
from mlp import MLP2, train


# ─────────────────────────────────────────────
# Example 1 — 1D Regression
# ─────────────────────────────────────────────

def example_regression():
    print("=" * 50)
    print("Example 1: 1D Piecewise-Linear Regression")
    print("=" * 50)

    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (400, 1))

    def target(x):
        return np.maximum(0, 0.5*x + 0.2) + 0.3 * np.maximum(0, -x + 0.5)

    y = target(X) + 0.05 * rng.normal(size=(400, 1))

    model = MLP2(in_dim=1, hidden_dim=32, out_dim=1, mode='regression', seed=1)
    losses = train(model, X, y, epochs=50, lr=1e-2, batch_size=64,
                   weight_decay=1e-4, verbose=True)

    print(f"\nFinal loss: {losses[-1]:.4f}")
    print("\nSample predictions (x | target | predicted):")
    xt   = np.linspace(-2, 2, 9).reshape(-1, 1)
    pred = model.forward(xt)
    for xi, ti, pi in zip(xt.flatten(), target(xt).flatten(), pred.flatten()):
        print(f"  x={xi:+.2f}  target={ti:.4f}  pred={pi:.4f}")


# ─────────────────────────────────────────────
# Example 2 — Binary Classification (Bullseye)
# ─────────────────────────────────────────────

def make_rings(n=600, inner_r=0.5, gap=0.2, noise=0.06, seed=0):
    """Generate a two-circle (concentric rings) dataset."""
    rng = np.random.default_rng(seed)
    n2  = n // 2

    # Inner ring — class 0
    theta1 = rng.uniform(0, 2*np.pi, n2)
    r1     = inner_r + noise * rng.normal(size=n2)
    x1     = np.c_[r1 * np.cos(theta1), r1 * np.sin(theta1)]

    # Outer ring — class 1
    theta2 = rng.uniform(0, 2*np.pi, n - n2)
    r2     = inner_r + gap + noise * rng.normal(size=n - n2)
    x2     = np.c_[r2 * np.cos(theta2), r2 * np.sin(theta2)]

    X = np.vstack([x1, x2])
    y = np.array([0]*n2 + [1]*(n - n2))
    return X, y


def example_binary():
    print("=" * 50)
    print("Example 2: Binary Classification — Concentric Rings")
    print("=" * 50)

    X, y = make_rings(n=800, inner_r=0.6, gap=0.5, noise=0.07, seed=1)

    model = MLP2(in_dim=2, hidden_dim=64, out_dim=2,
                 mode='classification', seed=2)
    train(model, X, y, epochs=200, lr=5e-3, batch_size=64,
          weight_decay=1e-4, verbose=True)

    probs = model.forward(X)
    preds = probs.argmax(axis=1)
    acc   = (preds == y).mean()
    print(f"\nTrain accuracy: {acc * 100:.2f}%")


# ─────────────────────────────────────────────
# Example 3 — Multiclass Classification (3 Blobs)
# ─────────────────────────────────────────────

def make_three_blobs(n=900, seed=0):
    """Generate 3 Gaussian clusters for multiclass classification."""
    rng   = np.random.default_rng(seed)
    means = np.array([[0, 0], [2.5, 0.5], [-2.0, 1.5]])
    cov   = np.array([[0.4, 0.0], [0.0, 0.4]])
    Xs, ys = [], []
    for k, m in enumerate(means):
        Xk = rng.multivariate_normal(m, cov, size=n // 3)
        yk = np.full(n // 3, k)
        Xs.append(Xk)
        ys.append(yk)
    return np.vstack(Xs), np.hstack(ys)


def example_multiclass():
    print("=" * 50)
    print("Example 3: Multiclass Classification — 3 Blobs")
    print("=" * 50)

    X, y = make_three_blobs(n=900, seed=4)

    model = MLP2(in_dim=2, hidden_dim=64, out_dim=3,
                 mode='classification', seed=3)
    train(model, X, y, epochs=200, lr=5e-3, batch_size=64, verbose=True)

    preds = model.forward(X).argmax(axis=1)
    acc   = (preds == y).mean()
    print(f"\nTrain accuracy: {acc * 100:.2f}%")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLP Examples')
    parser.add_argument(
        '--task',
        choices=['regression', 'binary', 'multiclass', 'all'],
        default='all',
        help='Which example to run (default: all)'
    )
    args = parser.parse_args()

    if args.task in ('regression', 'all'):
        example_regression()
        print()

    if args.task in ('binary', 'all'):
        example_binary()
        print()

    if args.task in ('multiclass', 'all'):
        example_multiclass()
