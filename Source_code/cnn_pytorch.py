"""
cnn_pytorch.py — GPU-Accelerated CNN with PyTorch
Author: Based on tutorial by Young H. Cho, Ph.D.

Architecture:
    Input (3×64×64)
        → Conv(3→16) → ReLU → MaxPool(2×2)
        → Conv(16→32) → ReLU → MaxPool(2×2)
        → Flatten
        → FC(32×14×14 → 128) → ReLU
        → FC(128 → 1) → Output

Usage:
    python cnn_pytorch.py --data_dir data/train --epochs 10
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ─────────────────────────────────────────────
# Device Setup
# ─────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def get_dataloaders(data_dir, img_size=64, batch_size=32,
                    val_split=0.2, seed=42):
    """
    Load Cats vs Dogs dataset using ImageFolder.
    Expects data_dir/cat/ and data_dir/dog/ subfolders.
    Returns (train_loader, val_loader).
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Normalize with ImageNet stats for stable training
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print(f"Classes: {full_dataset.classes}  |  Total images: {len(full_dataset)}")

    val_size   = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


# ─────────────────────────────────────────────
# Model Definition
# ─────────────────────────────────────────────

class CatDogCNN(nn.Module):
    """
    Two-block CNN for binary image classification.

    Block 1: Conv(3→16, 3×3, pad=1) → ReLU → MaxPool(2×2)
    Block 2: Conv(16→32, 3×3, pad=1) → ReLU → MaxPool(2×2)
    Head   : Flatten → FC(32·16·16 → 128) → ReLU → FC(128 → 1)
    """

    def __init__(self, img_size=64):
        super().__init__()
        # Convolutional blocks
        self.conv1 = nn.Conv2d(3,  16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # After two MaxPool(2×2): spatial size = img_size // 4
        reduced = img_size // 4
        self.fc1 = nn.Linear(32 * reduced * reduced, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # (B, 16, 32, 32)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # (B, 32, 16, 16)
        x = x.view(x.size(0), -1)                    # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)                               # raw logit
        return x


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(model, loader, optimizer, criterion):
    """One epoch of training. Returns average loss."""
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # (B,1) for MSELoss

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(model, loader):
    """Evaluate accuracy on a DataLoader. Returns accuracy (0–1)."""
    model.eval()
    correct, total = 0, 0

    t0 = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze(1)
            preds   = (outputs > 0.5).long()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    elapsed = time.time() - t0
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%  |  "
          f"Inference time: {elapsed:.2f}s  ({total} images)")
    return accuracy


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CNN — Cats vs Dogs')
    parser.add_argument('--data_dir',    type=str,   default='data/train')
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--img_size',    type=int,   default=64)
    args = parser.parse_args()

    # ── Data ──
    train_loader, val_loader = get_dataloaders(
        data_dir   = args.data_dir,
        img_size   = args.img_size,
        batch_size = args.batch_size,
    )

    # ── Model ──
    model     = CatDogCNN(img_size=args.img_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # ── Training loop ──
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        loss = train(model, train_loader, optimizer, criterion)
        elapsed = time.time() - t_epoch
        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Loss: {loss:.4f} | Time: {elapsed:.2f}s")

    total_time = time.time() - t_start
    print(f"\nTotal Training Time: {total_time:.2f}s")

    # ── Evaluation ──
    print("\nValidation set:")
    evaluate(model, val_loader)

    # ── Save model ──
    torch.save(model.state_dict(), 'catdog_cnn.pth')
    print("\nModel saved to catdog_cnn.pth")
