# train.py
import os
import math
from typing import Tuple, Dict

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms as T
import matplotlib.pyplot as plt

from model import create_model, IMG_SIZE, IN_CHANNELS, LATENT_DIM, MODEL_TYPE, device

# ------------------------
# Dataset: high-res noisy handwriting
# ------------------------

class NoisyMNISTHighRes(Dataset):
    """
    MNIST -> resize to IMG_SIZE x IMG_SIZE, add blur + Gaussian noise.
    __getitem__ returns (noisy_img, clean_img)
    """

    def __init__(self, root: str = "data", train: bool = True,
                 noise_std: float = 0.25):
        super().__init__()

        transform_clean = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),               # [0,1]
        ])

        self.base = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transform_clean,
        )

        self.noise_std = noise_std
        # blur + mild affine, similar to "a bit shaky / blurry" handwriting
        self.augment = T.Compose([
            T.GaussianBlur(kernel_size=3, sigma=(0.5, 1.2)),
            T.RandomAffine(
                degrees=5,
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
            ),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        clean, _ = self.base[idx]           # (1,H,W)
        aug = self.augment(clean)

        # random noise strength per sample → more robust
        factor = 0.5 + torch.rand(1).item()   # [0.5, 1.5]
        noise = torch.randn_like(aug) * self.noise_std * factor

        noisy = (aug + noise).clamp(0.0, 1.0)
        return noisy, clean


# ------------------------
# Metrics & perceptual loss
# ------------------------

def _gaussian_window(window_size: int, sigma: float,
                     device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    gauss_1d = torch.exp(- (coords ** 2) / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
    window = gauss_2d.unsqueeze(0).unsqueeze(0)  # (1,1,K,K)
    return window


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM) for grayscale images in [0,1].
    x, y: (B, C, H, W)
    Returns mean SSIM over batch.
    """
    assert x.shape == y.shape, "SSIM: shapes must match"

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    B, C, H, W = x.shape
    window = _gaussian_window(
        window_size=window_size,
        sigma=1.5,
        device=x.device,
        dtype=x.dtype,
    )
    window = window.expand(C, 1, window_size, window_size)

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=C) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    return ssim_map.mean()


def mse_loss_with_ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    L = 0.7 * MSE + 0.3 * (1 - SSIM)
    """
    mse = torch.mean((x - y) ** 2)
    ssim_val = ssim(x, y)
    loss = 0.7 * mse + 0.3 * (1.0 - ssim_val)
    return loss


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = torch.mean((x - y) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


# ------------------------
# Training / Eval loops
# ------------------------

def train_epoch(model, loader, optimizer, epoch, total_epochs) -> Dict:
    model.train()
    totals = {"loss": 0.0, "psnr": 0.0}
    n_batches = 0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        recon, z = model(noisy)
        loss = mse_loss_with_ssim(recon, clean)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals["loss"] += loss.item()
        totals["psnr"] += psnr(recon.detach(), clean)
        n_batches += 1

    for k in totals.keys():
        totals[k] /= n_batches

    print(f"[Train] Epoch {epoch}/{total_epochs} | "
          f"loss={totals['loss']:.6f} | psnr={totals['psnr']:.2f} dB")

    return totals


def eval_epoch(model, loader, epoch, total_epochs) -> Dict:
    model.eval()
    totals = {"loss": 0.0, "psnr": 0.0}
    n_batches = 0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            recon, z = model(noisy)
            loss = mse_loss_with_ssim(recon, clean)

            totals["loss"] += loss.item()
            totals["psnr"] += psnr(recon, clean)
            n_batches += 1

    for k in totals.keys():
        totals[k] /= n_batches

    print(f"[Valid] Epoch {epoch}/{total_epochs} | "
          f"loss={totals['loss']:.6f} | psnr={totals['psnr']:.2f} dB")

    return totals


# ------------------------
# Visualization helpers
# ------------------------

def save_training_curve(history: dict, path: str = "training_curve.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title(f"{MODEL_TYPE.upper()} Denoiser – {IMG_SIZE}x{IMG_SIZE}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[INFO] Saved training curve to {path}")


def save_sample_grid(model, loader, path: str = "samples.png", n_samples: int = 4):
    model.eval()
    noisy_batch, clean_batch = next(iter(loader))
    noisy_batch = noisy_batch[:n_samples].to(device)
    clean_batch = clean_batch[:n_samples].to(device)

    with torch.no_grad():
        recon_batch, _ = model(noisy_batch)

    noisy = noisy_batch.cpu().numpy()
    clean = clean_batch.cpu().numpy()
    recon = recon_batch.cpu().numpy()

    fig, axes = plt.subplots(3, n_samples, figsize=(2.2 * n_samples, 6))
    titles = ["Noisy", "Denoised", "Clean"]

    for i in range(n_samples):
        axes[0, i].imshow(noisy[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"{titles[0]} #{i+1}", fontsize=9)

        axes[1, i].imshow(recon[i, 0], cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title(titles[1], fontsize=9)

        axes[2, i].imshow(clean[i, 0], cmap="gray")
        axes[2, i].axis("off")
        axes[2, i].set_title(titles[2], fontsize=9)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[INFO] Saved sample grid to {path}")


# ------------------------
# Main
# ------------------------

def main():
    os.makedirs("weights", exist_ok=True)

    full_train = NoisyMNISTHighRes(root="data", train=True, noise_std=0.25)

    # use subset so training stays reasonable
    max_samples = 20000
    if len(full_train) > max_samples:
        full_train, _ = random_split(full_train, [max_samples, len(full_train) - max_samples])
        print(f"[INFO] Using subset of MNIST: {max_samples} samples")

    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size])

    batch_size = 64 if device.type == "cpu" else 128
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    weights_path = f"weights/mnist_{MODEL_TYPE}_denoiser_{IMG_SIZE}.pth"

    for epoch in range(1, num_epochs + 1):
        train_stats = train_epoch(model, train_loader, optimizer, epoch, num_epochs)
        val_stats = eval_epoch(model, val_loader, epoch, num_epochs)

        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(model.state_dict(), weights_path)
            print(f"[SAVE] New best model (val_loss={best_val_loss:.6f}) → {weights_path}")

    save_training_curve(history, "training_curve.png")
    save_sample_grid(model, val_loader, "samples.png", n_samples=4)


if __name__ == "__main__":
    main()
