# train.py
import os
import math
from typing import Tuple, Dict

import torch
from torch import nn, optim
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
                 noise_std: float = 0.4):
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
        # Blur + random affine – бичмэл үсгийн real world distortion-тэй төстэй болгоно.
        self.augment = T.Compose([
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        clean, _ = self.base[idx]           # (1,H,W)
        aug = self.augment(clean)
        noise = torch.randn_like(aug) * self.noise_std
        noisy = (aug + noise).clamp(0.0, 1.0)
        return noisy, clean


# ------------------------
# Metrics
# ------------------------

def mse_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((x - y) ** 2)


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = torch.mean((x - y) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def vae_loss(recon: torch.Tensor,
             target: torch.Tensor,
             mu: torch.Tensor,
             logvar: torch.Tensor,
             beta_kl: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    recon: model output
    target: clean image
    """
    rec = F.mse_loss(recon, target, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = rec + beta_kl * kl
    return loss, rec.detach(), kl.detach()


# ------------------------
# Training / Eval loops
# ------------------------

def train_epoch(model, loader, optimizer, epoch, total_epochs) -> Dict:
    model.train()
    totals = {"loss": 0.0, "rec": 0.0, "kl": 0.0, "psnr": 0.0}
    n_batches = 0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        if getattr(model, "is_vae", False):
            recon, z, mu, logvar = model(noisy)
            loss, rec, kl = vae_loss(recon, clean, mu, logvar, beta_kl=1e-3)
        else:
            recon, z = model(noisy)
            loss = mse_loss(recon, clean)
            rec = loss.detach()
            kl = torch.tensor(0.0)

        loss.backward()
        optimizer.step()

        totals["loss"] += loss.item()
        totals["rec"] += rec.item()
        totals["kl"] += kl.item()
        totals["psnr"] += psnr(recon.detach(), clean)
        n_batches += 1

    for k in totals.keys():
        totals[k] /= n_batches

    print(f"[Train] Epoch {epoch}/{total_epochs} | "
          f"loss={totals['loss']:.6f} | rec={totals['rec']:.6f} | "
          f"kl={totals['kl']:.6f} | psnr={totals['psnr']:.2f} dB")

    return totals


def eval_epoch(model, loader, epoch, total_epochs) -> Dict:
    model.eval()
    totals = {"loss": 0.0, "rec": 0.0, "kl": 0.0, "psnr": 0.0}
    n_batches = 0

    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            if getattr(model, "is_vae", False):
                recon, z, mu, logvar = model(noisy)
                loss, rec, kl = vae_loss(recon, clean, mu, logvar, beta_kl=1e-3)
            else:
                recon, z = model(noisy)
                loss = mse_loss(recon, clean)
                rec = loss.detach()
                kl = torch.tensor(0.0)

            totals["loss"] += loss.item()
            totals["rec"] += rec.item()
            totals["kl"] += kl.item()
            totals["psnr"] += psnr(recon, clean)
            n_batches += 1

    for k in totals.keys():
        totals[k] /= n_batches

    print(f"[Valid] Epoch {epoch}/{total_epochs} | "
          f"loss={totals['loss']:.6f} | rec={totals['rec']:.6f} | "
          f"kl={totals['kl']:.6f} | psnr={totals['psnr']:.2f} dB")

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
        if getattr(model, "is_vae", False):
            recon_batch, _, _, _ = model(noisy_batch)
        else:
            recon_batch, _ = model(noisy_batch)

    noisy = noisy_batch.cpu().numpy()
    clean = clean_batch.cpu().numpy()
    recon = recon_batch.cpu().numpy()

    # 3 rows × n_samples: Noisy / Denoised / Clean
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

    full_train = NoisyMNISTHighRes(root="data", train=True, noise_std=0.4)

    train_size = int(0.9 * len(full_train))
    val_size = len(full_train) - train_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    model = create_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 25
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
