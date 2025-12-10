# train.py
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import UNetDenoiser


# ---------------- SSIM ----------------
def ssim_index(x, y, C1=0.01**2, C2=0.03**2):
    mu_x = torch.mean(x, dim=[2, 3], keepdim=True)
    mu_y = torch.mean(y, dim=[2, 3], keepdim=True)

    sigma_x = torch.var(x, dim=[2, 3], unbiased=False, keepdim=True)
    sigma_y = torch.var(y, dim=[2, 3], unbiased=False, keepdim=True)
    sigma_xy = torch.mean((x - mu_x) * (y - mu_y), dim=[2, 3], keepdim=True)

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    return (numerator / (denominator + 1e-8)).mean()


class CombinedLoss(nn.Module):
    """
    total_loss = α*MSE + β*(1-SSIM)
    """
    def __init__(self, alpha=.5, beta=.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_val = ssim_index(pred, target)
        ssim_loss = 1 - ssim_val
        total = self.alpha * mse_loss + self.beta * ssim_loss
        return total, mse_loss, ssim_val


# -------------- Noise ---------------
def add_noise(img, sigma=0.4, sp=0.01):
    """
    Gaussian + бага зэрэг salt-and-pepper noise
    """
    g = img + torch.randn_like(img) * sigma
    rand = torch.rand_like(g)
    g[rand < sp] = 0.0
    g[rand > 1 - sp] = 1.0
    return torch.clamp(g, 0, 1)


# ---------- save samples (optional, nice to have) ----------
@torch.no_grad()
def save_samples(model, data_loader, device, epoch, save_dir, num_samples=8):
    model.eval()
    imgs, _ = next(iter(data_loader))
    imgs = imgs.to(device)[:num_samples]
    noisy = add_noise(imgs)

    preds = model(noisy)

    grid = torch.cat([imgs, noisy, preds], dim=0)
    grid = utils.make_grid(grid, nrow=num_samples, pad_value=1.0)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"samples_epoch_{epoch:03d}.png")
    utils.save_image(grid, out_path)
    print(f"   → Saved sample grid to {out_path}")


# -------------- TRAIN ---------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt_dir = "checkpoints_unet_dae"
    os.makedirs(ckpt_dir, exist_ok=True)

    # dataset (MNIST 28×28 grayscale)
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST("data", train=False, download=True, transform=transform)

    # num_workers=0 → Windows дээр илүү safe
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    # model
    model = UNetDenoiser().to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {n_params}")

    # optimizer → AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # scheduler → StepLR (PyTorch бүх хувилбар дээр ажилладаг, энгийн)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,   # 5 epoch тутам LR-ийг бууруулна
        gamma=0.5
    )

    criterion = CombinedLoss(alpha=.5, beta=.5)

    best_loss = math.inf
    epochs = 20

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_ssim = 0.0

        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            noisy = add_noise(imgs)

            preds = model(noisy)

            loss, mse_val, ssim_val = criterion(preds, imgs)

            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_mse  += mse_val.item() * bs
            total_ssim += ssim_val.item() * bs

        scheduler.step()

        train_loss = total_loss / len(train_ds)
        train_mse  = total_mse / len(train_ds)
        train_ssim = total_ssim / len(train_ds)

        print(f"[Epoch {epoch:02d}/{epochs}] "
              f"Loss={train_loss:.6f} | MSE={train_mse:.6f} | SSIM={train_ssim:.4f}")

        # save best model
        if train_loss < best_loss:
            best_loss = train_loss
            ckpt_path = os.path.join(ckpt_dir, "unet_dae_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"   → Saved BEST model to {ckpt_path}")

        # save visualization every 5 epochs
        if epoch % 5 == 0:
            save_samples(model, test_loader, device, epoch, ckpt_dir)

    print("\nTraining DONE.")


if __name__ == "__main__":
    main()
