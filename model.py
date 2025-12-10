# model.py
import math
import torch
from torch import nn

# ------------------------
# Global config
# ------------------------
IMG_SIZE = 64          # 64 is fine for MNIST-style digits
IN_CHANNELS = 1        # grayscale
LATENT_DIM = 256       # size of latent vector shown in UI
MODEL_TYPE = "ae"      # we now use a deterministic denoising AE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _check_img_size(img_size: int):
    if 2 ** int(math.log2(img_size)) != img_size:
        raise ValueError("IMG_SIZE must be power-of-two (32, 64, 128, ...)")
    if img_size < 32:
        raise ValueError("IMG_SIZE should be >= 32 for this architecture.")


# ------------------------
# Basic building blocks
# ------------------------

class ConvBlock(nn.Module):
    """
    Conv -> BN -> ReLU (×2)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------
# U-Net style denoiser
# ------------------------

class UNetDenoiser(nn.Module):
    """
    Lightweight U-Net for denoising digits.

    Input:  (B, 1, H, W)
    Output: (B, 1, H, W)  in [0,1]
    Latent: (B, LATENT_DIM)
    """

    def __init__(self,
                 img_size: int = IMG_SIZE,
                 in_channels: int = IN_CHANNELS,
                 base_channels: int = 32,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        _check_img_size(img_size)

        self.img_size = img_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.is_vae = False   # so train.py knows we are plain AE

        # ------- Encoder -------
        self.enc1 = ConvBlock(in_channels, base_channels)         # 64x64 -> 64x64
        self.down1 = nn.Conv2d(base_channels, base_channels * 2,
                               kernel_size=3, stride=2, padding=1)  # 64 -> 32

        self.enc2 = ConvBlock(base_channels * 2, base_channels * 2)  # 32x32
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4,
                               kernel_size=3, stride=2, padding=1)  # 32 -> 16

        self.enc3 = ConvBlock(base_channels * 4, base_channels * 4)  # 16x16
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 8,
                               kernel_size=3, stride=2, padding=1)  # 16 -> 8

        # bottleneck 8x8
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 8)

        # latent from global average pooled bottleneck
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_latent = nn.Linear(base_channels * 8, latent_dim)

        # ------- Decoder -------
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4,
                                      kernel_size=3, stride=2, padding=1, output_padding=1)  # 8 -> 16
        self.dec3 = ConvBlock(base_channels * 4 + base_channels * 4, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                                      kernel_size=3, stride=2, padding=1, output_padding=1)  # 16 -> 32
        self.dec2 = ConvBlock(base_channels * 2 + base_channels * 2, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels,
                                      kernel_size=3, stride=2, padding=1, output_padding=1)  # 32 -> 64
        self.dec1 = ConvBlock(base_channels + base_channels, base_channels)

        # final prediction
        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        self.final_act = nn.Sigmoid()

    def encode(self, x: torch.Tensor):
        # encoder with skip connections
        x1 = self.enc1(x)                  # (B, 32, 64, 64)
        x2_in = torch.relu(self.down1(x1))
        x2 = self.enc2(x2_in)              # (B, 64, 32, 32)

        x3_in = torch.relu(self.down2(x2))
        x3 = self.enc3(x3_in)              # (B, 128, 16, 16)

        x4_in = torch.relu(self.down3(x3))
        bottleneck = self.bottleneck(x4_in)  # (B, 256, 8, 8)

        # latent vector via global avg pool
        gap = self.gap(bottleneck).view(bottleneck.size(0), -1)  # (B, 256)
        z = self.fc_latent(gap)                                  # (B, LATENT_DIM)

        return (x1, x2, x3, bottleneck), z

    def decode(self, skips, z: torch.Tensor):
        x1, x2, x3, bottleneck = skips

        # decoder with skip concatenations
        d3 = self.up3(bottleneck)          # (B, 128, 16, 16)
        d3 = torch.cat([d3, x3], dim=1)    # concat skip
        d3 = self.dec3(d3)

        d2 = self.up2(d3)                  # (B, 64, 32, 32)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                  # (B, 32, 64, 64)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        out = self.final_act(out)
        return out

    def forward(self, x: torch.Tensor):
        skips, z = self.encode(x)
        recon = self.decode(skips, z)
        return recon, z


# ------------------------
# Factory
# ------------------------

def create_model():
    # we keep the same API: either VAE or "not VAE"
    # but for quality, we just use UNetDenoiser (is_vae = False)
    model = UNetDenoiser(
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        base_channels=32,
        latent_dim=LATENT_DIM,
    ).to(device)
    return model
