import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv = ConvBlock(out_ch, out_ch)

    def forward(self, x):
        x = self.down(x)
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch):
        """
        in_ch  = channels of input feature (from deeper level)
        out_ch = channels after upsample + conv
        skip_ch = channels of skip connection feature
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # If shapes ever misalign by 1 px (odd sizes), pad to match skip
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            diff_y = skip.shape[-2] - x.shape[-2]
            diff_x = skip.shape[-1] - x.shape[-1]
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2]
            )

        # concat along channel dim
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.in_conv = ConvBlock(1, 32)    # 1x28x28 -> 32x28x28
        self.down1   = Down(32, 64)        # 32x28x28 -> 64x14x14
        self.down2   = Down(64, 128)       # 64x14x14 -> 128x7x7

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)  # 128x7x7 -> 256x7x7

        # Decoder (note skip_ch sizes!)
        self.up1 = Up(in_ch=256, out_ch=128, skip_ch=64)  # skip x2 (64ch, 14x14)
        self.up2 = Up(in_ch=128, out_ch=64,  skip_ch=32)  # skip x1 (32ch, 28x28)

        # Output head
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        # Global latent vector (for fun / future use)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, return_latent=False):
        # Encoder
        x1 = self.in_conv(x)   # [B, 32, 28, 28]
        x2 = self.down1(x1)    # [B, 64, 14, 14]
        x3 = self.down2(x2)    # [B, 128, 7, 7]

        # Bottleneck
        b = self.bottleneck(x3)  # [B, 256, 7, 7]

        # Latent vector
        latent = self.global_pool(b).view(b.size(0), -1)  # [B, 256]

        # Decoder (proper skip levels)
        u1 = self.up1(b, x2)   # [B, 128, 14, 14]
        u2 = self.up2(u1, x1)  # [B, 64,  28, 28]

        out = self.out_conv(u2)  # [B, 1, 28, 28]

        if return_latent:
            return out, latent
        return out
