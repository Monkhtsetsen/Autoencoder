# model.py
import math
import torch
from torch import nn

# ------------------------
# Global config
# ------------------------
IMG_SIZE = 64          # 64 or 128 (must be power-of-two)
IN_CHANNELS = 1        # 1 = grayscale
LATENT_DIM = 128       # bottleneck vector length
MODEL_TYPE = "vae"     # "vae" эсвэл "ae"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _check_img_size(img_size: int):
    if 2 ** int(math.log2(img_size)) != img_size:
        raise ValueError("IMG_SIZE must be power-of-two (32, 64, 128, ...)")
    if img_size < 32:
        raise ValueError("IMG_SIZE should be >= 32 for this architecture.")


def _make_encoder(in_channels: int, base_channels: int, img_size: int):
    """
    Convolutional encoder: img_size -> 4x4 хүртэл downsample.
    Returns (nn.Sequential, last_channels, feature_size)
    """
    _check_img_size(img_size)

    # final feature map size 4x4 болтол stride=2 conv хийнэ
    num_down = int(math.log2(img_size) - 2)  # e.g. 64 -> 4 downs, 128 -> 5 downs
    layers = []
    ch = in_channels
    for i in range(num_down):
        out_ch = base_channels * (2 ** i)
        layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        ch = out_ch

    encoder = nn.Sequential(*layers)
    feat_size = img_size // (2 ** num_down)  # should be 4
    return encoder, ch, feat_size


def _make_decoder(last_ch: int, base_channels: int, img_size: int, out_channels: int):
    """
    Transposed-conv decoder: 4x4 -> img_size
    """
    _check_img_size(img_size)
    num_up = int(math.log2(img_size) - 2)

    layers = []
    ch = last_ch
    for i in reversed(range(num_up)):
        out_ch = base_channels * (2 ** i)
        layers.append(
            nn.ConvTranspose2d(
                ch,
                out_ch,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        )
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        ch = out_ch

    # final conv to 1 channel, sigmoid [0,1]
    layers.append(nn.Conv2d(ch, out_channels, kernel_size=3, padding=1))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


# ------------------------
# Plain Denoising Autoencoder
# ------------------------

class DenoisingAutoencoder(nn.Module):
    def __init__(self,
                 img_size: int = IMG_SIZE,
                 in_channels: int = IN_CHANNELS,
                 latent_dim: int = LATENT_DIM,
                 base_channels: int = 32):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.is_vae = False

        self.encoder_conv, last_ch, feat_size = _make_encoder(
            in_channels, base_channels, img_size
        )
        self.feature_shape = (last_ch, feat_size, feat_size)
        self.flat_dim = last_ch * feat_size * feat_size

        self.fc_enc = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)

        self.decoder_conv = _make_decoder(last_ch, base_channels, img_size, in_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        h_flat = h.view(h.size(0), -1)
        z = self.fc_enc(h_flat)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h_flat = self.fc_dec(z)
        h = h_flat.view(-1, *self.feature_shape)
        out = self.decoder_conv(h)
        return out

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        out = self.decode(z)
        return out, z


# ------------------------
# Denoising Variational Autoencoder (VAE)
# ------------------------

class DenoisingVAE(nn.Module):
    """
    Conv-based Denoising VAE.
      Input:  (B, 1, H, W)
      Latent: (B, LATENT_DIM)
      Output: (B, 1, H, W)
    """

    def __init__(self,
                 img_size: int = IMG_SIZE,
                 in_channels: int = IN_CHANNELS,
                 latent_dim: int = LATENT_DIM,
                 base_channels: int = 32):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.is_vae = True

        self.encoder_conv, last_ch, feat_size = _make_encoder(
            in_channels, base_channels, img_size
        )
        self.feature_shape = (last_ch, feat_size, feat_size)
        self.flat_dim = last_ch * feat_size * feat_size

        # q(z | x)
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        # p(x | z)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.decoder_conv = _make_decoder(last_ch, base_channels, img_size, in_channels)

    def encode(self, x: torch.Tensor):
        h = self.encoder_conv(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h_flat = self.fc_dec(z)
        h = h_flat.view(-1, *self.feature_shape)
        out = self.decoder_conv(h)
        return out

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, z, mu, logvar


# ------------------------
# Factory
# ------------------------

def create_model():
    if MODEL_TYPE.lower() == "vae":
        return DenoisingVAE(
            img_size=IMG_SIZE,
            in_channels=IN_CHANNELS,
            latent_dim=LATENT_DIM,
        ).to(device)
    else:
        return DenoisingAutoencoder(
            img_size=IMG_SIZE,
            in_channels=IN_CHANNELS,
            latent_dim=LATENT_DIM,
        ).to(device)
