import torch
import torch.nn as nn
from torch.types import Tensor

from torch.nn import (
    Linear, Conv2d, ConvTranspose2d,
    ReLU, Sigmoid,
    Flatten, Unflatten,
)
    
class VAE(nn.Module):
    """Convolutional Variational Autoencoder (VAE)

    Attrs:
        Encoder     : Encoder model.
        Decoder     : Decoder model.
        FC_mu       : Fully-connected layer for latent space mu.
        FC_logvar   : Fully-connected layer for latent space logvar.
    """
    def __init__(self, in_channels: int, latent_dim: int):
        """
        Args:
            in_channels : number of input image channels.
            latent_dim  : number of latent dimensions.
        """
        super(VAE, self).__init__()

        self.Encoder = nn.Sequential(
            Conv2d(in_channels, 32, 3, 2, 1),
            ReLU(),
            Conv2d(32, 64, 3, 2, 1),
            ReLU(),
            Flatten(),
        )

        # separate mu and logvarvariance layers
        # unbounded, no activation
        self.FC_mu = Linear(7*7*64, latent_dim)
        self.FC_logvar = Linear(7*7*64, latent_dim)

        self.Decoder = nn.Sequential(
            Linear(latent_dim, 7*7*64),
            Unflatten(1, (64, 7, 7)),
            ReLU(),
            ConvTranspose2d(64, 64, 3, 2, 1, output_padding=1),
            ReLU(),
            ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            ReLU(),
            ConvTranspose2d(32, in_channels, 3, 1, 1),
            Sigmoid()
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode, Sample, and Decode input image x.

        *mu and logvar are returned for loss computation.

        Args:
            x : input image.

        Returns:
            tuple[Tensor, Tensor, Tensor]: reconstruction, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_recon = self.Decoder(z)
        return x_recon, mu, logvar

    def sample(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample mulivariate Gaussian latent vector.
         
        z = mu + std * eps

        Args:
            mu     : mean vector.
            logvar : log(variance).

        Returns:
            Tensor : sampled latent vector.
        """
        eps = torch.randn_like(logvar)
        return mu + torch.exp(0.5*logvar)*eps

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input image x into latent space.

        Args:
            x : input image.

        Returns:
            tuple[Tensor, Tensor] : corresponding mu and logvar in the latent space.
        """
        x = self.Encoder(x)
        return self.FC_mu(x), self.FC_logvar(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode a latent space vector z.

        Args:
            z : latent space vector.

        Returns:
            Tensor : reconstructed image.
        """
        return self.Decoder(z)

