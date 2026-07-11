import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing import List


class VAE(nn.Module):
    """
    Vanilla Convolutional Variational Autoencoder for 96x96 images.

    Adapted from the PyTorch-VAE VanillaVAE implementation. The reference
    targets 64x64 inputs; here the encoder downsamples a 96x96 image five
    times (96 -> 48 -> 24 -> 12 -> 6 -> 3), so the flattened feature map is
    hidden_dims[-1] * 3 * 3 and the decoder reshapes accordingly.

    Args:
        in_channels: number of image channels (1 for grayscale, 3 for RGB)
        latent_dim: dimension of the latent space
        hidden_dims: list of channel widths for the encoder conv stack
        img_size: spatial size of the (square) input image
    """

    def __init__(self,
                 in_channels: int = 3,
                 latent_dim: int = 128,
                 hidden_dims: List = None,
                 img_size: int = 96) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims.copy()

        # Spatial size of the feature map after the encoder
        # (one stride-2 downsample per hidden dim).
        self.final_size = img_size // (2 ** len(hidden_dims))  # 96 // 32 = 3
        self.flatten_dim = hidden_dims[-1] * self.final_size * self.final_size

        # Build Encoder
        enc_in = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(enc_in, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            enc_in = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_var = nn.Linear(self.flatten_dim, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        rev_dims = hidden_dims[::-1]

        for i in range(len(rev_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(rev_dims[i],
                                       rev_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(rev_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(rev_dims[-1],
                               rev_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(rev_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(rev_dims[-1], out_channels=in_channels,
                      kernel_size=3, padding=1),
            nn.Sigmoid())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.final_size, self.final_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Log variance of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(mu, sigma), N(0, 1)) = log(1/sigma) + (sigma^2 + mu^2)/2 - 1/2
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # Account for the minibatch samples from the dataset
        kld_weight = kwargs.get('M_N', 1.0)
        recons_loss = F.mse_loss(recons, input)

        # Per-dimension KL divergence, averaged over the batch -> [latent_dim].
        kld_per_dim = torch.mean(
            -0.5 * (1 + log_var - mu ** 2 - log_var.exp()), dim=0)
        # Total KL divergence (summed over dims, averaged over batch).
        kld_loss = torch.sum(kld_per_dim)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss.detach(),
                'KLD': kld_loss.detach(),
                'KLD_per_dim': kld_per_dim.detach()}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and returns the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image.
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]
