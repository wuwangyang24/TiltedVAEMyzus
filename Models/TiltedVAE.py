import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing import List

from scipy.special import eval_genlaguerre


def _kld_radial(mu_norm: float, tau: float, d: int) -> float:
    """KL divergence of the tilted prior as a function of the posterior-mean
    radius ``mu_norm`` (Floto et al., 2021). The optimal radius does not depend
    on the sampled ``z``, so it can be found by 1D minimisation over this term.

    Uses the generalised Laguerre function L_{1/2}^{d/2-1}.
    """
    return (-tau * math.sqrt(math.pi / 2)
            * eval_genlaguerre(0.5, d / 2 - 1, -(mu_norm ** 2) / 2)
            + (mu_norm ** 2) / 2)


def kld_min(tau: float, d: int) -> float:
    """Find the posterior-mean radius ``gamma`` that minimises the KL divergence
    to the exponentially tilted Gaussian prior with tilt ``tau`` in ``d``
    dimensions, via gradient descent (the objective is convex in the radius).
    """
    steps = [1e-1, 1e-2, 1e-3, 1e-4]
    dx = 5e-3

    # Initial guess (very close to the optimal value).
    x = math.sqrt(max(tau ** 2 - d, 0.0))

    for step in steps:
        for _ in range(10000):
            y1 = _kld_radial(x - dx / 2, tau, d)
            y2 = _kld_radial(x + dx / 2, tau, d)
            grad = (y2 - y1) / dx
            x -= grad * step
    return x


class TiltedVAE(nn.Module):
    """
    Tilted Convolutional Variational Autoencoder for 96x96 images.

    Replaces the standard N(0, I) prior of the vanilla VAE with the
    exponentially tilted Gaussian prior of Floto, Kremer & Nica
    (arXiv:2111.15646), which concentrates probability mass on the surface of a
    hyper-sphere of radius ``gamma`` in latent space. Following that work, the
    approximate posterior uses a fixed unit variance (``log_var = 0``), so the
    encoder only predicts the mean and the KL term reduces to a simple radial
    penalty ``0.5 * (||mu|| - gamma)^2``.

    The encoder / decoder convolutional stacks are identical in depth and width
    to the vanilla ``VAE`` in ``VAE.py`` (hidden dims [32, 64, 128, 256, 512],
    BatchNorm + LeakyReLU, five stride-2 downsamples for a 96x96 input).

    Args:
        in_channels: number of image channels (1 for grayscale, 3 for RGB)
        latent_dim: dimension of the latent space
        tau: tilt parameter of the prior. If None, defaults to sqrt(2 * latent_dim),
            which places the prior mode at radius ~sqrt(latent_dim).
        hidden_dims: list of channel widths for the encoder conv stack
        img_size: spatial size of the (square) input image
    """

    def __init__(self,
                 in_channels: int = 3,
                 latent_dim: int = 128,
                 tau: float = None,
                 hidden_dims: List = None,
                 img_size: int = 96) -> None:
        super(TiltedVAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels

        # Tilt parameter and the corresponding optimal posterior-mean radius.
        if tau is None:
            tau = math.sqrt(2 * latent_dim)
        self.tau = float(tau)
        self.gamma = kld_min(self.tau, latent_dim)

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
        # The tilted posterior has a fixed unit variance, so only the mean is
        # predicted (no fc_var head).
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)

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
        Encodes the input by passing through the encoder network and returns the
        latent mean. The tilted posterior uses a fixed unit variance, so
        ``log_var`` is returned as zeros for interface compatibility.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of [mu, log_var]
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = torch.zeros_like(mu)

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
        Computes the tilted VAE loss function.

        The KL divergence to the exponentially tilted Gaussian prior reduces, for
        a unit-variance posterior, to a penalty pulling the posterior-mean norm
        towards the prior radius ``gamma``:
            KL = 0.5 * (||mu|| - gamma)^2
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # Account for the minibatch samples from the dataset.
        kld_weight = kwargs.get('M_N', 1.0)
        recons_loss = F.mse_loss(recons, input)

        # Radial KL penalty towards the tilted-prior hyper-sphere of radius gamma.
        mu_norm = torch.norm(mu, dim=1)
        kld_loss = 0.5 * torch.mean((mu_norm - self.gamma) ** 2)

        # The tilted KL is a single radial term rather than a per-dimension sum;
        # distribute it uniformly across dims for interface compatibility so that
        # KLD_per_dim.sum() == total KLD.
        kld_per_dim = torch.full(
            (self.latent_dim,),
            (kld_loss.detach() / self.latent_dim).item(),
            device=mu.device)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss.detach(),
                'KLD': kld_loss.detach(),
                'KLD_per_dim': kld_per_dim}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the tilted prior and returns the corresponding image space
        map. Samples are drawn with uniform direction on the hyper-sphere of
        radius ``gamma`` (the mode of the tilted radial distribution).
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        u = torch.randn(num_samples, self.latent_dim)
        u = u.to(current_device)
        z = self.gamma * u / torch.norm(u, dim=1, keepdim=True)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image.
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]
