import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import eval_genlaguerre as L


def compute_kld(mu, tau, d):
    """Compute KL divergence for the tilted prior."""
    return -tau * np.sqrt(np.pi / 2) * L(1 / 2, d / 2 - 1, -(mu**2) / 2) + (mu**2) / 2


def compute_gamma(tau, d):
    """
    Compute optimal gamma (||mu||) that minimizes KL divergence
    between the approximate posterior and the tilted prior.

    Args:
        tau: tilt parameter controlling concentration of the prior
        d: latent dimension
    Returns:
        gamma: optimal norm of mu
    """
    steps = [1e-1, 1e-2, 1e-3, 1e-4]
    dx = 5e-3

    # initial guess (close to optimal value)
    x = np.sqrt(max(tau**2 - d, 0))

    # gradient descent (kld is convex)
    for step in steps:
        for _ in range(10000):
            y1 = compute_kld(x - dx / 2, tau, d)
            y2 = compute_kld(x + dx / 2, tau, d)
            grad = (y2 - y1) / dx
            x -= grad * step

    return x


class TiltedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, gamma=None):
        """
        Encoder for the Tilted VAE.

        Args:
            input_dim: input feature dimension
            hidden_dim: hidden layer dimension
            latent_dim: latent space dimension
            gamma: if not None, logvar is fixed to zero (tilted prior mode)
        """
        super().__init__()
        self.gamma = gamma
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        if gamma is None:
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        if self.gamma is None:
            logvar = self.fc_logvar(h)
        else:
            logvar = torch.zeros_like(mu)
        return mu, logvar


class TiltedDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc_out(h))


class TiltedVAE(nn.Module):
    """
    Variational Autoencoder with Exponentially Tilted Gaussian Prior.

    Based on: Griffin Floto, Stefan Kremer and Mihai Nica,
    "Exponentially Tilted Gaussian Prior for Variational Autoencoder",
    arXiv:2111.15646, 2021.

    The tilted prior concentrates probability mass on a sphere of radius gamma
    in the latent space, which improves OOD detection by separating in-distribution
    encodings from out-of-distribution ones.

    Args:
        input_dim: input feature dimension
        hidden_dim: hidden layer dimension
        latent_dim: latent space dimension
        tilt (float or None): tau parameter for the tilted prior.
            If None, behaves as a standard VAE with learned variance.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, tilt=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.tilt = tilt

        if tilt is not None:
            self.gamma = compute_gamma(tilt, latent_dim)
        else:
            self.gamma = None

        self.encoder = TiltedEncoder(input_dim, hidden_dim, latent_dim, self.gamma)
        self.decoder = TiltedDecoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def loss_function(self, x, x_recon, mu, logvar, beta=1.0):
        """
        Compute the tilted VAE loss.

        Args:
            x: input data
            x_recon: reconstructed data
            mu: encoder mean
            logvar: encoder log-variance
            beta: weight for KL term (for beta-VAE style training)
        Returns:
            total loss, reconstruction loss, KL divergence loss
        """
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

        if self.gamma is None:
            # Standard VAE KL divergence
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            # Tilted prior KL divergence
            mu_norm = torch.linalg.norm(mu, dim=1)
            kld = 0.5 * torch.sum(torch.square(mu_norm - self.gamma))

        total_loss = recon_loss + beta * kld
        return total_loss, recon_loss, kld

    def sample(self, num_samples, device='cpu'):
        """
        Sample from the tilted prior.

        For the tilted prior, samples are drawn on a sphere of radius gamma
        rather than from a standard Gaussian.
        """
        if self.gamma is None:
            z = torch.randn(num_samples, self.latent_dim).to(device)
        else:
            # Sample from tilted prior: points on sphere of radius gamma
            z = torch.randn(num_samples, self.latent_dim).to(device)
            z = F.normalize(z, dim=1) * self.gamma
        return self.decoder(z)
