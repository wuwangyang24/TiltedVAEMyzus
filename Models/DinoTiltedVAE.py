import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing import List

from .TiltedVAE import kld_min


class DinoTiltedVAE(nn.Module):
    """
    Hybrid DINOv2 + Tilted VAE model.

    A frozen DINOv2 ViT-S/14 backbone encodes images into 384-dimensional
    embeddings.  An MLP-based Tilted VAE then maps those embeddings to a
    structured latent space (with the exponentially tilted Gaussian prior of
    Floto, Kremer & Nica, arXiv:2111.15646) and reconstructs them.

    The reconstruction loss is computed in DINO embedding space (MSE between
    the original and reconstructed DINO embedding), not in pixel space.

    Args:
        latent_dim: dimension of the latent space.
        tau: tilt parameter of the prior. If None, defaults to
            sqrt(2 * latent_dim).
        hidden_dims: list of hidden-layer widths for the MLP encoder/decoder.
            Defaults to [512, 256].
        dino_dim: output dimension of the DINOv2 backbone (384 for ViT-S/14).
        freeze_dino: if True (default), the DINO backbone parameters are
            frozen and the model only trains the MLP encoder/decoder.
    """

    # DINOv2 ViT-S/14 output dimension.
    DINO_DIM = 384

    def __init__(self,
                 latent_dim: int = 128,
                 tau: float = None,
                 hidden_dims: List[int] = None,
                 dino_dim: int = 384,
                 freeze_dino: bool = True) -> None:
        super(DinoTiltedVAE, self).__init__()

        self.latent_dim = latent_dim
        self.dino_dim = dino_dim

        # Tilt parameter and the corresponding optimal posterior-mean radius.
        if tau is None:
            tau = math.sqrt(2 * latent_dim)
        self.tau = float(tau)
        self.gamma = kld_min(self.tau, latent_dim)

        # ---- Frozen DINOv2 backbone ----
        self.dino = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14', pretrained=True,
        )
        if freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()
        self.freeze_dino = freeze_dino

        # ImageNet normalisation applied to [0, 1] inputs before DINO.
        self.register_buffer(
            '_img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            '_img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ---- MLP encoder ----
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.hidden_dims = hidden_dims.copy()

        enc_layers: List[nn.Module] = []
        in_dim = dino_dim
        for h_dim in hidden_dims:
            enc_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(),
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*enc_layers)

        # Tilted posterior: fixed unit variance → only predict the mean.
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)

        # ---- MLP decoder ----
        dec_layers: List[nn.Module] = []
        rev_dims = hidden_dims[::-1]
        in_dim = latent_dim
        for h_dim in [rev_dims[0]] if len(rev_dims) == 1 else rev_dims:
            dec_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(),
            ])
            in_dim = h_dim
        # Final projection back to DINO embedding space (no activation —
        # DINO embeddings are unbounded real-valued vectors).
        dec_layers.append(nn.Linear(in_dim, dino_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def train(self, mode: bool = True):
        """Override to keep the DINO backbone in eval mode at all times when
        it is frozen (disables dropout / running-stats updates in BN)."""
        super().train(mode)
        if self.freeze_dino:
            self.dino.eval()
        return self

    @torch.no_grad()
    def dino_encode(self, images: Tensor) -> Tensor:
        """Pass images through the frozen DINOv2 backbone.

        Applies ImageNet normalisation to [0, 1] inputs before feeding them
        to the DINO encoder.

        Args:
            images: [B, 3, H, W] float tensor in [0, 1].
        Returns:
            [B, dino_dim] CLS-token embedding.
        """
        images = (images - self._img_mean) / self._img_std
        return self.dino(images)

    def encode(self, input: Tensor) -> List[Tensor]:
        """Encode images to the tilted latent space.

        Passes images through the frozen DINOv2 backbone, then through the
        MLP encoder to predict the posterior mean.

        :param input: (Tensor) [B, 3, H, W]
        :return: list of [mu, log_var]  (log_var is zeros)
        """
        dino_emb = self.dino_encode(input)
        h = self.encoder(dino_emb)
        mu = self.fc_mu(h)
        log_var = torch.zeros_like(mu)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """Map latent codes back to DINO embedding space.

        :param z: (Tensor) [B, latent_dim]
        :return: (Tensor) [B, dino_dim]
        """
        return self.decoder(z)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        """
        Full forward pass.

        Returns [recons_embedding, dino_embedding, mu, log_var] so the loss
        function computes reconstruction in DINO space.
        """
        dino_emb = self.dino_encode(input)
        h = self.encoder(dino_emb)
        mu = self.fc_mu(h)
        log_var = torch.zeros_like(mu)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        return [recons, dino_emb, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Tilted VAE loss in DINO embedding space.

        Reconstruction loss: MSE between original and reconstructed DINO
        embedding.  KL divergence: radial penalty towards the tilted-prior
        hyper-sphere of radius gamma.
        """
        recons = args[0]
        target = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs.get('M_N', 1.0)
        recons_loss = F.mse_loss(recons, target)

        mu_norm = torch.norm(mu, dim=1)
        kld_loss = 0.5 * torch.mean((mu_norm - self.gamma) ** 2)

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
        Samples from the tilted prior and returns DINO embeddings.
        """
        u = torch.randn(num_samples, self.latent_dim)
        u = u.to(current_device)
        z = self.gamma * u / torch.norm(u, dim=1, keepdim=True)
        return self.decode(z)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given input images, returns the reconstructed DINO embedding.
        """
        return self.forward(x)[0]
