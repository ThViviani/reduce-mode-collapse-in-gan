import torch
import torch.nn as nn
import numpy as np

from scipy.spatial.distance import pdist, squareform
from trainers.standard_gan import StandardGAN
from trainers.rp_gan import RpGAN


class NeighborsEmbeddingMixin:
    def _ae_loss(self, x_real: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        x_hat = self.generator(self.encoder(x_real))
        ae_loss = criterion(x_hat, x_real)
        lambda_r = self.opt.latent_distance_penalty
        return ae_loss + lambda_r * self._regularization_loss(x_real)

    def _regularization_loss(self, x_real: torch.Tensor):
        z = torch.randn(x_real.shape[0], self.opt.latent_dim, device=self.device)
        z_hat = self.encoder(x_real)
        z_batch = torch.concat([z, z_hat]) # [B_size * 2, latent_dim]

        x_hat = self.generator(z)
        x_rec = self.generator(z_hat)
        x_batch = torch.concat([x_hat, x_rec]).flatten(start_dim=1) # [B_size * 2, 1 * 28 * 28]

        P_joint = self._joint_probabilities(z_batch)
        Q_joint = self._joint_probabilities(x_batch)

        machine_eps = np.finfo(np.float32).eps
        kl_loss = np.sum( P_joint * np.log((P_joint + machine_eps) / (Q_joint + machine_eps)) )
        return kl_loss

    def _joint_probabilities(self, batch: torch.Tensor) -> torch.Tensor:
        D = torch.cdist(batch, batch)
        q_cond = (1.0 + D / (2 * D.var())) ** -1
        q_cond.fill_diagonal_(0.0)
        q_cond /= q_cond.sum(dim=1, keepdim=True)
        
        n = batch.shape[0]
        q_symmetric = (q_cond + q_cond.T) / (2 * n)
        return q_symmetric


class NEVanilaGAN(NeighborsEmbeddingMixin, StandardGAN): pass

class NERpGAN(NeighborsEmbeddingMixin, RpGAN): pass