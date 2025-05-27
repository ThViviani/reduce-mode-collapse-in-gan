import torch
import torch.nn as nn

from trainers.standard_gan import StandardGAN
from trainers.rp_gan import RpGAN


class NeighborsEmbeddingMixin:
    def _ae_loss(self, x_real: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        x_hat = self.generator(self.encoder(x_real))
        ae_loss = criterion(x_hat, x_real)
        lambda_r = self.opt.latent_distance_penalty
        regularization_loss = self._regularization_loss(x_real)
        self.log_dict({'kl_loss': regularization_loss}, prog_bar=True)
        return ae_loss + lambda_r * regularization_loss 

    def _regularization_loss(self, x_real: torch.Tensor):
        z = self._sample_z(batch_size=x_real.shape[0])
        z_hat = self.encoder(x_real)
        z_batch = torch.concat([z, z_hat]) # [B_size * 2, latent_dim]

        x_hat = self.generator(z)
        x_rec = self.generator(z_hat)
        x_batch = torch.concat([x_hat, x_rec]).flatten(start_dim=1) # [B_size * 2, 1 * 28 * 28]

        P_joint = self._joint_probabilities(z_batch)
        Q_joint = self._joint_probabilities(x_batch)

        eps = torch.finfo(x_real.dtype).eps
        kl_loss = torch.sum(P_joint * torch.log((P_joint + eps) / (Q_joint + eps)))
        return kl_loss

    def _joint_probabilities(self, batch: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(batch.dtype).eps
        D = torch.cdist(batch, batch)
        q_cond = (1.0 + D / (2 * D.var() + eps)) ** -1
        q_cond.fill_diagonal_(0.0)
        q_cond /= q_cond.sum(dim=1, keepdim=True) + eps
        
        n = batch.shape[0]
        q_symmetric = (q_cond + q_cond.T) / (2 * n)
        return q_symmetric


class NEVanilaGAN(NeighborsEmbeddingMixin, StandardGAN): pass

class NERpGAN(NeighborsEmbeddingMixin, RpGAN): pass