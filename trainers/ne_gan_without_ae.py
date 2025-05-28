import torch
import torch.nn as nn

from trainers.standard_gan import StandardGAN
from trainers.rp_gan import RpGAN


class NeighborsEmbeddingMixin_hat:
    def _generator_loss(self, x_real: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        base_loss = super()._generator_loss(x_real, criterion)
        lambda_r = self.opt.latent_distance_penalty
        regularization_loss = self._regularization_loss(x_real)
        self.log_dict({'kl_loss': regularization_loss}, prog_bar=True)
        return base_loss + lambda_r * regularization_loss 

    def _regularization_loss(self, x_real: torch.Tensor):
        z = torch.randn(x_real.shape[0], self.opt.latent_dim, device=self.device)
        x_hat = self.generator(z).flatten(start_dim=1)

        P_joint = self._joint_probabilities(z)
        Q_joint = self._joint_probabilities(x_hat)

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


class NEVanilaGAN_hat(NeighborsEmbeddingMixin_hat, StandardGAN): pass
class NERpGAN_hat(NeighborsEmbeddingMixin_hat, RpGAN): pass