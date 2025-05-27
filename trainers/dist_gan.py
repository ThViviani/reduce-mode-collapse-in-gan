import torch
import torch.nn as nn
import numpy as np

from trainers.standard_gan import StandardGAN
from trainers.rp_gan import RpGAN


class DistMixin:
    def _ae_loss(self, x_real: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        x_hat = self.generator(self.encoder(x_real))
        ae_loss = criterion(x_hat, x_real)
        lambda_r = self.opt.latent_distance_penalty
        return ae_loss + lambda_r * self._regularization_loss(x_real)

    def _regularization_loss(self, x_real: torch.Tensor):
        z = self._sample_z(batch_size=x_real.size(0))
        md_x = torch.mean(self.generator(self.encoder(x_real)) - self.generator(z))
        lambda_w = np.sqrt(self.opt.latent_dim / (x_real.shape[-1]**2))
        md_z = torch.mean(self.encoder(x_real) - z) * lambda_w
        reg_loss = (md_x - md_z).square()
        return reg_loss
    
class DistVanilaGAN(DistMixin, StandardGAN): pass

class DistRpGAN(DistMixin, RpGAN): pass