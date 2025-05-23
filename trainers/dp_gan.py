import torch
import torch.nn as nn

from trainers.standard_gan import StandardGAN
from trainers.rp_gan import RpGAN


class DiversityPenaltyMixin:
    def _diversity_penalty_loss(self, x_real: torch.Tensor, scale: float = 5.0) -> torch.Tensor:
        z = torch.randn(x_real.size(0), self.opt.latent_dim, device=self.device)
        G_z = nn.functional.sigmoid(scale * self._cosine_gram_matrix(z))
        
        _, features = self.critic(x_real, return_features=True)
        features = features.reshape(features.shape[0], -1)
        G_f = nn.functional.sigmoid(scale * self._cosine_gram_matrix(features))
        
        dp = G_f / (G_z + 1e-8)
        return dp.mean()

    def _cosine_gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        # features: torch.Tensor with shape: [Batch_size, embedding_dim]
        
        normilized_features = nn.functional.normalize(features, p=2, dim=1)
        G = torch.mm(normilized_features, normilized_features.t())
        return G
    
    def _compute_scale_for_dp(self):
        return 5.0



class DpVanilaGan(DiversityPenaltyMixin, StandardGAN):
    def _generator_loss(self, x_real: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        base_loss = super()._generator_loss(x_real, criterion)
        lambda_dp = self.opt.diversity_penalty
        dp_loss = self._diversity_penalty_loss(x_real, self._compute_scale_for_dp())
        g_loss = base_loss + lambda_dp * dp_loss
        self.log_dict({'dp_loss': dp_loss}, prog_bar=True)  
        return g_loss

class DpRpGAN(DiversityPenaltyMixin, RpGAN): pass
