import lightning as L
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np

from utils.train_options import TrainOptions
from trainers.adversarial_trainer import AdversarialTraining


class RpGAN(AdversarialTraining):
    """
    Relativistic paired GAN implementation using zero-centered R1/R2 gradient penalty.
    """
    def __init__(
        self,
        critic: nn.Module,
        generator: nn.Module,
        encoder: nn.Module | None,
        opt: TrainOptions,
        use_r1r2_penalty: bool = False,
    ):
        super().__init__(critic=critic, generator=generator, encoder=encoder, opt=opt)
        self.use_grad_penalty = use_r1r2_penalty

    @staticmethod
    def zero_centered_gradient_penalty(critics_out: torch.Tensor, critics_in: torch.Tensor) -> torch.Tensor:
        grads, = torch.autograd.grad(
            outputs=critics_out.sum(), inputs=critics_in, create_graph=True
        )
        # Sum of squares over channel, height, width dims, then mean over batch
        return grads.pow(2).sum(dim=[1, 2, 3]).mean()

    def _critic_loss(self, x_real: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        x_real.requires_grad_(True)
        critic_real = self.critic(x_real)

        z = torch.randn(x_real.size(0), self.opt.latent_dim, device=self.device)
        x_fake = self.generator(z)
        x_fake.requires_grad_(True)
        critic_fake = self.critic(x_fake.detach())

        real_vs_fake = critic_real - critic_fake
        loss_d = criterion(real_vs_fake, torch.ones_like(critic_real, device=self.device))

        if self.use_grad_penalty:
            r1 = RpGAN.zero_centered_gradient_penalty(critic_real, x_real)
            r2 = RpGAN.zero_centered_gradient_penalty(self.critic(x_fake), x_fake)
            loss_d = loss_d + self.opt.gamma_gp / 2 * (r1 + r2)

        return loss_d

    def _generator_loss(self, x_real: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        critic_real = self.critic(x_real)

        z = torch.randn(x_real.size(0), self.opt.latent_dim, device=self.device)
        critic_fake = self.critic(self.generator(z))

        real_vs_fake = critic_fake - critic_real
        loss_g = criterion(real_vs_fake, torch.ones_like(critic_real, device=self.device))
        return loss_g
