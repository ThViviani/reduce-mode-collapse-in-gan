import torch
import torch.nn as nn

from trainers.adversarial_trainer import AdversarialTraining


class StandardGAN(AdversarialTraining):
    def _critic_loss(self, x_real: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        critic_real = self.critic(x_real)
        real_err = criterion(critic_real, torch.ones_like(critic_real, device=self.device))
        
        z = self._sample_z(batch_size=x_real.size(0))
        x_fake = self.generator(z)
        critic_fake = self.critic(x_fake.detach())
        fake_err = criterion(critic_fake, torch.zeros_like(critic_fake, device=self.device))
        return real_err + fake_err      
        
    def _generator_loss(self, x_real: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        z = self._sample_z(batch_size=x_real.size(0))
        x_fake = self.generator(z)
        critic_fake = self.critic(x_fake)
        gen_loss = criterion(critic_fake, torch.ones_like(critic_fake, device=self.device))
        return gen_loss