import lightning as L
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import itertools

from utils.train_options import TrainOptions


class AdversarialTraining(L.LightningModule):
    def __init__(
            self, 
            critic: nn.Module, 
            generator: nn.Module,
            encoder: nn.Module, 
            opt: TrainOptions,
    ):
        super().__init__()
        
        self.automatic_optimization = False
        self.critic = critic
        self.generator = generator
        self.encoder = encoder
        
        self.critic.apply(self._initialize_weights)
        self.generator.apply(self._initialize_weights)
        self.encoder.apply(self._initialize_weights)

        self.opt = opt 
        self.fixed_latents = torch.randn(opt.batch_size, opt.latent_dim, 1, 1)

    def _initialize_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            nn.init.normal_(m.weight.data, 0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def configure_optimizers(self):
        ae_optimizer = torch.optim.Adam(params=itertools.chain(self.encoder, self.generator), lr=self.opt.lr, betas=self.opt.betas)
        crititc_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.opt.lr, betas=self.opt.betas)
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=self.opt.betas)
        return [ae_optimizer, crititc_optimizer, generator_optimizer]
    
    def __log_images(self, n_generated_imgs: int = 32):    
        generated_imgs = self.generator(self.fixed_latents.to(self.device))
        
        # Convert to proper format and normalize
        grid = vutils.make_grid(generated_imgs[:n_generated_imgs], padding=2, normalize=True).cpu()
        
        self.logger.log_image(
            key="generated_progress",
            images=[grid],
            caption=[f"epoch {self.current_epoch}"],
            step=self.current_epoch
        )
    
    def on_train_epoch_end(self):
        self.__log_images()

    def __log_gradients(self):
        g_grad = [p.grad.view(-1).cpu().numpy() for p in list(self.generator.parameters())]
        d_grad = [p.grad.view(-1).cpu().numpy() for p in list(self.critic.parameters())]
        
        grad_history = {
            'g_grads_mean': np.concatenate(g_grad).mean().item(),
            'g_grads_std': np.concatenate(g_grad).std().item(),
            'd_grads_mean': np.concatenate(d_grad).mean(),
            'd_grads_std': np.concatenate(d_grad).std()
        }
        self.log_dict(grad_history, prog_bar=True)

    def _critic_loss(self, x_real: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        raise NotImplementedError

    def _generator_loss(self, x_real: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        raise NotImplementedError

    def _ae_loss(self, x_real: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        return None

    def _generator_step(self, x, optimizer, criterion, log_history):
        pass

    def _critic_step(self, x, optimizer, criterion):
        optimizer.zero_grad()
        critic_loss = self._critic_loss(x, criterion)
        self.manual_backward(critic_loss)
        optimizer.step()
        return critic_loss 

    def _ae_step(self, x, optimizer, criterion, log_history):
        pass

    def training_step(self, batch, batch_idx):
        ae_optimizer, critic_optimizer, generator_optimizer = self.optimizers()
        x_real, _ = batch
        criterion = nn.BCEWithLogitsLoss()
        history = {}

        # AE stage
        ae_loss = self._ae_loss(x_real, nn.MSELoss)
        if ae_loss is not None:
            pass
        else:
            ae_optimizer = None

        # Critic's train
        for _ in range(self.opt.n_critic_steps):
            critic_loss = self._critic_step(x_real, critic_optimizer, criterion)

        # Generator's train
        generator_optimizer.zero_grad()
        gen_loss = self._generator_loss(x_real, criterion)
        self.manual_backward(gen_loss)
        generator_optimizer.step()

        history['loss_g'] = gen_loss.item()
        history['loss_d'] = critic_loss.item()
        
        self.log_dict(history, prog_bar=True)
        self.__log_gradients()
        return history  