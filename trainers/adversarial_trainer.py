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
            encoder: nn.Module | None = None, 
            opt: TrainOptions = TrainOptions(),
    ):
        super().__init__()
        
        self.automatic_optimization = False
        self.critic = critic
        self.generator = generator
        self.encoder = encoder
        
        for net in (self.encoder, self.critic, self.generator):
            if net is not None:
                net.apply(self._initialize_weights)
        
        self.opt = opt 
        self.fixed_latents = torch.randn(opt.batch_size, opt.latent_dim, 1, 1)

    def _initialize_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            nn.init.normal_(m.weight.data, 0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def configure_optimizers(self):
        optimizers = []
        if self.encoder is not None:
            ae_optimizer = torch.optim.Adam(
                params=itertools.chain(self.encoder.parameters(), self.generator.parameters()), 
                lr=self.opt.lr, 
                betas=self.opt.betas
            )
            optimizers.append(ae_optimizer)
        
        crititc_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.opt.lr, betas=self.opt.betas)
        optimizers.append(crititc_optimizer)
        
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=self.opt.betas)
        optimizers.append(generator_optimizer)
        return optimizers
    
    def __log_images(self, n_generated_imgs: int = 32): 
        if self.logger is None:
            return


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
        grad_history = {}
        
        g_grad = [p.grad.view(-1).cpu().numpy() for p in list(self.generator.parameters())]
        d_grad = [p.grad.view(-1).cpu().numpy() for p in list(self.critic.parameters())]

        if self.encoder is not None:
            e_grad = [p.grad.view(-1).cpu().numpy() for p in list(self.encoder.parameters())]
            grad_history['e_grads_mean'] = np.concatenate(e_grad).mean().item() 
            grad_history['e_grads_std'] = np.concatenate(e_grad).std().item()
   
        grad_history['g_grads_mean'] = np.concatenate(g_grad).mean().item()
        grad_history['g_grads_std'] = np.concatenate(g_grad).std().item()
        grad_history['d_grads_mean'] = np.concatenate(d_grad).mean()
        grad_history['d_grads_std'] = np.concatenate(d_grad).std()
            
        self.log_dict(grad_history, prog_bar=True)

    def _critic_loss(self, x_real: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        raise NotImplementedError

    def _generator_loss(self, x_real: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        raise NotImplementedError

    def _ae_loss(self, x_real: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        return None

    def _generator_step(self, x, optimizer, criterion):
        optimizer.zero_grad()
        gen_loss = self._generator_loss(x, criterion)
        self.manual_backward(gen_loss)
        optimizer.step()
        return gen_loss

    def _critic_step(self, x, optimizer, criterion):
        optimizer.zero_grad()
        critic_loss = self._critic_loss(x, criterion)
        self.manual_backward(critic_loss)
        optimizer.step()
        return critic_loss 

    def _ae_step(self, x, optimizer, criterion):
        optimizer.zero_grad()
        ae_loss = self._ae_loss(x, criterion)
        self.manual_backward(ae_loss)
        optimizer.step()
        return ae_loss

    def training_step(self, batch, batch_idx):
        if self.encoder is not None:
          ae_optimizer, critic_optimizer, generator_optimizer = self.optimizers()
        else:        
          critic_optimizer, generator_optimizer = self.optimizers()

        x_real, _ = batch
        criterion = nn.BCEWithLogitsLoss()
        history = {}

        # AE stage
        if self.encoder is not None:
            self.critic.freeze()
            ae_criterion = nn.MSELoss(reduction='mean')
            ae_loss = self._ae_step(x_real, ae_optimizer, ae_criterion)
            history['loss_ae'] = ae_loss.item()
            self.critic.unfreeze()

        for _ in range(self.opt.n_critic_steps):
            critic_loss = self._critic_step(x_real, critic_optimizer, criterion)

        gen_loss = self._generator_step(x_real, generator_optimizer, criterion)

        history['loss_g'] = gen_loss.item()
        history['loss_d'] = critic_loss.item()
        
        self.log_dict(history, prog_bar=True)
        self.__log_gradients()
        return history  