import torch.nn as nn
import torch
import lightning as L
import torch.nn.functional as F

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.image import FrechetInceptionDistance, InceptionScore


class ModesCoveredKLMNIST(Callback):
    def __init__(
        self,
        classifier: nn.Module,
        target_labels_for_kl: torch.Tensor,
        z_dim: int,
        total_samples: int = 25600,
        batch_size: int = 256,
        confidence=0.99,
        n_classes=10,
        log_every_n_epochs=5,
    ):
        super().__init__()

        self.log_every_n_epochs = log_every_n_epochs
        self.classifier = classifier.eval()
        self.z_dim = z_dim
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.confidence = confidence
        self.n_classes = n_classes
        self.modes_covered_results = []
        self.kl_losses = []
        self.target_labels_for_kl = target_labels_for_kl

    def _compute_modes_covered(self, preds_labels, n_classes=1000):
        modes_covered = torch.bincount(preds_labels.int(), minlength=n_classes)
        return (modes_covered > 0).sum().item()

    def _get_preds_labels(self, pl_module):
        self.classifier.to(pl_module.device)
        num_batches = self.total_samples // self.batch_size
        preds_labels = []
        for _ in range(num_batches):
            z = torch.randn(self.batch_size, self.z_dim, device=pl_module.device)
            with torch.no_grad():
                fake_imgs = pl_module.generator(z)
                logits    = self.classifier(fake_imgs)
                p_logits = torch.nn.functional.softmax(logits, dim=1).detach()
                confidences, labels_hat = torch.max(p_logits, dim=1)
                preds_labels.extend(labels_hat * (confidences > self.confidence))

        preds_labels = torch.Tensor(preds_labels)
        return preds_labels
    
    def _compute_kl_loss(self, true_labels, hat_labels, n_classes=1000):
        counts = torch.bincount(true_labels.int(), minlength=n_classes)
        p_real = (counts.float() / counts.sum())

        counts = torch.bincount(hat_labels.int(), minlength=n_classes)
        q = counts.float() / counts.sum()
        q = torch.log(q + 1e-12)
        
        kl_loss = nn.KLDivLoss(reduction="sum")
        output = kl_loss(q, p_real)
        return output.item() 

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        preds_labels = self._get_preds_labels(pl_module)
        modes_covered = self._compute_modes_covered(preds_labels, self.n_classes)
        kl_loss = self._compute_kl_loss(self.target_labels_for_kl, preds_labels, self.n_classes)
        
        self.modes_covered_results.append(modes_covered)
        self.kl_losses.append(kl_loss)

        if isinstance(trainer.logger, WandbLogger):
            wandb_run = trainer.logger.experiment 
            
            wandb_run.log(
                {"modes_covered": modes_covered, "epoch": trainer.current_epoch},
            )

            wandb_run.log(
                {"kl": kl_loss, "epoch": trainer.current_epoch},
            )

class ModesCoveredKLStackedMNIST(ModesCoveredKLMNIST):
    def _get_preds_labels(self, pl_module):
        results = []
        self.classifier.to(pl_module.device)
        num_batches = self.total_samples // self.batch_size
        for _ in range(num_batches):
            z = torch.randn(self.batch_size, self.z_dim, device=pl_module.device)
            x = pl_module.generator(z)
            labels = []
            confidences = []
            for channel in range(3):
                digits_in_channel = x[:, channel, :, :].unsqueeze(dim=1)
                logits = self.classifier(digits_in_channel)
                p_logits = torch.nn.functional.softmax(logits, dim=1).detach()
                channel_confidences, labels_hat = torch.max(p_logits, dim=1)
                labels.append(labels_hat)
                confidences.append(channel_confidences)
                
            accepted_labels = (confidences[0] > self.confidence) * (confidences[1] > self.confidence) * (confidences[2] > self.confidence) # хотя бы одну цифру неуверено классифицирует == даем общий лейбл 000
            result = 100 * labels[0] + 10 * labels[1] + labels[2]       
            results.extend(result * accepted_labels)
        return torch.Tensor(results)
    
class FIDIS(Callback):
    def __init__(self, log_every_n_epochs=5):
        self.fid = FrechetInceptionDistance(normalize=True)
        self.is_score = InceptionScore(normalize=True)
        self.fid_results = []
        self.is_results = []
        self.log_every_n_epochs = log_every_n_epochs 

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
            
        fake, real = outputs['fake_images'], outputs['real_images']
        
        self.fid.to(pl_module.device)
        self.is_score.to(pl_module.device)
        
        if real.shape[1] == 1:
            real = real.repeat(1, 3, 1, 1)
            fake = fake.repeat(1, 3, 1, 1)
            
        self.fid.update(real, real=True)
        self.fid.update(fake, real=False)            
        self.is_score.update(fake)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        
        current_fid = self.fid.compute().item()
        current_is = self.is_score.compute()

        if isinstance(trainer.logger, WandbLogger):
            wandb_run = trainer.logger.experiment 
            
            wandb_run.log(
                {"fid": current_fid, "epoch": trainer.current_epoch},
            )

            wandb_run.log(
                {'is': current_is[0].item(), 'epoch': trainer.current_epoch}
            )
        
        self.fid.reset(); self.is_score.reset()

        self.fid_results.append(current_fid)
        self.is_results.append(current_is)