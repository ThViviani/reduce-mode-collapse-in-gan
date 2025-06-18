import torch.nn as nn
import torch
import lightning as L

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger


class ModesCoveredMNIST(Callback):
    def __init__(
        self,
        classifier: nn.Module,
        z_dim: int,
        total_samples: int = 25600,
        batch_size: int = 256,
        confidence=0.99,
        n_classes=1000,
    ):
        super().__init__()
        self.classifier = classifier.eval()
        self.z_dim = z_dim
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.confidence = confidence
        self.n_classes = n_classes
        self.results = []

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

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        preds_labels = self._get_preds_labels(pl_module)
        modes_covered = self._compute_modes_covered(preds_labels, self.n_classes)
        
        self.results.append(modes_covered)

        if isinstance(trainer.logger, WandbLogger):
            wandb_run = trainer.logger.experiment 
            
            wandb_run.log(
                {"modes_covered": modes_covered, "epoch": trainer.current_epoch},
            )