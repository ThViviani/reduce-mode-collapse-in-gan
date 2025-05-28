import wandb
import io
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from trainers.adversarial_trainer import AdversarialTraining
from utils.train_options import TrainOptions
from PIL import Image
from torch.optim.lr_scheduler import ExponentialLR
from trainers.standard_gan import StandardGAN
from trainers.neighbors_embedding_gan import NeighborsEmbeddingMixin
from trainers.rp_gan import RelativisticGanMixin
from trainers.dist_gan import DistMixin
from trainers.dp_gan import DiversityPenaltyMixin


class SyntheticAdversarialTraining(AdversarialTraining):
    def __init__(self, centroids, var, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.centroids = centroids
        self.var = var 

    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def __evaluate_mode_covered(self, data):
        mode_covered = [0 for _ in range(len(self.centroids))]
        for i in range(len(self.centroids)):
            subdata = data - self.centroids[i]
            distance = np.linalg.norm(subdata,axis=1)
            point_in_mode = (distance<=self.var).sum()
            mode_covered[i] = point_in_mode
        return np.array(mode_covered)

    def __log_images(self, n_generated_imgs: int = 32):
        if self.logger is None:
            return
        
        z = self._sample_z(batch_size=n_generated_imgs)
        x_hat = self.generator(z).detach().cpu().numpy()
        mode_covered = self.__evaluate_mode_covered(x_hat)

        threshold = 20 # TODO: move to TrainOptions
        history = {}
        history['modes_covered'] = (mode_covered >= threshold).sum() 
        history['registered_samples'] = mode_covered.sum() 
        self.log_dict(history, prog_bar=True)
        self._log_histogram_for_modes_covered(mode_covered)
        self._log_grid(x_hat)

    def on_train_epoch_end(self):
        self.__log_images(n_generated_imgs=self.opt.n_gen_points_in_synth_experiment)

    def _log_histogram_for_modes_covered(self, points_in_mode):
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(points_in_mode)), points_in_mode)
        plt.axhline(y=20, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='>= 20 mode covered')
        plt.title('Samples per mode')
        plt.xlabel('Mode id')
        plt.ylabel('Number of samples')
        plt.xticks(range(len(points_in_mode)))  # Показываем все моды на оси X

        buf = io.BytesIO() 
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        histogram_image = Image.open(buf)
        self.logger.log_image(
            key="Samples per mode",
            images=[histogram_image],
            caption=[f"epoch {self.current_epoch}"],
            step=self.current_epoch
        )

    def _log_grid(self, data):
        plt.figure(figsize=(10, 5))
        plt.scatter(data[:,0], data[:,1], color='b', s=1)
        plt.scatter(self.centroids[:,0], self.centroids[:,1], marker='x', color='r', s=5)
        
        for centroid in self.centroids:
            circle = plt.Circle(centroid, self.var, color='r', fill=False) 
            plt.gca().add_patch(circle)
        
        plt.title('Generated samples')

        buf = io.BytesIO() 
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close()
        buf.seek(0)

        grid_image = Image.open(buf)
        self.logger.log_image(
            key="Generated samples",
            images=[grid_image],
            caption=[f"epoch {self.current_epoch}"],
            step=self.current_epoch
        )

    def configure_optimizers(self):
        base_optimizers = super().configure_optimizers()
        lr_schedulers = [ExponentialLR(optimizer, gamma=0.99) for optimizer in base_optimizers]
        return base_optimizers, lr_schedulers 

    def compute_modes_covered(self, threshold=20):
        z = self._sample_z(batch_size=self.opt.n_gen_points_in_synth_experiment)
        x_hat = self.generator(z).detach().cpu().numpy()
        mode_covered = self.__evaluate_mode_covered(x_hat)
        registered_samples = mode_covered.sum()
        modes_covered_count = (mode_covered >= threshold).sum()
        return modes_covered_count, registered_samples   


class SyntheticVanilaGAN(SyntheticAdversarialTraining, StandardGAN): pass
class SyntheticRpGAN(RelativisticGanMixin, SyntheticAdversarialTraining): pass

class SynthNEVanilaGAN(NeighborsEmbeddingMixin, SyntheticVanilaGAN): pass
class SynthNERpGAN(NeighborsEmbeddingMixin, SyntheticRpGAN): pass

class SynthDistVanilaGAN(DistMixin, SyntheticVanilaGAN): pass
class SynthDistRpGAN(DistMixin, SyntheticRpGAN): pass

class SynthDpVanilaGAN(DiversityPenaltyMixin, SyntheticVanilaGAN): pass
class SynthDpRpGAN(DiversityPenaltyMixin, SyntheticRpGAN): pass   