# 🧪 Mode Collapse Mitigation in GANs

This project investigates various techniques to **reduce mode collapse** in Generative Adversarial Networks (GANs). The core component is the **RpGAN discriminator**, evaluated in combination with architectural modifications and regularization strategies.

---

## Key Directions

* **RpGAN** – discriminator with relative-pairwise comparison mechanism.
* **Autoencoder-based architectures**  integrated into the GAN pipeline.
* **Latent-distance regularization** and other loss-based strategies.
* **Experiments on diverse datasets to evaluate**:

  * **Stability**
  * **Mode coverage**
  * **Sample quality**

---

## Loss Functions

The following loss functions are defined for each method investigated in this project: ([`notebooks/losses_for_testing_models.ipynb`](./notebooks/losses_for_testing_models.ipynb))

1. **Standard GAN Loss** ([`trainers/standard_gan.py`](./trainers/standard_gan.py)):

2. **DistVanilaGAN** ([`trainers/dist_gan.py`](./trainers/dist_gan.py)):  
It's StandardGAN + Latent-distance regularization from Dist-GAN [1]

3. **DpVanilaGAN** ([`trainers/dp_gan.py`](./trainers/dp_gan.py)):
It's StandardGAN + Latent-distance regularization from DpGAN [2]

4. **NEVanilaGAN** ([`trainers/neighbors_embedding_gan.py`](./trainers/neighbors_embedding_gan.py)):  
It's StandardGAN + Latent-distance regularization from GN-GAN [3].

5. **NEVanilaGAN_hat** ([`trainers/ne_gan_without_ae.py`](./trainers/ne_gan_without_ae.py):  
It's StandardGAN + Latent-distance regularization from GN-GAN [3], where $\mathcal{L}_W(\theta)$ and used on the G-step.
---
6. **RpGAN** ([`trainers/rp_gan.py`](./trainers/rp_gan.py)) [4]:

7. **DistRpGAN** ([`trainers/dist_gan.py`](./trainers/dist_gan.py)):  
It's RpGAN + Latent-distance regularization from Dist-GAN [1]  

8. **DpRpGAN** ([`trainers/dp_gan.py`](./trainers/dp_gan.py)):  
It's RpGAN + Latent-distance regularization from DpGAN [2]

9. **NeRpGAN** ([`trainers/neighbors_embedding_gan.py`](./trainers/neighbors_embedding_gan.py)):  
It's RpGAN + Latent-distance regularization from GN-GAN [3]

10. **NERpGAN_hat** ([`trainers/ne_gan_without_ae.py`](./trainers/ne_gan_without_ae.py):  
It's RpGAN + Latent-distance regularization from GN-GAN [3], where $\mathcal{L}_W(\theta)$ and used on the G-step.
---

11. **RpGAN+R1R2** ([`trainers/rp_gan.py`](./trainers/rp_gan.py), use_r1r2_penalty=True) [5]:  
It's RpGAN + R1,R2 grad penaltys for the Discriminator. 

12. **DistRpGAN+R1R2** ([`trainers/dist_gan.py`](./trainers/dist_gan.py), use_r1r2_penalty=True):  
It's RpGAN + R1,R2 grad penaltys for the Discriminator + Latent-distance regularization from Dist-GAN [1]  

13. **DpRpGAN+R1R2** ([`trainers/dp_gan.py`](./trainers/dp_gan.py), use_r1r2_penalty=True):    
It's RpGAN + R1,R2 grad penaltys for the Discriminator + Latent-distance regularization from DpGAN [2]

14. **NeRpGAN+R1R2** ([`trainers/neighbors_embedding_gan.py`](./trainers/neighbors_embedding_gan.py), use_r1r2_penalty=True):    
It's RpGAN + R1,R2 grad penaltys for the Discriminator + Latent-distance regularization from GN-GAN [3]

15. **NERpGAN_hat+R1R2** ([`trainers/ne_gan_without_ae.py`](./trainers/ne_gan_without_ae.py), use_r1r2_penalty=True):  
It's RpGAN + R1,R2 grad penaltys for the Discriminator + Latent-distance regularization from GN-GAN [3], where $\mathcal{L}_W(\theta)$ and used on the G-step.
---

## Implementation details

All variants of tested model variants combine one of two base trainer classes with zero or more mixins:

- Base Trainers

  - StandardGAN — vanilla GAN loss functions

  - RpGAN — relativistic average GAN loss

- Regularization Mixins

  - DistMixin — adds Dist‑GAN regularization and encoder loss
  
  - DiversityPenaltyMixin - adds DpGAN regularization

  - NeighborsEmbeddingMixin - adds GN-GAN regularization and encoder loss

  - NeighborsEmbeddingMixin_hat - add GN-GAN regularization without AE loss

**GAN Variant Inheritance Hierarchy:**
![GAN Variant Inheritance Hierarchy](/assets/image.png)

## 📊 Results

**2D - synthetic experiment [`notebooks/2d-synthetic-experiments.ipynb`](./notebooks/2d-synthetic-experiments.ipynb):**

![2D - synthetic experiment](/assets/2d.gif)

50 000 points, 25 gaussians with var = 0.1, 500 epochs

**Best epoch:**
| Group  |           Model         | Modes covered| Points in modes        |
|--------|-------------------------|-------------:|-----------------------:|
| 1      | StandardGAN             |           24 |                   1710 |
| 1      | DistVanilaGAN           |           11 |                    715 |
| 1      | DpVanilaGan             |            7 |                    469 |
| 1      | NEVanilaGAN             |            3 |                    160 |
| 1      | NEVanilaGAN_hat         |           25 |                   1870 |
| 2      | RpGAN                   |           25 |                   1892 |
| 2      | DistRpGAN               |           15 |                   1007 |
| 2      | NERpGAN                 |            2 |                    163 |
| 2      | DpRpGAN                 |           25 |                   1939 |
| 2      | NERpGAN_hat             |           25 |                   1907 |
| 3      | RpGAN+R1R2              |            5 |                    169 |
| 3      | DistRpGAN+R1R2          |            1 |                     70 |
| 3      | NERpGAN+R1R2            |            0 |                    148 |
| 3      | DpRpGAN+R1R2            |            3 |                    148 |
| 3      | NERpGAN_hat+R1R2        |            1 |                    985 |

**Stacked-MNIST experiment [`notebooks/stacked_mnist_experiments.ipynb`](./notebooks/stacked_mnist_experiments.ipynb):**

  Dataset builder: [`dataset_builders/stacked_mnist.py`](./dataset_builders/stacked_mnist.py) 

![Stacked-MNIST experiment](/assets/stacked_mnist.gif)

120 000 train, 20 000 test, 300 epochs. 

**Best epoch:**
| Group  | Model                     |   IS ↑ |  Modes covered ↑ |    KL ↓ |
|--------|---------------------------|-------:|-----------------:|--------:|
| 1      | DpVanilaGAN               |   1.32 |                3 |   20.66 |
| 1      | NEVanilaGAN               |   1.56 |               90 |   19.11 |
| 1      | DistVanilaGAN             |   1.81 |               98 |   19.03 |
| 1      | NEVanilaGAN_hat           |   1.58 |               42 |   20.01 |
| 1      | StandardGAN               |   1.43 |               18 |   20.38 |
| 2      | RpGAN                     |   1.50 |               57 |   20.35 |
| 2      | NERpGAN_hat               |   1.66 |               45 |   19.97 |
| 2      | DistRpGAN                 |   1.66 |              101 |   18.96 |
| 2      | NERpGAN                   |   1.30 |              105 |   18.87 |
| 2      | DpRpGAN                   |   1.64 |                5 |   20.63 |
| 3      | RpGAN_R1R2                |   1.53 |               41 |   20.03 |
| 3      | NERpGAN_hat+R1R2          |   1.50 |               73 |   19.46 |
| 3      | DistRpGAN+R1R2            |   1.72 |              104 |   18.94 |
| 3      | NERpGAN+R1R2              |   1.68 |              102 |   18.95 |
| 3      | DpRpGAN+R1R2              |   1.45 |               26 |   20.27 |


## References
[1] Tran N.-T., Bui T.-A., Cheung N.-M. Dist-GAN: An Improved GAN using Distance Constraints. URL: https://doi.org/10.48550/arXiv.1803.08887

[2] Pei S., Xu R.Y.D., Xiang S., Meng G. Alleviating Mode Collapse in GAN via Diversity Penalty Module. URL: https://doi.org/10.48550/arXiv.2108.02353

[3] Tran N.-T., Bui T.-A., Cheung N.-M. Improving GAN with neighbors embedding and gradient matching. URL: https://doi.org/10.48550/arXiv.1811.01333

[4] Jolicoeur-Martineau A. The relativistic discriminator: a key element missing from standard GAN. URL: https://doi.org/10.48550/arXiv.1807.00734

[5]  Huang Y., Gokaslan A., Kuleshov V., Tompkin J. The GAN is dead; long live the GAN! A Modern GAN Baseline. URL: https://doi.org/10.48550/arXiv.2501.05441

---

*Last updated: June 28, 2025*
