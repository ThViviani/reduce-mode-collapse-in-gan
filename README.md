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

50 000 points, 25 gaussians with var = 0.1, 500 epochs, seed=999

**Mode Coverage Criteria: 20 points in the mode.**

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

**MNIST experiment ['notebooks/mnist_experiments.ipynb'](./notebooks/mnist_experiments.ipynb):**

![MNIST - experiment](/assets/MNIST.gif)

50 000 train, 10 000 test, 50 epochs, 5 runs with differenet seeds.


**Mode Coverage Criteria: The pretrained classifier on MNIST must detect digits with at least 99% confidence.**

**Mean of 5 runs on the best epoch:**
| Group  | Model                 | IS ↑           | FID ↓   | Modes covered ↑| KL ↓   |
|--------|-----------------------|----------------|---------|----------------|--------|
| 1      | DpVanillaGAN          | 2.12 ±0.02     | 12.51   | 10             | 0.96   |
|        | NEVanillaGAN          | 2.02 ±0.02     | 19.89   | 10             | 1.03   |
|        | DistVanillaGAN        | 1.98 ±0.02     | 18.99   | 10             | 0.86   |
|        | NEVanillaGAN_hat      | 2.09 ±0.03     | 17.61   | 10             | 1.08   |
|        | StandardGAN           | 2.11 ±0.03     | 15.87   | 10             | 0.96   |
| 2      | RpGAN                 | 1.98 ±0.03     | 19.31   | 10             | 1.03   |
|        | NErpGAN_hat           | 2.10 ±0.02     | 21.98   | 10             | 1.10   |
|        | DistRpGAN             | 1.98 ±0.02     | 21.32   | 10             | 1.01   |
|        | NErpGAN               | 2.03 ±0.03     | 24.12   | 10             | 1.10   |
|        | DpRpGAN               | 2.09 ±0.02     | 18.95   | 10             | 1.16   |
| 3      | NErpGAN_hat+R1R2      | 2.05 ±0.02     | 65.31   | 9.2            | 3.03   |
|        | DistRpGAN+R1R2        | 2.07 ±0.02     | 14.38   | 10             | 0.77   |
|        | NErpGAN+R1R2          | 2.11 ±0.02     | 14.76   | 10             | 0.82   |
|        | DpRpGAN+R1R2          | 1.37 ±0.01     | 366.29  | 2              | 20.11  |
|        | RpGAN_R1R2            | 2.14 ±0.03     | 8.08    | 10             | 0.74   |
|        | RpGAN_R1R2            | 2.16 ±0.02     | 7.80    | 10             | 0.75   |
| 4      | NErpGAN+R1R2          | 2.17 ±0.03     | 7.57    | 10             | 0.77   |


**Stacked-MNIST experiment [`notebooks/stacked_mnist_experiments.ipynb`](./notebooks/stacked_mnist_experiments.ipynb):**

  Dataset builder: [`dataset_builders/stacked_mnist.py`](./dataset_builders/stacked_mnist.py) 

![Stacked-MNIST experiment](/assets/stacked_mnist.gif)

120 000 train, 20 000 test, 300 epochs, seed=999. 

**Mode Coverage Criteria: The pretrained classifier on MNIST must detect all digits with at least 99% confidence.**

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

**CIFAR-10 experiment [`notebooks/cifar10_experiments.ipynb`](./notebooks/cifar10_experiments.ipynb):**

![CIFAR-10 experiment](/assets/CIFAR-10.gif)

50 000 train, 10 000 test, 150 epochs, seed=999.

**Best epoch:**
| Group  | Model                     |   IS ↑ |   FID ↓ |
|--------|---------------------------|-------:|--------:|
| 1      | DpVanilaGAN               |   6.84 |   35.52 |
| 1      | NEVanilaGAN               |   2.88 |  139.53 |
| 1      | DistVanilaGAN             |   4.22 |   76.70 |
| 1      | NEVanilaGAN_hat           |   6.41 |   38.52 |
| 1      | StandardGAN               |   6.38 |   39.83 |
| 2      | RpGAN                     |   6.75 |   34.16 |
| 2      | NERpGAN_hat               |   6.82 |   33.26 |
| 2      | DistRpGAN                 |   4.06 |   93.49 |
| 2      | NERpGAN                   |   1.58 |   257.22|
| 2      | DpRpGAN                   |   6.76 |   38.44 |
| 3      | RpGAN_R1R2                |   6.28 |   41.18 |
| 3      | NERpGAN_hat+R1R2          |   6.08 |   41.67 |
| 3      | DistRpGAN+R1R2            |   5.40 |   57.22 |
| 3      | NERpGAN+R1R2              |   1.63 |  266.17 |
| 3      | DpRpGAN+R1R2              |   6.03 |   43.93 |


## References
[1] Tran N.-T., Bui T.-A., Cheung N.-M. Dist-GAN: An Improved GAN using Distance Constraints. URL: https://doi.org/10.48550/arXiv.1803.08887

[2] Pei S., Xu R.Y.D., Xiang S., Meng G. Alleviating Mode Collapse in GAN via Diversity Penalty Module. URL: https://doi.org/10.48550/arXiv.2108.02353

[3] Tran N.-T., Bui T.-A., Cheung N.-M. Improving GAN with neighbors embedding and gradient matching. URL: https://doi.org/10.48550/arXiv.1811.01333

[4] Jolicoeur-Martineau A. The relativistic discriminator: a key element missing from standard GAN. URL: https://doi.org/10.48550/arXiv.1807.00734

[5]  Huang Y., Gokaslan A., Kuleshov V., Tompkin J. The GAN is dead; long live the GAN! A Modern GAN Baseline. URL: https://doi.org/10.48550/arXiv.2501.05441

---

*Last updated: July 3, 2025*
