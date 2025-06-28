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

14. **NeRpGAN** ([`trainers/neighbors_embedding_gan.py`](./trainers/neighbors_embedding_gan.py), use_r1r2_penalty=True):    
It's RpGAN + R1,R2 grad penaltys for the Discriminator + Latent-distance regularization from GN-GAN [3]

15. **NERpGAN_hat** ([`trainers/ne_gan_without_ae.py`](./trainers/ne_gan_without_ae.py), use_r1r2_penalty=True):  
It's RpGAN + Latent-distance regularization from GN-GAN [3], where $\mathcal{L}_W(\theta)$ and used on the G-step.
---

## 📊 Results

TODO

## References
[1] Tran N.-T., Bui T.-A., Cheung N.-M. Dist-GAN: An Improved GAN using Distance Constraints. URL: https://doi.org/10.48550/arXiv.1803.08887

[2] Pei S., Xu R.Y.D., Xiang S., Meng G. Alleviating Mode Collapse in GAN via Diversity Penalty Module. URL: https://doi.org/10.48550/arXiv.2108.02353

[3] Tran N.-T., Bui T.-A., Cheung N.-M. Improving GAN with neighbors embedding and gradient matching. URL: https://doi.org/10.48550/arXiv.1811.01333

[4] Jolicoeur-Martineau A. The relativistic discriminator: a key element missing from standard GAN. URL: https://doi.org/10.48550/arXiv.1807.00734

[5]  Huang Y., Gokaslan A., Kuleshov V., Tompkin J. The GAN is dead; long live the GAN! A Modern GAN Baseline. URL: https://doi.org/10.48550/arXiv.2501.05441

---

*Last updated: June 28, 2025*
