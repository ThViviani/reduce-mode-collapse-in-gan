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

The following loss functions are defined for each method investigated in this project:

1. **Standard GAN Loss** ([`trainers/standard_gan.py`](./trainers/standard_gan.py)):
- Ae-step:  
- D-step: 

  $$
    -\mathbb{E}_{x \sim p_{data}}[log(D_{\phi}(x))] -\mathbb{E}_{z \sim p_{z}}[log(1 - D_{\phi}(G_{\theta}(z)))] \to \min_{\phi}.\quad(1)
  $$

- G-step: 
$$
  -\mathbb{E}_{z \sim p_z} [log(D_{\phi}(G_{\theta}(z))] \to \min_{\theta}.\quad(2)
$$

  $$
    -\mathbb{E}_{x \sim p_{data}}[log(D_{\phi}(x))] -\mathbb{E}_{z \sim p_{z}}[log(1 - D_{\phi}(G_{\theta}(z)))] \to \min_{\phi}.\quad(1)
  $$

2. **DistVanilaGAN** ([`trainers/dist_gan.py`](./trainers/dist_gan.py)):  
It's StandardGAN + Latent-distance regularization from Dist-GAN [1]
- Ae-step:  
  $$
  \mathcal{L}_{AE}(\gamma, \theta) + \lambda_{dist}\mathcal{L}_W \to \min_{\gamma, \theta}, \quad (3)
  $$
  where  
  $$\mathcal{L}_{AE}(\gamma, \theta) = \mathbb{E}_{x \sim p_{data}} \| x - G_\theta(E_\gamma(x)) \|^2,$$
  encoder $E_{\gamma}: X \to Z$ codes images to the latent space Z ,
  $$\mathcal{L}_W(\gamma, \theta) = ||𝑓(𝑥,𝐺_𝜃 (𝑧))−𝜆_𝑊 𝑔(𝐸_γ (𝑥),𝑧)||_2^2,$$
  $$𝑓(𝑥,𝐺_𝜃 (𝑧)):=𝑀_𝑑 (𝔼_𝑥 [𝐺_𝜃 (𝐸_γ (𝑥))]−𝔼_𝑧 [𝐺_𝜃 (𝑧)])),$$
  $$𝑔(𝐸_𝜔 (𝑥),𝑧):=𝑀_𝑑 (𝔼_𝑥 (𝐸_γ (𝑥))−𝔼_𝑧 𝑧),$$
  $$𝑀_𝑑 (\overline{x})=\frac{1}{dim(\overline{x})} \sum_{i = 1}^{dim(\overline{x})}x_i.$$
- D-step: $(1)$
- G-step: $(2)$

3. **DpVanilaGAN** ([`trainers/dp_gan.py`](./trainers/dp_gan.py)):
It's StandardGAN + Latent-distance regularization from DpGAN [2]
- Ae-step:  
- D-step: $(1)$
- G-step: 
  $$
    -\mathbb{E}_{z \sim p_z} [log(D_{\phi}(G_{\theta}(z))] + DP(z) \to \min_{\theta},
  $$
  where $$DP(z) = \mathbb{E_z}[\frac{G_f(i, j)}{G_z(i, j)}] \approx \frac{1}{m^2} \sum_i^m\sum_j^m \frac{\sigma(s \frac{f_i^Tf_j}{||f_i||_2 ||f_j||_2})}{\sigma(s \frac{z_i^Tz_j}{||z_i||_2 ||z_j||_2})}. \quad(4)$$

4. **NEVanilaGAN** ([`trainers/neighbors_embedding_gan.py`](./trainers/neighbors_embedding_gan.py)):  
It's StandardGAN + Latent-distance regularization from GN-GAN [3].
$\mathcal{L}_W(\gamma, \theta)$ is computed using the t-SNE algorithm to preserve latent structure.
- Ae-step: $(3)$, where 
  $$\mathcal{L}_W(\gamma, \theta) = \sum_i \sum_j p_{i, j}log\frac{p_{i, j}}{q_{i, j}}. \quad(5)$$
- D-step: $(1)$
- G-step: $(2)$

5. **NEVanilaGAN_hat** ([`trainers/ne_gan_without_ae.py`](./trainers/ne_gan_without_ae.py):  
It's StandardGAN + Latent-distance regularization from GN-GAN [3], where $\mathcal{L}_W(\theta)$ and used on the G-step.
- Ae-step:
- D-step: $(1)$
- G-step: $$-\mathbb{E}_{z \sim p_z} [log(D_{\phi}(G_{\theta}(z))] +  \lambda_{dist}\mathcal{L}_W \to \min_{\theta},\quad \mathcal{L}_W \quad from \quad(5).$$
---
6. **RpGAN** ([`trainers/rp_gan.py`](./trainers/rp_gan.py)) [4]:
- Ae-step:
- D-step: $$\mathcal{L}_D^{RpGAN} = 
-\mathbb{E}_{\tilde x}[log\sigma(C_\omega(x_r) - C_\omega(x_f))] \to \min_{\omega}, \quad (6)$$
  where $\tilde x = (x_r, x_f),\quad x_r \sim p_{data}, \quad x_f \sim G_{\theta}(z), \quad C_{\omega} -$ logits from the Discriminator.
- G-step: $$\mathcal{L}_G^{RpGAN} = 
-\mathbb{E}_{\tilde x}[log\sigma(C_\omega(x_f) - C_\omega(x_r))] \to \min_{\theta}.\quad (7)$$

7. **DistRpGAN** ([`trainers/dist_gan.py`](./trainers/dist_gan.py)):  
It's RpGAN + Latent-distance regularization from Dist-GAN [1]  
- Ae-step: $(3)$
- D-step: $(6)$
- G-step: $(7)$

8. **DpRpGAN** ([`trainers/dp_gan.py`](./trainers/dp_gan.py)):  
It's RpGAN + Latent-distance regularization from DpGAN [2]
- Ae-step:
- D-step: $(6)$
- G-step:$$\mathcal{L}_G^{RpGAN} + DP(z) \to \min_{\theta}.$$

9. **NeRpGAN** ([`trainers/neighbors_embedding_gan.py`](./trainers/neighbors_embedding_gan.py)):  
It's RpGAN + Latent-distance regularization from GN-GAN [3]
- Ae-step $(3)$:
- D-step: $(6)$
- G-step: $(7)$


10. **NERpGAN_hat** ([`trainers/ne_gan_without_ae.py`](./trainers/ne_gan_without_ae.py):  
It's RpGAN + Latent-distance regularization from GN-GAN [3], where $\mathcal{L}_W(\theta)$ and used on the G-step.
- Ae-step:
- D-step: $(6)$
- G-step:$$\mathcal{L}_G^{RpGAN} + \mathcal{L}_W(\theta) \to \min_{\theta}, \quad \mathcal{L}_W \quad from \quad(5).$$
---

11. **RpGAN+R1R2** ([`trainers/rp_gan.py`](./trainers/rp_gan.py), use_r1r2_penalty=True) [5]:  
It's RpGAN + R1,R2 grad penaltys for the Discriminator. 
- Ae-step:
- D-step: $$\mathcal{L}_D^{RpGAN + R_1 + R_2} = 
\mathcal{L}_D^{RpGAN} + \frac{\gamma}{2}(R_1(\omega) + R_2(\theta, \omega)) \to \min_{\omega} \quad (8)$$
- G-step: $(7)$ 

12. **DistRpGAN+R1R2** ([`trainers/dist_gan.py`](./trainers/dist_gan.py), use_r1r2_penalty=True):  
It's RpGAN + R1,R2 grad penaltys for the Discriminator + Latent-distance regularization from Dist-GAN [1]  
- Ae-step: $(3)$
- D-step: $(8)$
- G-step: $(7)$

13. **DpRpGAN+R1R2** ([`trainers/dp_gan.py`](./trainers/dp_gan.py), use_r1r2_penalty=True):    
It's RpGAN + R1,R2 grad penaltys for the Discriminator + Latent-distance regularization from DpGAN [2]
- Ae-step:
- D-step: $(8)$
- G-step:
$$
  \mathcal{L}_G^{RpGAN} + DP(z) \to \min_{\theta}.
$$

14. **NeRpGAN** ([`trainers/neighbors_embedding_gan.py`](./trainers/neighbors_embedding_gan.py), use_r1r2_penalty=True):    
It's RpGAN + R1,R2 grad penaltys for the Discriminator + Latent-distance regularization from GN-GAN [3]
- Ae-step $(3)$:
- D-step: $(8)$
- G-step: $(7)$

15. **NERpGAN_hat** ([`trainers/ne_gan_without_ae.py`](./trainers/ne_gan_without_ae.py):  
It's RpGAN + Latent-distance regularization from GN-GAN [3], where $\mathcal{L}_W(\theta)$ and used on the G-step.
- Ae-step:
- D-step: $(8)$
- G-step:$$\mathcal{L}_G^{RpGAN} + \mathcal{L}_W(\theta) \to \min_{\theta}, \quad \mathcal{L}_W \quad from \quad(5).$$
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
