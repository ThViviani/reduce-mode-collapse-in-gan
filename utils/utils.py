import torch
import matplotlib.pyplot as plt
import torch.utils as vutils
import numpy as np


def generate_some_examples(generator, batch_size, z_dim, device, n=64):
    z = torch.randn(batch_size, z_dim, device=device)
    generated_imgs = generator(z).detach()

    # Create grid and display
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Generated Images")
    grid = vutils.make_grid(generated_imgs[:n], padding=2, normalize=True).cpu()
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show();