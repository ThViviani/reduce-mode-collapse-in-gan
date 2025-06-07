import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import torchvision.transforms as transforms
import torch_fidelity
import gc
import torch.nn as nn

from torch.utils.data import Dataset


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

class GeneratorForFIDWrapper(nn.Module):
    """A wrapper which transforms generator's images from [-1, 1] to [0, 255] and [B, 1, H, W] -> [B, 3, H, W]"""

    def __init__(self, generator, device, img_size=299):
        super().__init__()
        self.generator = generator
        self.img_size = img_size
        self.device = device

    def forward(self, z):
        with torch.no_grad():
            imgs = self.generator(z)  # [B, 1, H, W], float32, [-1, 1]
            imgs = (imgs + 1) / 2  # [-1, 1] → [0, 1]
            imgs = imgs * 255
            imgs = imgs.clamp(0, 255).to(torch.uint8)

            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)  # [B, 1, H, W] → [B, 3, H, W]

        return imgs
    
class DatasetForFIDWrapper(Dataset):
    """A dataset wrapper which returns only images (without labels)"""
    def __init__(self, dataset_class, **kwargs):
        """
            Args:
                dataset_class: Dataset class to wrap (e.g., torchvision.datasets.MNIST)
                **kwargs: All arguments required to initialize the dataset_class
        """
        self.dataset = dataset_class(**kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        image = (image + 1) / 2  # [-1, 1] → [0, 1]
        image = image * 255
        image = image.clamp(0, 255).to(torch.uint8) # -> [0, 255]
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        return image
    
def regist_dataset_in_torch_fidelity(torchvision_dataset, root_to_regist_dataset='test_dataset'):
    compose = transforms.Compose([
        transforms.ToTensor(),
    ])


    test_ds = torch_fidelity.register_dataset(
        root_to_regist_dataset,
        lambda root, download: DatasetForFIDWrapper(torchvision_dataset, root=root, download=download, train=False, transform=compose)
    )

def compute_metrics(model, seed, latent_space_dim, input2, input2_root='', input1_model_num_samples=10_000, input2_model_num_samples=5_000):
    generator = GeneratorForFIDWrapper(model.generator, model.device)
    wrapped_generator = torch_fidelity.GenerativeModelModuleWrapper(generator, latent_space_dim, 'normal', 0)

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=wrapped_generator,
        input2=input2,
        input1_model_num_samples=input1_model_num_samples,
        input2_model_num_samples=input2_model_num_samples,
        cuda=True,
        isc=True,
        fid=True,
        verbose=False,
        input2_root=input2_root,
        datasets_download=True,
        rng_seed=seed,
    )

    print(metrics_dict)
    del generator
    del wrapped_generator
    gc.collect()
    torch.cuda.empty_cache()

    return metrics_dict