import torchvision.datasets as datasets
import numpy as np
import torch

from PIL import Image


class StackedMNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(StackedMNIST, self).__init__(root=root, train=train, transform=transform,
                                           target_transform=target_transform, download=download)

        index1 = np.hstack([np.random.permutation(len(self.data)), np.random.permutation(len(self.data))])
        index2 = np.hstack([np.random.permutation(len(self.data)), np.random.permutation(len(self.data))])
        index3 = np.hstack([np.random.permutation(len(self.data)), np.random.permutation(len(self.data))])
        self.num_images = 2 * len(self.data)

        self.index = []
        new_targets = []
        for i in range(self.num_images):
            self.index.append((index1[i], index2[i], index3[i]))

            a = int(self.targets[index1[i]])
            b = int(self.targets[index2[i]])
            c = int(self.targets[index3[i]])
            new_targets.append(a * 100 + b * 10 + c)
        self.targets = torch.tensor(new_targets)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        target = 0
        img_per_channels = []
        for i in range(3):
            img_ = self.data[self.index[index][i]]
            img_per_channels.append(img_)

        img = np.stack(img_per_channels, dtype=np.uint8).transpose((1, 2, 0))
        img = Image.fromarray(img, mode="RGB")
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target