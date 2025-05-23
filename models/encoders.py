import torch
import torch.nn as nn


class EncoderMNIST(nn.Module):
    def __init__(self, x_shape, z_dim=128, dim=64, kernel_size=5, stride=2):
        super().__init__()
        in_channels = x_shape[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size, stride, padding=2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim*2, kernel_size, stride, padding=2),
            nn.BatchNorm2d(dim*2),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(dim*2, dim*4, kernel_size, stride, padding=2),
            nn.BatchNorm2d(dim*4),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *x_shape)
            dummy = self.conv3(self.conv2(self.conv1(dummy)))
            fc_input_dim = dummy.view(1, -1).size(1)

        self.fc = nn.Linear(fc_input_dim, z_dim)

    def forward(self, img):
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)