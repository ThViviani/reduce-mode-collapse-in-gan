import torch
import torch.nn as nn

from torchvision import models


class ModelFreezeMixin:
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

class MLP(ModelFreezeMixin, nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNet50OnMNIST(nn.Module):
    """
    Initializes a ResNet-50 model adapted for grayscale MNIST digit classification.
    Loads pretrained weights from the given checkpoint path.
    """

    def __init__(self, path_to_checkpoint=''):
        super().__init__()

        self.model = models.resnet50(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(2048, 10, bias=True)
        self.model.load_state_dict(torch.load(path_to_checkpoint, weights_only=True))

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        outputs = self.model(x)
        _, predicted = torch.max(outputs.data, 1)
        return predicted