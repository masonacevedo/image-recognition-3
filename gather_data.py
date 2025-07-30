import torchvision.datasets
import torchvision.transforms as transforms

from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import models


folder = "cifar_10_imagery"

dataset = torchvision.datasets.CIFAR10(root=folder, download=True)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
