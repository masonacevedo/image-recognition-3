import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from torch.utils.data import random_split, DataLoader
from torchvision import models

import random


folder = "cifar_10_imagery"

dataset = torchvision.datasets.CIFAR10(root=folder, download=True)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# model.fc = nn.linear(model.fc.in_features, 10)

training_data, val_data = random_split(dataset, [0.8, 0.2])
dataloader = DataLoader(training_data, batch_size=32, shuffle=True, num_workers=4)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.4),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.ToTensor(),
])


indices = random.sample(range(0, len(dataset)), 5)

for index in indices:
    image, label = dataset[index]
    image.show()


    transformed_tensor = transform(image)
    transformed_image = F.to_pil_image(transformed_tensor)
    transformed_image.show()