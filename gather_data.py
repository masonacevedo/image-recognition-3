import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import models


folder = "cifar_10_imagery"

dataset = torchvision.datasets.CIFAR10(root=folder, download=True)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# model.fc = nn.linear(model.fc.in_features, 10)

training_data, val_data = random_split(dataset, [0.8, 0.2])
dataloader = DataLoader(training_data, batch_size=32, shuffle=True, num_workers=4)

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),           # Add padding and random crop
    transforms.RandomHorizontalFlip(p=0.5),         # Random horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image, label = dataset[0]
image.show()


transformed_tensor = transform(image)
transformed_image = F.to_pil_image(transformed_tensor)
transformed_image.show()