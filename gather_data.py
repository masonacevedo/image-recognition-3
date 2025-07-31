import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import models


folder = "cifar_10_imagery"

dataset = torchvision.datasets.CIFAR10(root=folder, download=True)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
training_data, val_data = random_split(dataset, [0.8, 0.2])
dataloader = DataLoader(training_data, batch_size=32, shuffle=True, num_workers=4)

transformations = [
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
]

transform = transforms.Compose(transformations)
image, label = dataset[0]
image.show()


transformed_tensor = transform(image)
transformed_image = F.to_pil_image(transformed_tensor)
transformed_image.show()