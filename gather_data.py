import torchvision.datasets
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torchvision import models


folder = "cifar_10_imagery"

dataset = torchvision.datasets.CIFAR10(root=folder, download=True)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

model.eval()

PIL_image = dataset[0][0]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
])

image_as_tensor = transform(PIL_image).unsqueeze(0)

outputs = model(image_as_tensor)

print(type(outputs))
print(outputs.shape)