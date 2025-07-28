import torchvision.datasets
from torch.utils.data import Dataset


folder = "cifar_10_imagery"

dataset = torchvision.datasets.CIFAR10(root=folder, download=True)

image, label = dataset[0]
print(f"type(image): {type(image)}")
image.show()
