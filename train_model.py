import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

import torchvision.datasets

import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import torch.nn as nn

from torch.utils.data import random_split, DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def show_tensor_as_image(t, batch_dimension_index=0):
    img = t[batch_dimension_index].detach().cpu().numpy()
    img = np.transpose(img, (1,2,0))
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    plt.show()

myModel = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
myModel.fc = nn.Linear(myModel.fc.in_features, 10)
myModel.to(device)

# for param in myModel.parameters():
#     param.requires_grad = False

# for param in myModel.fc.parameters():
#     param.requires_grad = True

folder = "cifar_10_imagery"
raw_dataset = torchvision.datasets.CIFAR10(root=folder, download=True)
raw_training_data, val_data, _ = random_split(raw_dataset, [0.80, 0.20, 0.0])

# TODO: ADD NORMALIZATION!
training_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.ToTensor(),
])

validation_transform = transforms.ToTensor()

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label



transformed_training_data = TransformedDataset(raw_training_data, training_transform)
transformed_validation_data = TransformedDataset(val_data, validation_transform)

print("Type of transformed_training_data[0]:", type(transformed_training_data[0]))
print("Type of transformed_training_data[0][0]:", type(transformed_training_data[0][0]))
print("Type of transformed_training_data[0][1]:", type(transformed_training_data[0][1]))

training_loader = DataLoader(transformed_training_data, batch_size=16, shuffle=True, num_workers=0)
validation_loader = DataLoader(transformed_training_data, batch_size=16, shuffle=True, num_workers=0)



loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(myModel.parameters(), lr=0.001)

BATCH_SIZE = 32
EPOCHS = 30

epochs_list = []
losses = []
accuracies = []
for epoch in range(0, EPOCHS):
    print("Epoch:", epoch)
    myModel.train()
    for batch, (images, labels) in enumerate(training_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = myModel(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()

    num_correct = 0
    num_seen = 0
    epoch_loss = 0.0
    myModel.eval()
    with torch.no_grad():
        for batch, (images, labels) in enumerate(validation_loader):
            images, labels = images.to(device), labels.to(device)
            logits = myModel(images)
            predictions = torch.argmax(logits, dim=1)
            num_correct += (predictions == labels).sum().item()
            num_seen += len(labels)
            loss = loss_function(logits, labels)
            epoch_loss += loss.item()
    
    # Calculate average loss for this epoch
    avg_epoch_loss = epoch_loss / len(validation_loader)
    losses.append(avg_epoch_loss)
    
    accuracy = float(num_correct) / float(num_seen)
    accuracies.append(accuracy)
    
    epochs_list.append(epoch)
    print(f"Epoch {epoch}: Loss = {avg_epoch_loss:.4f}, Accuracy = {accuracy:.4f}")
    

plt.figure(figsize=(15, 6))


plt.subplot(1, 2, 1)
plt.plot(epochs_list, losses, 'r-', linewidth=2, marker='o')
plt.title('Training Loss vs Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(epochs_list[::5] if len(epochs_list) > 10 else epochs_list)


plt.subplot(1, 2, 2)
plt.plot(epochs_list, accuracies, 'g-', linewidth=2, marker='s')
plt.title('Validation Accuracy vs Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(epochs_list[::5] if len(epochs_list) > 10 else epochs_list)

plt.tight_layout()
plt.show()




    