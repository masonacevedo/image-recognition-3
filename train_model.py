import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

import torchvision.datasets
import torchvision.transforms.functional as F
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

for param in myModel.parameters():
    param.requires_grad = False

for param in myModel.fc.parameters():
    param.requires_grad = True

folder = "cifar_10_imagery"
dataset = torchvision.datasets.CIFAR10(root=folder, download=True)
# training_data, val_data = random_split(dataset, [0.8, 0.2])
# training_loader = DataLoader(training_data, batch_size=16, shuffle=True, num_workers=4)
# validation_loader = DataLoader(val_data, batch_size=16, shuffle=True, num_workers=4)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(myModel.parameters(), lr=0.001)

BATCH_SIZE = 16
EPOCHS = 1000

myModel.train()
x_axis = []
y_axis = []
for epoch in range(0, EPOCHS):
    # for batch, (data, labels) in enumerate(training_loader):
    #     data, labels = data.to(device), labels.to(device)
    # images = [dataset[epoch*BATCH_SIZE+j][0] for j in range(0, BATCH_SIZE)]
    # for image in images:
    #     show_tensor_as_image(F.to_tensor(image).reshape(1,3,32,32))
    x = torch.stack([F.to_tensor(dataset[epoch*BATCH_SIZE+j][0]) for j in range(0, BATCH_SIZE)])
    x = x.to(device)
    labels = torch.tensor([dataset[epoch*BATCH_SIZE+j][1] for j in range(0, BATCH_SIZE)])
    labels = labels.to(device)
    predictions = myModel(x)
    loss = loss_function(predictions, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


    x_axis.append(epoch)
    y_axis.append(loss.item())

plt.plot(x_axis, y_axis)
plt.title("Epochs vs Loss")
plt.show()

    