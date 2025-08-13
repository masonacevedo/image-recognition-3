import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

import torchvision.datasets
import torchvision.transforms.functional as F
import torch.nn as nn

def show_tensor_as_image(t, batch_dimension_index=0):
    img = t[batch_dimension_index].detach().cpu().numpy()
    img = np.transpose(img, (1,2,0))
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    plt.show()


myModel = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

folder = "cifar_10_imagery"
dataset = torchvision.datasets.CIFAR10(root=folder, download=True)
print("type(dataset):", type(dataset))
print("len(dataset):", len(dataset))
print("type(dataset[0]):", type(dataset[0]))
print("type(dataset[0][0]):", type(dataset[0][0]))
print("type(dataset[0][1]):", type(dataset[0][1]))

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(myModel.parameters(), lr=0.001)

BATCH_SIZE = 16
EPOCHS = 25

myModel.train()
x_axis = []
y_axis = []
for epoch in range(0, EPOCHS):
    print("Epoch:", epoch)
    # images = [dataset[epoch*BATCH_SIZE+j][0] for j in range(0, BATCH_SIZE)]
    # for image in images:
    #     show_tensor_as_image(F.to_tensor(image).reshape(1,3,32,32))
    x = torch.stack([F.to_tensor(dataset[epoch*BATCH_SIZE+j][0]) for j in range(0, BATCH_SIZE)])
    labels = torch.tensor([dataset[epoch*BATCH_SIZE+j][1] for j in range(0, BATCH_SIZE)])
    predictions = myModel(x)
    loss = loss_function(predictions, labels)
    print("loss:", loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


    x_axis.append(epoch)
    y_axis.append(loss.item())

plt.plot(x_axis, y_axis)
plt.title("Epochs vs Loss")
plt.show()

    