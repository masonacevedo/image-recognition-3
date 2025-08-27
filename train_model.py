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

import time
import glob
import os


class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform=None):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            image, label = self.dataset[index]
            return self.transform(image), label

if __name__ == "__main__":
    torch.manual_seed(42)
    last_update = time.time()

    model_name = input("Model Name (for saving):")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if glob.glob(f"models/{model_name}_epoch_*.pth"):
        print("Loading model from existing models...")
        all_models = os.listdir("models")
        raw_model_paths = list(filter(lambda s: model_name in s, all_models))
        model_paths = [os.path.join("models", name) for name in raw_model_paths]
        model_paths.sort(key=lambda x: int(x.split("_epoch_")[-1].split(".pth")[0]))
        myModel = torch.load(model_paths[-1], weights_only=False)
        print(f"Loaded model from {model_paths[-1]}")
        epochs_already_trained = len(model_paths)
    else:
        print("Creating new model...")
        myModel = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        myModel.fc = nn.Linear(myModel.fc.in_features, 10)
        myModel.to(device)
        epochs_already_trained = 0

    folder = "cifar_10_imagery"
    raw_dataset = torchvision.datasets.CIFAR10(root=folder, download=True)
    raw_training_data, val_data, _ = random_split(raw_dataset, [0.80, 0.20, 0.0])

    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.4),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    validation_transform = transforms.ToTensor()

    transformed_training_data = TransformedDataset(raw_training_data, training_transform)
    transformed_validation_data = TransformedDataset(val_data, validation_transform)

    BATCH_SIZE = 32
    NUM_WORKERS = 0
    training_loader = DataLoader(transformed_training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    validation_loader = DataLoader(transformed_training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)



    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(myModel.parameters(), lr=0.001)

    EPOCHS = 200

    epochs_list = []
    losses = []
    accuracies = []
    for epoch in range(epochs_already_trained, EPOCHS):
        myModel.train()
        for batch, (images, labels) in enumerate(training_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = myModel(images)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
            if time.time() - last_update > 30:
                last_update = time.time()
                print(f"Epoch {epoch} | Batch {batch} | Loss: {loss.item()}")

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
        print(f"Epoch {epoch} | Loss: {avg_epoch_loss:.4f} | Accuracy: {accuracy:.4f}")

        torch.save(myModel, f"models/{model_name}_epoch_{epoch}.pth")


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




        