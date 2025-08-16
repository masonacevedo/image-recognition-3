import torch

from PIL import Image

import torchvision.transforms as transforms
import torch.nn.functional as F

import os

# CIFAR-10 class labels in the correct order
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

files = os.listdir("inference_images")
paths = [os.path.join("inference_images", file) for file in files]
images = [Image.open(path) for path in paths]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])

for image in images:
    print(f"Image #{images.index(image)}")
    image_as_tensor = transforms(image).unsqueeze(0).to(device)

    myModel = torch.load("models/90_percent_no_augmentation_with_normalization_epoch_29.pth", weights_only=False)
    myModel.eval()

    myModel.to(device)

    with torch.no_grad():
        logits = myModel(image_as_tensor)
        probabilities = F.softmax(logits, dim=1)
        
        predicted_index = torch.argmax(logits, dim=1).item()
        predicted_category = CIFAR10_CLASSES[predicted_index]
        
        print(f"Raw logits: {logits}")
        print(f"Probabilities: {probabilities}")
        print(f"Predicted index: {predicted_index}")
        print(f"Predicted category: {predicted_category}")
        # print(f"P(bird): {probabilities[0][2]}")
        
        print("\nAll probabilities:")
        for i, (class_name, prob) in enumerate(zip(CIFAR10_CLASSES, probabilities[0])):
            print(f"{i}: {class_name}: {prob:.4f}")







