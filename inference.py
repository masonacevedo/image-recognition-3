import torch

from PIL import Image

import torchvision.transforms as transforms
import torch.nn.functional as F

import os

from train_model import show_tensor_as_image

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
    transforms.Resize((32, 32)),
    # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])

model_base_name = input("model base name:")
all_models = os.listdir("models")

raw_model_paths = list(filter(lambda s: model_base_name in s, all_models))
model_paths = [os.path.join("models", name) for name in raw_model_paths]

# sort according to epoch number at the end
model_paths.sort(key=lambda x: int(x.split("_epoch_")[-1].split(".pth")[0]))

for model_index, model in enumerate(model_paths):
    correct_count = 0
    images_as_tensors = [transforms(image).to(device) for image in images]
    images_as_tensors = torch.stack(images_as_tensors)

    myModel = torch.load(model, weights_only=False)
    myModel.eval()

    myModel.to(device)

    with torch.no_grad():
        logits = myModel(images_as_tensors)
        probabilities = F.softmax(logits, dim=1)
        # print("probabilities: ", probabilities)

        predicted_indices = torch.argmax(logits, dim=1)
        # print("predicted_indices: ", predicted_indices)
        # input("Press Enter to continue...")

    for predicted_index in predicted_indices:
        predicted_category = CIFAR10_CLASSES[predicted_index]
        if predicted_category == "bird":
            correct_count += 1

    print(f"{model}:{correct_count}/{len(images)}")
    print()






