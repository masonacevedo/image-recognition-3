import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt


def show_tensor_as_image(t, batch_dimension_index=0):
    img = t[batch_dimension_index].detach().cpu().numpy()
    img = np.transpose(img, (1,2,0))
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    plt.show()


resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet18.eval()
random_image = torch.randn(1, 3, 200, 200)
result_logits = resnet18(random_image)

result_probs = torch.softmax(result_logits, dim=1)
top_5_probs, top_5_categories = torch.topk(result_probs, 5)

category_names = models.ResNet18_Weights.DEFAULT.meta["categories"]
for index in list(top_5_categories[0]):
    print(category_names[index])
