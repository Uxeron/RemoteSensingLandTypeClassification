import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision
from PIL import Image, ImageEnhance
import numpy as np
from matplotlib import pyplot as plt
import os
import time

CLASSES = (
    "Unknown",
    "Water (Permanent)",
    "Artificial Bare Ground",
    "Natural Bare Ground",
    "Snow/Ice (Permanent)",
    "Woody",
    "Non-Woody Cultivated",
    "Non-Woody (Semi) Natural"
)

dir_name = os.path.dirname(__file__)

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    ])
    return transform(image)

def mark_different_classes(image1, image2, class1, class2):
    mask1 = np.array(image1)
    mask2 = np.array(image2)

    plt.imshow(mask1.squeeze())
    plt.axis('off')
    plt.show()
    
    plt.imshow(mask2.squeeze())
    plt.axis('off')
    plt.show()

    different_mask = np.ones_like(mask1) * 255
    different_mask[(mask1 == class1) & (mask2 == class2)] = 0

    plt.imshow(different_mask.squeeze(0))
    plt.axis('off')
    plt.show()

    return different_mask



def get_masks(image1_path, image2_path):

    image1 = Image.open(image1_path).convert("RGB")
    image2 = Image.open(image2_path).convert("RGB")

    image1_tensor = preprocess_image(image1).unsqueeze(0)
    image2_tensor = preprocess_image(image2).unsqueeze(0)

    model_path = os.path.join(dir_name, "model_true_color.pt")

    model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        output1 = model(image1_tensor)['out'].argmax(dim=1)
        output2 = model(image2_tensor)['out'].argmax(dim=1)
    class1 = 6
    class2 = 2 

    different_areas = mark_different_classes(output1, output2, class1, class2)

    timestamp = int(time.time())
    maskpath = os.path.join(dir_name, 'static', f'mask{timestamp}.jpg')

    different_areas_image = Image.fromarray(different_areas.squeeze().astype(np.uint8))
    different_areas_image.show()
    different_areas_image.save(maskpath)
    
    return maskpath
