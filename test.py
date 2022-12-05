import torchvision
import torch
import torch.nn as nn
from msdnet import MSDNet
from CondenseNet_converted import CondenseNet
import numpy as np
from torchvision import transforms
from PIL import Image
from ax import optimize

img = Image.open("C:/Users/KPS/Desktop/download.jfif")

convert_tensor = transforms.ToTensor()

x = convert_tensor(img)
model = torchvision.models.densenet121(pretrained=True)

x = torch.unsqueeze(x, 0)
model.eval()
print(x.shape)
pred = model(x)
print(torch.argmax(pred))
