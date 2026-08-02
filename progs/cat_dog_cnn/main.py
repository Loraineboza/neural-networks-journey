from PIL import Image
import torch

img = Image.open("./dataset/train/Cat/1.jpg")
print(type(img))

tensor = torch.tensor(img)