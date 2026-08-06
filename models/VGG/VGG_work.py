from PIL import Image
from torchvision import models
import torch
import torchvision.transforms.v2 as v2

 
vgg_weights = models.VGG16_Weights.DEFAULT
cats = vgg_weights.meta['categories']

# transforms_1 = vgg_weights.transforms()

transforms_2 = v2.Compose([
    v2.RandomResizedCrop(size=(224, 244), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]) 

img_net = transforms_2(img).unsqueeze(0)

model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

model.eval()
p = model(img_net).squeeze() # (1000)
res =  p.softmax(dim=0).sort(descending=True)

for s, i in zip(res[0][:5], res[1][:5]):
    print(f"{cats[i]}: {s:.4f}")
