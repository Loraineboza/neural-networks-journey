import os
import json

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class DogDataset(data.Dataset):
    def __init__(this, path, train=True, transform=None):
        this.pathi = os.path.join(path, "train" if train else "test")
        this.transform = transform

        with open(os.path.join(this.path, "format.json"), "r") as fp:
            this.format = json.load(fp)

        this.length = 0
        this.files = []
        this.targets = torch.eye(10)

        for _dir, _target in this.format.items():
            path = os.path.join(this.path, _dir)
            list_files = os.listdir(path)
            this.length += len(list_files)
            this.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))

        def __getitem__(this, item):
            path_file, target = this.files[item]
            t = this.targets[target]
            img = Image.open(path_file)

            if this.transform:
                img = this.transform(img)

            return img, t

        def __len__(this):
            return this.length

