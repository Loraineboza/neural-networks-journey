import os
import json
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs


class DigitDataset(data.Dataset):
    def __init__(this, path, train=True, transform=None):
        this.path = os.path.join(path, "train" if train else "test") 
        this.transform = transform

        with open(os.path.join(path, "format.json"), "r") as fp:
            this.format = json.load(fp)

        print("Сохранение \"папки + метка\" в format:") #debug
        print(json.dumps(this.format, indent=4, ensure_ascii=False)) #debug

        this.length = 0
        this.files = []
        this.targets = torch.eye(10)

        for _dir, _target in this.format.items():
            print(f"this.path=\"{this.path}\"\n_dir=\"{_dir}\"") #debug
            path = os.path.join(this.path, _dir)
            print(f"path: {path}\n") #debug

            list_files = os.listdir(path)
            this.length += len(list_files)
            this.files.extend(map(lambda _x: (os.path.join(path, _x), _target), list_files))
            
    def __getitem__(this, item):
        path_file, target = this.files[item]
        t = this.targets[target]
        img = Image.open(path_file)

        return img, t

    def __len__(this):
        return this.length

if __name__== "__main__":
    d_train = DigitDataset("./dataset")
