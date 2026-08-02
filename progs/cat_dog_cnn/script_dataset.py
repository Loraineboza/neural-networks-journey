import os
import shutil
import random

src_dir = "./PetImages"          # твоя папка с Cat и Dog
dst_dir = "./dataset"            # куда будем складывать разбитые данные

split_ratio = [0.8, 0.1, 0.1]    # 80% train, 10% val, 10% test

for class_name in ["Cat", "Dog"]:
    class_path = os.path.join(src_dir, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * split_ratio[0])
    n_val   = int(n_total * split_ratio[1])

    train_imgs = images[:n_train]
    val_imgs   = images[n_train:n_train + n_val]
    test_imgs  = images[n_train + n_val:]

    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        split_dir = os.path.join(dst_dir, split_name, class_name)
        os.makedirs(split_dir, exist_ok=True)
        for img in split_imgs:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(split_dir, img)
            shutil.move(src_path, dst_path)