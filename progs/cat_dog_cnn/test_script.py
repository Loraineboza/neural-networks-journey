import os

for split in ["train", "val", "test"]:
    for cls in ["Cat", "Dog"]:
        path = f"./dataset/{split}/{cls}"
        if os.path.exists(path):
            print(f"{path}: {len(os.listdir(path))} файлов")
        else:
            print(f"{path}: не найдено")