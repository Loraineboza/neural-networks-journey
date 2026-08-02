import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = datasets.ImageFolder("./dataset/train", transform=transform)
val_dataset = datasets.ImageFolder("./dataset/val", transform=transform)
test_dataset = datasets.ImageFolder("./dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train: {len(train_dataset)}")
print(f"Val: {len(val_dataset)}")
print(f"Test: {len(test_dataset)}")
print(f"Классы: {train_dataset.classes}")

class SimpleCNN(nn.Module):
    def __init__(this):
        super().__init__()
        this.conv1 = nn.Conv2d(in_channels=3, out_channels=16, 
                               kernel_size=3, padding=1)
        
        this.conv2 = nn.Conv2d(in_channels=16, out_channels=32, 
                               kernel_size=3, padding=1)
        
        this.conv3 = nn.Conv2d(in_channels=32, out_channels=64, 
                               kernel_size=3, padding=1)
        
        this.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        this.fc1 = nn.Linear(64 * 8 * 8, 128)
        this.fc2 = nn.Linear(128, 2)

    def forward(this, x):
        x = this.pool(torch.relu(this.conv1(x)))
        x = this.pool(torch.relu(this.conv2(x)))
        x = this.pool(torch.relu(this.conv3(x)))

        x = x.view(x.size(0), -1)
        x = torch.relu(this.fc1(x))
        x = this.fc2(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train

epochs = 10

patience = 5
best_val_acc = 0.0
patience_counter = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    train_acc = 100 * correct / total

    # validation

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            val_total +=  labels.size(0)
            val_correct += (pred == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    print(f"Эпоха {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "./best_model.pth")
        print("save model")
    else:
        patience_counter += 1
        print(f"без улучшений ({patience_counter}/{patience})")
    if patience_counter >= patience:
        print(f"stopping train на эпохе {epoch+1}")
        break

model.load_state_dict(torch.load("./best_model.pth"))
print("load the best model")
    
#eval

model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, pred = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (pred == labels).sum().item()

print(f"\nТочность на тесте: {100 * test_correct / test_total:.2f}%")