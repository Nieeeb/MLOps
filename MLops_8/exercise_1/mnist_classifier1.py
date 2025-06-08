import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.ToTensor()

# Load MNIST
full_train = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
full_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)


# Filter digits 0–4
def filter_digits(dataset, max_digit=4):
    indices = [i for i, (_, label) in enumerate(dataset) if label <= max_digit]
    return Subset(dataset, indices)


train_dataset = filter_digits(full_train)
test_dataset = filter_digits(full_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# Define model with 10 outputs (0–9)
class MNISTClassifier10(nn.Module):
    def __init__(self):
        super(MNISTClassifier10, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # 10 outputs for 0–9
        )

    def forward(self, x):
        return self.network(x)


model = MNISTClassifier10().to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# Save model
# os.makedirs("checkpoints", exist_ok=True)
# torch.save(model.state_dict(), "checkpoints/mnist_0_to_4.pth")
# print("Model saved to checkpoints/mnist_0_to_4.pth")

# Evaluate
model.eval()
all_preds = []
all_labels = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"\nAccuracy on digits 0–4 test set: {accuracy:.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, labels=[0, 1, 2, 3, 4]))

# Confusion matrix
conf_mat = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4])
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_mat,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(5),
    yticklabels=range(5),
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix: Digits 0–4")
plt.tight_layout()
plt.show()
