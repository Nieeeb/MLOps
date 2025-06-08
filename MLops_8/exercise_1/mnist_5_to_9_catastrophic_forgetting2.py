import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.ToTensor()
full_train = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
full_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)


# Helper: filter dataset by label range
def filter_digits(dataset, min_digit, max_digit):
    indices = [
        i
        for i, (_, label) in enumerate(dataset)
        if min_digit <= label <= max_digit
    ]
    return Subset(dataset, indices)


# Split datasets
train_5to9 = filter_digits(full_train, 5, 9)
test_0to4 = filter_digits(full_test, 0, 4)
test_5to9 = filter_digits(full_test, 5, 9)

train_loader_5to9 = DataLoader(train_5to9, batch_size=64, shuffle=True)
test_loader_0to4 = DataLoader(test_0to4, batch_size=1000, shuffle=False)
test_loader_5to9 = DataLoader(test_5to9, batch_size=1000, shuffle=False)


# Define model (must match earlier architecture)
class MNISTSmallClassifier(nn.Module):
    def __init__(self):
        super(MNISTSmallClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(
                64, 10
            ),  # Full output (0–9), though only trained on 5–9 here
        )

    def forward(self, x):
        return self.network(x)


# Load previously trained model on 0–4
model = MNISTSmallClassifier().to(device)
model.load_state_dict(torch.load("checkpoints/mnist_0_to_4.pth"))
print("Loaded model trained on digits 0–4.")

# Modify last layer (if needed) — but in this case, we used 10 outputs originally
model.train()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Track accuracy on 0–4 and 5–9
acc_0to4 = []
acc_5to9 = []

# Training on digits 5–9
num_epochs = 50
for epoch in range(num_epochs):
    for images, labels in train_loader_5to9:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation at the end of each epoch
    model.eval()

    def eval_accuracy(loader, label_range):
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                mask = (labels >= label_range[0]) & (labels <= label_range[1])
                all_preds.extend(preds[mask].cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())
        return accuracy_score(all_labels, all_preds)

    acc_old = eval_accuracy(test_loader_0to4, (0, 4))
    acc_new = eval_accuracy(test_loader_5to9, (5, 9))

    acc_0to4.append(acc_old)
    acc_5to9.append(acc_new)

    print(
        f"Epoch {epoch+1}: Accuracy on 0–4 = {acc_old:.4f}, on 5–9 = {acc_new:.4f}"
    )
    model.train()

# Plot catastrophic forgetting
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), acc_0to4, label="Accuracy on 0–4 (old)")
plt.plot(range(1, num_epochs + 1), acc_5to9, label="Accuracy on 5–9 (new)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Catastrophic Forgetting: Accuracy Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("checkpoints/forgetting_plot.png")
plt.show()
