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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define model (same as in Task 1)
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.network(x)


# Load model
model = MNISTClassifier().to(device)
model.load_state_dict(torch.load("checkpoints/mnist_model.pth"))
print("✅ Loaded pre-trained model.")
model.train()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=5e-5)

# Load MNIST training data (same transform as before)
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

# Target class to forget
target_class = 7

# Get all samples of class "7"
indices = [
    i for i, (_, label) in enumerate(train_dataset) if label == target_class
]
unlearn_dataset = Subset(train_dataset, indices)
unlearn_loader = DataLoader(unlearn_dataset, batch_size=64, shuffle=True)

# Perform gradient ascent (increase loss for class 7)
epochs_unlearn = 25
for epoch in range(epochs_unlearn):
    total_loss = 0
    for images, labels in unlearn_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        (-loss).backward()  # Gradient ascent
        optimizer.step()
        total_loss += loss.item()

    print(f"[Unlearn Epoch {epoch+1}] Total Loss Increased: {total_loss:.4f}")

# Save updated model
torch.save(model.state_dict(), "checkpoints/mnist_model_unlearned.pth")
print(
    "\n Model after unlearning saved to checkpoints/mnist_model_unlearned.pth"
)

# Evaluate on test set
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print results
accuracy = 100 * correct / total
print(f"\n Accuracy after unlearning: {accuracy:.2f}%")

print("\nUpdated Classification Report:\n")
print(classification_report(all_labels, all_preds))

# Plot confusion matrix
conf_mat = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix After Unlearning Class 7")
plt.tight_layout()
plt.show()
