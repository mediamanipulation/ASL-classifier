import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision  # Add this import
import timm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from tqdm import tqdm
import pickle


# Set seed for reproducibility
torch.manual_seed(42)

# Check system configurations
def print_system_info():
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU.")

print_system_info()

# Dataset class wrapper around ImageFolder
class ASLDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

# Data Augmentation Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

# Initialize datasets and data loaders
def initialize_datasets(train_dir, valid_dir, test_dir, transform):
    train_dataset = ASLDataset(train_dir, transform=transform)
    val_dataset = ASLDataset(valid_dir, transform=transform)
    test_dataset = ASLDataset(test_dir, transform=transform)

    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader, test_loader, train_dataset

# Paths to dataset folders
train_dir = './input/train'
valid_dir = './input/valid'
test_dir = './input/test'

train_loader, val_loader, test_loader, train_dataset = initialize_datasets(
    train_dir, valid_dir, test_dir, transform
)

# Define ASLClassifier model using timm
class ASLClassifier(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0'):
        super(ASLClassifier, self).__init__()
        self.base_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.base_model(x)

# Initialize model
num_classes = len(train_dataset.classes)
model = ASLClassifier(num_classes=num_classes, model_name='efficientnet_b0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# Scheduler for learning rate reduction
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.3)

# Training and validation loop
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=25):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'asl_classifier_best.pth')
            print("Saved the best model.")

        scheduler.step(val_loss)

    return train_losses, val_losses

train_losses, val_losses = train_and_validate(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=25
)

# Plot training and validation losses
def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.show()

plot_losses(train_losses, val_losses)

# Evaluate the model on test data
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

evaluate_model(model, test_loader, device)

# Save class names for later use
with open('class_names.pkl', 'wb') as f:
    pickle.dump(train_dataset.classes, f)

# Load and predict on a single image
def predict_image(model, image_path, transform, class_names, device):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_idx = np.argmax(probs)
        print(f"Predicted: {class_names[predicted_idx]} with probability {probs[predicted_idx]:.2f}")
        return class_names[predicted_idx], probs

# Example usage
image_path = './img/test2.jpg'
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)
predict_image(model, image_path, transform, class_names, device)
b  