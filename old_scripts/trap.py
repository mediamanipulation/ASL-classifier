# train_asl_classifier_refactored.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

# Set random seed for reproducibility
torch.manual_seed(42)

# Check system versions and configurations
def print_system_info():
    import sys
    import torchvision
    print('System Version:', sys.version)
    print('PyTorch version:', torch.__version__)
    print('Torchvision version:', torchvision.__version__)
    print('Numpy version:', np.__version__)
    print('CUDA Available:', torch.cuda.is_available())

    if torch.cuda.is_available():
        print('CUDA Version:', torch.version.cuda)
        print('Number of GPUs Available:', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i} - Name: {torch.cuda.get_device_name(i)}')
    else:
        print('CUDA is not available. Please check your installation.')

print_system_info()

# Step 1: Define the Dataset Class
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

# Define Transformations for Data Augmentation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Lambda(lambda image: image.convert('RGB')),  # Ensure images are in RGB format
    transforms.ToTensor(),
])

# Step 1.2: Initialize Dataset and DataLoader
def initialize_datasets(train_folder, valid_folder, test_folder, transform):
    train_dataset = ASLDataset(train_folder, transform=transform)
    val_dataset = ASLDataset(valid_folder, transform=transform)
    test_dataset = ASLDataset(test_folder, transform=transform)

    # Print number of classes and class names
    print(f"Number of classes in training dataset: {len(train_dataset.classes)}")
    print("Class names:", train_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset

# Replace these paths with the actual paths to your datasets
train_folder = './input/train'
valid_folder = './input/valid'
test_folder = './input/test'

train_loader, val_loader, test_loader, train_dataset = initialize_datasets(train_folder, valid_folder, test_folder, transform)

# Get the actual number of classes from the dataset
num_classes = len(train_dataset.classes)

# Step 2: Define the PyTorch Model
class ASLClassifier(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0'):
        super(ASLClassifier, self).__init__()
        # Create the model with the specified number of classes
        self.base_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.dropout = nn.Dropout(0.3)  # Reduce dropout for regularization

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        return x

# Initialize Model
model_name = 'efficientnet_b0'  # Change model name here if needed
model = ASLClassifier(num_classes=num_classes, model_name=model_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model is on: {next(model.parameters()).device}")  # Confirm model is on GPU or CPU

# Step 3: Define Training Loop Components
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 3.1: Define Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)  # Increase patience to 2

# Step 3.2: Training and Validation Loop with Early Stopping
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=15):  # Increase epochs to 15
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    no_improvement_epochs = 0
    patience = 2  # Number of epochs with no improvement after which training will be stopped

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Training loop Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation Phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Validation loop Epoch {epoch+1}/{num_epochs}'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Step scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= patience:
                print("Early stopping triggered.")
                break
    
    return train_losses, val_losses

train_losses, val_losses = train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=15)  # Increase epochs to 15

# Step 4: Visualize Training Results
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

plot_losses(train_losses, val_losses)

# Step 5: Evaluate Model on Test Data
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

def visualize_predictions(original_image, probabilities, class_names):
    # Get top 5 predictions
    top5_prob_indices = probabilities.argsort()[-5:][::-1]
    top5_probabilities = probabilities[top5_prob_indices]
    top5_class_names = [class_names[i] for i in top5_prob_indices]

    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))
    
    axarr[0].imshow(original_image)
    axarr[0].axis("off")
    
    axarr[1].barh(top5_class_names[::-1], top5_probabilities[::-1])
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Top 5 Class Predictions")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()

# Example usage
# Replace with your test image path
test_image = './input/test/a/hand1_a_bot_seg_1_cropped.jpeg'

# Ensure the same transform is used
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Lambda(lambda image: image.convert('RGB')),  # Ensure images are in RGB format
    transforms.ToTensor(),
])

original_image, image_tensor = preprocess_image(test_image, test_transform)
probabilities = predict(model, image_tensor, device)

# Assuming train_dataset.classes gives the class names
class_names = train_dataset.classes
visualize_predictions(original_image, probabilities, class_names)

# Test on multiple random images
def test_random_images(test_folder, transform, model, device):
    # Adjust the path to your test images
    test_images = glob(os.path.join(test_folder, '*', '*'))
    test_examples = np.random.choice(test_images, 5)  # Test on 5 random images
    
    for example in test_examples:
        original_image, image_tensor = preprocess_image(example, transform)
        probabilities = predict(model, image_tensor, device)
        
        # Assuming train_dataset.classes gives the class names
        class_names = train_dataset.classes
        visualize_predictions(original_image, probabilities, class_names)

test_random_images(test_folder, test_transform, model, device)









