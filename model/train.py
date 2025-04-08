import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy

# --- Configuration ---
data_dir = './gemini_images' 
model_save_path = './deepfake_detector_resnet18.pth'
num_epochs = 15 
batch_size = 32 
learning_rate = 0.001
use_pretrained = True # Use transfer learning

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Loading and Preprocessing ---

# Data augmentation and normalization for training
# Just normalization for validation/testing
# Use standard ImageNet means and stds as we use a model pretrained on ImageNet
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Add 'test' transform if you have a test set
    # 'test': transforms.Compose([...])
}

print("Initializing Datasets and Dataloaders...")

# Create datasets using ImageFolder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'validation']} 

# Create dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True if x == 'train' else False, num_workers=4)
               for x in ['train', 'validation']} 

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

print(f"Classes: {class_names}")
print(f"Dataset sizes: {dataset_sizes}")
if num_classes != 2:
    print("Warning: Expected 2 classes (real, fake), but found", num_classes)

# --- Model Definition ---

# Load a pre-trained ResNet18 model
model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if use_pretrained else None)

# Get the number of input features for the classifier
num_ftrs = model_ft.fc.in_features

# Replace the final fully connected layer for binary classification
# Output layer will have 1 neuron (probability of being 'fake' after sigmoid)
# Or 2 neurons for ('fake', 'real') probabilities with CrossEntropyLoss
# Using 2 neurons here as it's common with CrossEntropyLoss
model_ft.fc = nn.Linear(num_ftrs, num_classes)

model_ft = model_ft.to(device)

# --- Loss Function and Optimizer ---

# Use CrossEntropyLoss as we have 2 output neurons
criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)

# Decay LR by a factor of 0.1 every 7 epochs (optional)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# --- Training Function ---

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # Get the class index with the highest score
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step() # Step the learning rate scheduler

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item()) # Use .item() to get Python number
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item()) # Use .item() to get Python number

                # Deep copy the model if it's the best validation accuracy so far
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Save the best model weights immediately
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Best model saved to {model_save_path} with accuracy: {best_acc:.4f}")


        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# --- Start Training ---
print("Starting Training...")
model_ft, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                num_epochs=num_epochs)

print("Training Finished!")
print(f"Model saved to: {model_save_path}")

# --- (Optional) Plot Training History ---
def plot_history(history):
    epochs = range(len(history['train_loss']))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    # Convert accuracy tensors to floats if they are not already
    train_acc = [acc for acc in history['train_acc']]
    val_acc = [acc for acc in history['val_acc']]
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# plot_history(history) # Uncomment to plot graphs after training

