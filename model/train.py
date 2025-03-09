import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import ViTModel, ViTConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import cv2
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from dataset_preprocessing import VideoFrameDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the deepfake detection model
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetector, self).__init__()
        
        # CNN Feature Extractor (EfficientNet)
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.classifier = nn.Identity()  # Remove classifier to get features
        
        # Vision Transformer for temporal analysis
        self.vit_config = ViTConfig(hidden_size=1280, num_hidden_layers=6, num_attention_heads=8)
        self.transformer = ViTModel(self.vit_config)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Freeze early layers of EfficientNet
        for param in list(self.efficientnet.parameters())[:-10]:
            param.requires_grad = False
    
    def forward(self, x_frames):
        batch_size, seq_len, c, h, w = x_frames.shape
        
        # Reshape to process each frame through CNN
        x_frames = x_frames.view(batch_size * seq_len, c, h, w)
        
        # Extract CNN features
        cnn_features = self.efficientnet(x_frames)  # [batch_size*seq_len, 1280]
        
        # Reshape for transformer
        cnn_features = cnn_features.view(batch_size, seq_len, -1)  # [batch_size, seq_len, 1280]
        
        # Pass through transformer
        transformer_output = self.transformer(inputs_embeds=cnn_features).last_hidden_state
        
        # Use the [CLS] equivalent token from first position
        cls_output = transformer_output[:, 0, :]
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits

def train_model(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = VideoFrameDataset(
        root_dir=os.path.join(args.dataset_path, 'train'),
        num_frames=args.num_frames,
        transform=transform
    )
    
    val_dataset = VideoFrameDataset(
        root_dir=os.path.join(args.dataset_path, 'val'),
        num_frames=args.num_frames,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    logging.info(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
    
    # Create model
    model = DeepfakeDetector(num_classes=2)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for frames, labels in tqdm(train_loader, desc="Training"):
            frames = frames.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * frames.size(0)
            
            # Store predictions and targets for metrics
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        logging.info(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for frames, labels in tqdm(val_loader, desc="Validation"):
                frames = frames.to(device)
                labels = labels.to(device)
                
                outputs = model(frames)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * frames.size(0)
                
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='binary')
        
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logging.info(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'deepfake_model.pth'))
            logging.info(f"Saved new best model with validation accuracy: {val_acc:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    logging.info(f"Training curves saved to {os.path.join(args.output_dir, 'training_curves.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory for model and results")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames to sample from each video")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for regularization")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads for data loading")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_model(args)