"""
ASL Classifier Training Script
Unified training script with configuration management, logging, and best practices.
"""

import os
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import timm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)


def setup_logging(save_dir):
    """Setup logging configuration"""
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ASLDataset(Dataset):
    """Wrapper around ImageFolder for ASL dataset"""
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


class ASLClassifier(nn.Module):
    """ASL Classifier with proper dropout placement"""
    def __init__(self, num_classes, model_name='efficientnet_b0', dropout_rate=0.4, pretrained=True):
        super(ASLClassifier, self).__init__()
        # Create base model without classifier head
        self.base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        # Get number of features
        num_features = self.base_model.num_features

        # Custom classifier with dropout BEFORE final layer
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def get_transforms(config, is_training=True):
    """Create data transforms based on config"""
    img_size = config['data']['image_size']

    # ImageNet normalization (matches EfficientNet pretrained expectations)
    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_training and config['augmentation']['enabled']:
        transform_list = [
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip() if config['augmentation']['horizontal_flip'] else None,
            transforms.RandomRotation(config['augmentation']['rotation_degrees']),
            transforms.ColorJitter(**config['augmentation']['color_jitter']),
        ]

        # Advanced augmentation (if configured)
        aug_config = config.get('augmentation', {})
        if aug_config.get('random_perspective', False):
            transform_list.append(
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3)
            )
        if aug_config.get('random_affine', False):
            transform_list.append(
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
            )
        if aug_config.get('gaussian_blur', False):
            transform_list.append(
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            )

        transform_list.extend([
            transforms.ToTensor(),
            imagenet_normalize,
        ])

        if aug_config.get('random_erasing', False):
            transform_list.append(
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
            )
    else:
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            imagenet_normalize,
        ]

    # Filter out None values
    transform_list = [t for t in transform_list if t is not None]

    return transforms.Compose(transform_list)


def initialize_dataloaders(config, logger):
    """Initialize datasets and dataloaders"""
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)

    train_dataset = ASLDataset(config['data']['train_dir'], transform=train_transform)
    val_dataset = ASLDataset(config['data']['valid_dir'], transform=val_transform)
    test_dataset = ASLDataset(config['data']['test_dir'], transform=val_transform)

    logger.info(f"Number of classes: {len(train_dataset.classes)}")
    logger.info(f"Classes: {train_dataset.classes}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )

    return train_loader, val_loader, test_loader, train_dataset


def get_optimizer(model, config):
    """Get optimizer based on config"""
    opt_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']

    if opt_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(optimizer, config):
    """Get learning rate scheduler based on config"""
    sched_type = config['training']['scheduler']['type'].lower()

    if sched_type == 'reduce_on_plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config['training']['scheduler']['patience'],
            factor=config['training']['scheduler']['factor'],
            min_lr=config['training']['scheduler']['min_lr']
        )
    elif sched_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    elif sched_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def evaluate_model(model, dataloader, device, class_names, save_dir, config):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(len(class_names))
    )

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)

    # Confusion matrix
    if config['evaluation']['save_confusion_matrix']:
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        fig, ax = plt.subplots(figsize=(20, 20))
        disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()

    return accuracy, report, precision, recall, f1, support


def train_model(config, logger):
    """Main training function"""
    # Setup
    device = torch.device(config['experiment']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create save directories
    save_dir = os.path.join(config['logging']['save_dir'], config['experiment']['name'])
    checkpoint_dir = os.path.join(config['logging']['checkpoint_dir'], config['experiment']['name'])
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # TensorBoard
    writer = None
    if config['logging']['tensorboard']:
        writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))

    # Load data
    train_loader, val_loader, test_loader, train_dataset = initialize_dataloaders(config, logger)
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes

    # Save class names
    with open(os.path.join(save_dir, 'class_names.pkl'), 'wb') as f:
        pickle.dump(class_names, f)

    # Initialize model
    model = ASLClassifier(
        num_classes=num_classes,
        model_name=config['model']['architecture'],
        dropout_rate=config['model']['dropout_rate'],
        pretrained=config['model']['pretrained']
    )
    model.to(device)
    logger.info(f"Model: {config['model']['architecture']}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Freeze backbone layers if configured
    freeze_backbone = config['model'].get('freeze_backbone', False)
    if freeze_backbone:
        for param in model.base_model.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Backbone frozen. Trainable parameters: {trainable:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if config['mixed_precision']['enabled'] else None

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    no_improvement_epochs = 0

    logger.info("Starting training...")

    unfreeze_after = config['model'].get('unfreeze_after_epochs', 0)

    for epoch in range(config['training']['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        logger.info("-" * 50)

        # Unfreeze backbone after N epochs for fine-tuning
        if freeze_backbone and unfreeze_after > 0 and epoch == unfreeze_after:
            for param in model.base_model.parameters():
                param.requires_grad = True
            # Rebuild optimizer with all parameters and lower LR
            unfreeze_lr = config['training']['learning_rate'] * 0.1
            optimizer = get_optimizer(model, config)
            for pg in optimizer.param_groups:
                pg['lr'] = unfreeze_lr
            scheduler = get_scheduler(optimizer, config)
            logger.info(f"Backbone unfrozen at epoch {epoch+1}, LR set to {unfreeze_lr:.6f}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Learning rate scheduling
        if config['training']['scheduler']['type'].lower() == 'reduce_on_plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            no_improvement_epochs = 0

            if config['logging']['save_best_model']:
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'config': config
                }, best_model_path)
                logger.info(f"Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            no_improvement_epochs += 1

        # Early stopping
        if config['training']['early_stopping']['enabled']:
            if no_improvement_epochs >= config['training']['early_stopping']['patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Save last model
    if config['logging']['save_last_model']:
        last_model_path = os.path.join(checkpoint_dir, 'last_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': config
        }, last_model_path)

    # Plot training curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()

    # Load best model for final evaluation
    logger.info("\nLoading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    test_acc, report, precision, recall, f1, support = evaluate_model(
        model, test_loader, device, class_names, save_dir, config
    )

    logger.info(f"\nTest Accuracy: {test_acc*100:.2f}%")
    logger.info("\nClassification Report:")
    logger.info(report)

    # Save metrics
    with open(os.path.join(save_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    if writer:
        writer.close()

    logger.info(f"\nTraining complete! Results saved to {save_dir}")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Test accuracy: {test_acc*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Train ASL Classifier')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logging
    logger = setup_logging(config['logging']['save_dir'])
    logger.info("="*50)
    logger.info("ASL Classifier Training")
    logger.info("="*50)

    # Set seed
    set_seed(config['experiment']['seed'])
    logger.info(f"Random seed set to {config['experiment']['seed']}")

    # Print system info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Train model
    train_model(config, logger)


if __name__ == '__main__':
    main()
