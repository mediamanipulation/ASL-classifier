"""
Evaluation metrics and utilities
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)


def evaluate_model(model, dataloader, device, class_names, save_dir=None, save_confusion_matrix=True):
    """
    Comprehensive model evaluation

    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Data loader for evaluation
        device (torch.device): Device to use
        class_names (list): List of class names
        save_dir (str, optional): Directory to save results
        save_confusion_matrix (bool): Whether to save confusion matrix plot

    Returns:
        dict: Dictionary containing evaluation metrics
    """
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
        all_labels,
        all_preds,
        average=None,
        labels=range(len(class_names))
    )

    # Classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    if save_confusion_matrix and save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(20, 20))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )
        disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', values_format='d')
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Save per-class metrics
        with open(os.path.join(save_dir, 'per_class_metrics.txt'), 'w') as f:
            f.write("Per-Class Metrics\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
            f.write("-" * 80 + "\n")

            for i, cls_name in enumerate(class_names):
                f.write(f"{cls_name:<20} {precision[i]:>10.4f} {recall[i]:>10.4f} "
                       f"{f1[i]:>10.4f} {support[i]:>10}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n")

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': cm,
        'classification_report': report
    }

    return results


def compute_accuracy(outputs, labels):
    """
    Compute accuracy from model outputs and labels

    Args:
        outputs (torch.Tensor): Model outputs (logits)
        labels (torch.Tensor): Ground truth labels

    Returns:
        float: Accuracy as a percentage
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return 100 * correct / total
