"""
ASL Classifier model definition
"""

import torch
import torch.nn as nn
import timm


class ASLClassifier(nn.Module):
    """
    ASL Classifier using pretrained vision models

    Args:
        num_classes (int): Number of output classes
        model_name (str): Name of timm model to use
        dropout_rate (float): Dropout rate for regularization
        pretrained (bool): Whether to use pretrained weights
    """
    def __init__(self, num_classes, model_name='efficientnet_b0', dropout_rate=0.4, pretrained=True):
        super(ASLClassifier, self).__init__()

        # Create base model without classifier head
        # num_classes=0 removes the final classification layer
        self.base_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )

        # Get number of features from base model
        num_features = self.base_model.num_features

        # Custom classifier with dropout BEFORE final layer
        # This is the correct placement for effective regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Extract features from base model
        x = self.base_model(x)

        # Apply dropout for regularization
        x = self.dropout(x)

        # Final classification
        x = self.classifier(x)

        return x

    def get_num_parameters(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_parameters(self):
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
