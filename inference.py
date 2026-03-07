"""
ASL Classifier Inference Script
Make predictions on images using trained model.
"""

import os
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import timm


class ASLClassifier(nn.Module):
    """ASL Classifier model (must match training architecture)"""
    def __init__(self, num_classes, model_name='efficientnet_b0', dropout_rate=0.4, pretrained=False):
        super(ASLClassifier, self).__init__()
        self.base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        num_features = self.base_model.num_features
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def load_model(checkpoint_path, class_names, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config
    config = checkpoint.get('config', {})
    model_name = config.get('model', {}).get('architecture', 'efficientnet_b0')
    dropout_rate = config.get('model', {}).get('dropout_rate', 0.4)

    # Initialize model
    model = ASLClassifier(
        num_classes=len(class_names),
        model_name=model_name,
        dropout_rate=dropout_rate,
        pretrained=False
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")

    return model


def preprocess_image(image_path, image_size=128):
    """Load and preprocess image"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    return image, image_tensor


def predict_single(model, image_tensor, device, class_names, top_k=5):
    """Make prediction on a single image"""
    model.eval()

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

    probs = probabilities.cpu().numpy().flatten()

    # Get top-k predictions
    top_k_indices = probs.argsort()[-top_k:][::-1]
    top_k_probs = probs[top_k_indices]
    top_k_classes = [class_names[i] for i in top_k_indices]

    return top_k_classes, top_k_probs, probs


def visualize_prediction(image, top_classes, top_probs, save_path=None):
    """Visualize prediction results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Show image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(f"Predicted: {top_classes[0]} ({top_probs[0]*100:.1f}%)", fontsize=14, fontweight='bold')

    # Show top predictions
    colors = ['green' if i == 0 else 'blue' for i in range(len(top_classes))]
    ax2.barh(top_classes[::-1], top_probs[::-1], color=colors[::-1])
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Top Predictions', fontsize=14)
    ax2.set_xlim(0, 1)

    for i, (cls, prob) in enumerate(zip(top_classes[::-1], top_probs[::-1])):
        ax2.text(prob + 0.02, i, f'{prob*100:.1f}%', va='center', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def batch_predict(model, image_dir, device, class_names, output_dir=None, top_k=5):
    """Make predictions on all images in a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_paths = [
        p for p in Path(image_dir).rglob('*')
        if p.suffix.lower() in image_extensions
    ]

    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_paths)} images")

    results = []

    for img_path in image_paths:
        try:
            image, image_tensor = preprocess_image(str(img_path))
            top_classes, top_probs, _ = predict_single(model, image_tensor, device, class_names, top_k)

            result = {
                'image': str(img_path),
                'prediction': top_classes[0],
                'confidence': top_probs[0],
                'top_k': list(zip(top_classes, top_probs))
            }
            results.append(result)

            print(f"{img_path.name}: {top_classes[0]} ({top_probs[0]*100:.1f}%)")

            # Save visualization if output dir provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"{img_path.stem}_prediction.png")
                visualize_prediction(image, top_classes, top_probs, save_path)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description='ASL Classifier Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str,
                        help='Path to single image for prediction')
    parser.add_argument('--image_dir', type=str,
                        help='Path to directory of images for batch prediction')
    parser.add_argument('--class_names', type=str, default='class_names.pkl',
                        help='Path to class names pickle file')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save prediction visualizations')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to show')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load class names
    with open(args.class_names, 'rb') as f:
        class_names = pickle.load(f)
    print(f"Loaded {len(class_names)} classes")

    # Load model
    model = load_model(args.checkpoint, class_names, device)

    # Single image prediction
    if args.image:
        print(f"\nPredicting on: {args.image}")
        image, image_tensor = preprocess_image(args.image)
        top_classes, top_probs, _ = predict_single(model, image_tensor, device, class_names, args.top_k)

        print("\nTop predictions:")
        for i, (cls, prob) in enumerate(zip(top_classes, top_probs)):
            print(f"{i+1}. {cls}: {prob*100:.2f}%")

        # Visualize
        save_path = os.path.join(args.output_dir, 'prediction.png') if args.output_dir else None
        visualize_prediction(image, top_classes, top_probs, save_path)

    # Batch prediction
    elif args.image_dir:
        print(f"\nBatch prediction on: {args.image_dir}")
        results = batch_predict(model, args.image_dir, device, class_names, args.output_dir, args.top_k)

        if results:
            # Summary statistics
            predictions = [r['prediction'] for r in results]
            confidences = [r['confidence'] for r in results]

            print("\n" + "="*50)
            print(f"Processed {len(results)} images")
            print(f"Average confidence: {np.mean(confidences)*100:.2f}%")
            print(f"Min confidence: {np.min(confidences)*100:.2f}%")
            print(f"Max confidence: {np.max(confidences)*100:.2f}%")

    else:
        print("Please provide either --image or --image_dir")


if __name__ == '__main__':
    main()
