"""
ASL Model Testing UI
Evaluate model performance: single image, batch testing, confidence analysis, training results.
"""

import gradio as gr
import torch
import pickle
import os
import glob
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from src.models.classifier import ASLClassifier
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# Global variables
model = None
class_names = None
device = None
transform = None

CHECKPOINT_PATH = 'checkpoints/asl_efficientnet_b0/best_model.pth'
CLASS_NAMES_PATH = 'class_names.pkl'
EXPERIMENTS_DIR = 'experiments/asl_efficientnet_b0'


def load_model():
    """Load the trained model"""
    global model, class_names, device, transform

    with open(CLASS_NAMES_PATH, 'rb') as f:
        class_names = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model = ASLClassifier(
        num_classes=len(class_names),
        model_name='efficientnet_b0',
        dropout_rate=0.4,
        pretrained=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_acc = checkpoint.get('val_acc', 'N/A')
    epoch = checkpoint.get('epoch', 'N/A')
    print(f"Model loaded from epoch {epoch}, Val Acc: {val_acc:.2f}%")
    print(f"Device: {device}, Classes: {len(class_names)}")

    return val_acc, epoch


def predict(image):
    """Run prediction on a single PIL image"""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probs = probabilities.cpu().numpy()[0]

    top_5_idx = probs.argsort()[-5:][::-1]
    top_class = class_names[top_5_idx[0]]
    top_conf = float(probs[top_5_idx[0]])
    top_5 = {class_names[i]: float(probs[i]) for i in top_5_idx}

    return top_class, top_conf, top_5


# --- Tab 1: Single Image Test ---

def single_image_test(image):
    """Test a single image"""
    if image is None:
        return "Upload an image", {}, ""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    top_class, top_conf, top_5 = predict(image)

    # Status indicator
    if top_conf >= 0.9:
        status = "HIGH confidence"
    elif top_conf >= 0.7:
        status = "MEDIUM confidence"
    else:
        status = "LOW confidence - model is uncertain"

    result_text = f"## Prediction: **{top_class.upper()}**\n"
    result_text += f"Confidence: **{top_conf*100:.1f}%** ({status})"

    return result_text, top_5


# --- Tab 2: Batch Folder Test ---

def batch_folder_test(folder_path):
    """Test all images in a folder organized by class subfolders"""
    if not folder_path or not os.path.isdir(folder_path):
        return "Please enter a valid folder path", None, ""

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    results = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    wrong_predictions = []

    # Check if folder has class subfolders
    subdirs = [d for d in Path(folder_path).iterdir() if d.is_dir()]

    if subdirs:
        # Organized by class subfolders
        for class_dir in sorted(subdirs):
            true_label = class_dir.name.lower()
            images = [f for f in class_dir.iterdir() if f.suffix.lower() in image_extensions]

            for img_path in images:
                try:
                    image = Image.open(img_path).convert('RGB')
                    pred_class, conf, _ = predict(image)

                    class_total[true_label] += 1
                    if pred_class.lower() == true_label:
                        class_correct[true_label] += 1
                    else:
                        wrong_predictions.append(
                            f"  {img_path.name}: expected **{true_label}**, got **{pred_class}** ({conf*100:.1f}%)"
                        )

                    results.append({'correct': pred_class.lower() == true_label, 'conf': conf})
                except Exception as e:
                    wrong_predictions.append(f"  {img_path.name}: ERROR - {e}")
    else:
        # Flat folder - no ground truth, just show predictions
        images = [f for f in Path(folder_path).iterdir() if f.suffix.lower() in image_extensions]
        if not images:
            return "No images found in folder", None, ""

        for img_path in sorted(images):
            try:
                image = Image.open(img_path).convert('RGB')
                pred_class, conf, _ = predict(image)
                results.append({'correct': None, 'conf': conf})
                wrong_predictions.append(f"  {img_path.name}: **{pred_class}** ({conf*100:.1f}%)")
            except Exception as e:
                wrong_predictions.append(f"  {img_path.name}: ERROR - {e}")

        summary = f"## Predictions for {len(results)} images\n\n"
        summary += f"Average confidence: **{np.mean([r['conf'] for r in results])*100:.1f}%**\n\n"
        summary += "### Results:\n" + "\n".join(wrong_predictions)
        return summary, None, ""

    if not results:
        return "No images found", None, ""

    # Calculate metrics
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total * 100
    avg_conf = np.mean([r['conf'] for r in results]) * 100

    # Build per-class table
    table_lines = ["| Class | Correct | Total | Accuracy |", "|-------|---------|-------|----------|"]
    for cls in sorted(class_total.keys()):
        c = class_correct[cls]
        t = class_total[cls]
        acc = c / t * 100 if t > 0 else 0
        table_lines.append(f"| {cls.upper()} | {c} | {t} | {acc:.0f}% |")

    summary = f"## Batch Test Results\n\n"
    summary += f"**Overall Accuracy: {accuracy:.1f}%** ({correct}/{total})\n\n"
    summary += f"**Average Confidence: {avg_conf:.1f}%**\n\n"
    summary += "### Per-Class Breakdown\n\n"
    summary += "\n".join(table_lines)

    if wrong_predictions:
        summary += f"\n\n### Incorrect Predictions ({len(wrong_predictions)})\n\n"
        summary += "\n".join(wrong_predictions[:50])  # Limit display
        if len(wrong_predictions) > 50:
            summary += f"\n\n... and {len(wrong_predictions) - 50} more"

    # Confidence histogram
    confs = [r['conf'] * 100 for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(confs, bins=20, range=(0, 100), color='steelblue', edgecolor='white')
    ax.set_xlabel('Confidence (%)')
    ax.set_ylabel('Count')
    ax.set_title(f'Confidence Distribution (n={total})')
    ax.axvline(x=avg_conf, color='red', linestyle='--', label=f'Mean: {avg_conf:.1f}%')
    ax.legend()
    plt.tight_layout()

    return summary, fig, ""


# --- Tab 3: Multi-Image Confidence ---

def confidence_analysis(images):
    """Analyze confidence across multiple uploaded images"""
    if not images:
        return "Upload some images", None

    predictions = []

    for img_data in images:
        try:
            if isinstance(img_data, np.ndarray):
                image = Image.fromarray(img_data)
            elif isinstance(img_data, str):
                image = Image.open(img_data).convert('RGB')
            else:
                image = img_data

            if isinstance(image, Image.Image):
                pred_class, conf, _ = predict(image)
                predictions.append({'class': pred_class, 'conf': conf})
        except Exception:
            continue

    if not predictions:
        return "Could not process any images", None

    # Sort by confidence
    predictions.sort(key=lambda x: x['conf'])

    # Build results
    lines = ["| # | Prediction | Confidence | Status |",
             "|---|------------|------------|--------|"]
    for i, p in enumerate(predictions):
        status = "HIGH" if p['conf'] >= 0.9 else ("MED" if p['conf'] >= 0.7 else "LOW")
        lines.append(f"| {i+1} | {p['class'].upper()} | {p['conf']*100:.1f}% | {status} |")

    avg_conf = np.mean([p['conf'] for p in predictions]) * 100
    low_conf = sum(1 for p in predictions if p['conf'] < 0.7)

    summary = f"## Confidence Analysis ({len(predictions)} images)\n\n"
    summary += f"**Average: {avg_conf:.1f}%** | Low confidence: **{low_conf}/{len(predictions)}**\n\n"
    summary += "\n".join(lines)

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(4, len(predictions) * 0.3)))
    classes = [f"{p['class'].upper()}" for p in predictions]
    confs = [p['conf'] * 100 for p in predictions]
    colors = ['#e74c3c' if c < 70 else '#f39c12' if c < 90 else '#27ae60' for c in confs]

    ax.barh(range(len(classes)), confs, color=colors)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel('Confidence (%)')
    ax.set_xlim(0, 105)
    ax.axvline(x=70, color='orange', linestyle=':', alpha=0.5, label='70% threshold')
    ax.axvline(x=90, color='green', linestyle=':', alpha=0.5, label='90% threshold')
    ax.legend()
    ax.set_title('Confidence by Image (sorted low to high)')
    for i, v in enumerate(confs):
        ax.text(v + 1, i, f'{v:.0f}%', va='center', fontsize=9)
    plt.tight_layout()

    return summary, fig


# --- Tab 4: Training Results ---

def load_training_results():
    """Load and display training results"""
    results = "## Training Results\n\n"

    # Check for test metrics
    metrics_path = os.path.join(EXPERIMENTS_DIR, 'test_metrics.txt')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = f.read()
        results += f"### Test Metrics\n```\n{metrics}\n```\n\n"
    else:
        results += "No test_metrics.txt found yet (training may not have completed)\n\n"

    # Check for training log
    log_files = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, 'training_*.log')))
    log_summary = ""
    if log_files:
        latest_log = log_files[-1]
        with open(latest_log, 'r') as f:
            lines = f.readlines()

        # Extract epoch summaries
        epoch_lines = [l.strip() for l in lines if 'Train Loss:' in l or 'Val Loss:' in l
                       or 'Backbone unfrozen' in l or 'Backbone frozen' in l
                       or 'Early stopping' in l or 'Test Accuracy' in l
                       or 'Saved best model' in l]
        if epoch_lines:
            log_summary = "\n".join(epoch_lines[-30:])  # Last 30 relevant lines
            results += f"### Training Log (latest)\n```\n{log_summary}\n```\n\n"

    # Check for training curves image
    curves_path = os.path.join(EXPERIMENTS_DIR, 'training_curves.png')
    curves_img = None
    if os.path.exists(curves_path):
        curves_img = Image.open(curves_path)

    # Check for confusion matrix
    cm_path = os.path.join(EXPERIMENTS_DIR, 'confusion_matrix.png')
    cm_img = None
    if os.path.exists(cm_path):
        cm_img = Image.open(cm_path)

    # Checkpoint info
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        epoch = checkpoint.get('epoch', '?')
        val_acc = checkpoint.get('val_acc', 0)
        val_loss = checkpoint.get('val_loss', 0)
        results += f"### Best Checkpoint\n"
        results += f"- Epoch: **{epoch}**\n"
        results += f"- Val Accuracy: **{val_acc:.2f}%**\n"
        results += f"- Val Loss: **{val_loss:.4f}**\n"

    return results, curves_img, cm_img


def create_ui():
    """Create the testing UI"""

    with gr.Blocks(title="ASL Model Testing") as demo:
        gr.Markdown("# ASL Model Testing Dashboard\nEvaluate model performance after training")

        with gr.Tabs():

            # Tab 1: Single Image
            with gr.TabItem("Single Image Test"):
                with gr.Row():
                    with gr.Column():
                        single_image = gr.Image(label="Upload Image", type="pil", height=350)
                        single_btn = gr.Button("Test", variant="primary")
                    with gr.Column():
                        single_result = gr.Markdown(value="Upload an image to test")
                        single_conf = gr.Label(label="Top 5 Predictions", num_top_classes=5)

                single_btn.click(
                    fn=single_image_test,
                    inputs=single_image,
                    outputs=[single_result, single_conf]
                )
                single_image.change(
                    fn=single_image_test,
                    inputs=single_image,
                    outputs=[single_result, single_conf]
                )

            # Tab 2: Batch Test
            with gr.TabItem("Batch Folder Test"):
                gr.Markdown(
                    "Test all images in a folder. If folder has class subfolders "
                    "(e.g., `A/`, `B/`), accuracy is calculated per class."
                )
                with gr.Row():
                    batch_path = gr.Textbox(
                        label="Folder Path",
                        placeholder="e.g., input/test or path/to/real_world_images",
                        scale=3
                    )
                    batch_btn = gr.Button("Run Batch Test", variant="primary", scale=1)

                batch_result = gr.Markdown()
                batch_plot = gr.Plot(label="Confidence Distribution")

                batch_btn.click(
                    fn=batch_folder_test,
                    inputs=batch_path,
                    outputs=[batch_result, batch_plot, gr.Textbox(visible=False)]
                )

            # Tab 3: Confidence Analysis
            with gr.TabItem("Confidence Analysis"):
                gr.Markdown("Upload multiple images to compare confidence scores across them.")
                multi_images = gr.Gallery(label="Upload Images", type="filepath", columns=6, height=200)
                multi_btn = gr.Button("Analyze Confidence", variant="primary")
                multi_result = gr.Markdown()
                multi_plot = gr.Plot(label="Confidence Chart")

                multi_btn.click(
                    fn=lambda imgs: confidence_analysis(
                        [img[0] if isinstance(img, tuple) else img for img in imgs] if imgs else []
                    ),
                    inputs=multi_images,
                    outputs=[multi_result, multi_plot]
                )

            # Tab 4: Training Results
            with gr.TabItem("Training Results"):
                refresh_btn = gr.Button("Load / Refresh Results", variant="primary")
                train_result = gr.Markdown()
                with gr.Row():
                    curves_img = gr.Image(label="Training Curves", height=400)
                    cm_img = gr.Image(label="Confusion Matrix", height=400)

                refresh_btn.click(
                    fn=load_training_results,
                    outputs=[train_result, curves_img, cm_img]
                )

        # Model info footer
        gr.Markdown(
            f"---\n**Model**: EfficientNet-B0 | **Checkpoint**: `{CHECKPOINT_PATH}` | **Device**: {device}"
        )

    return demo


if __name__ == "__main__":
    val_acc, epoch = load_model()

    print("\n" + "="*60)
    print("Starting ASL Model Testing UI...")
    print("="*60 + "\n")

    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=False,
        show_error=True
    )
