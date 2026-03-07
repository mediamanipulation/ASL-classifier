"""
ASL Recognition GUI
Upload an image and get instant predictions for ASL letters (A-Z) and numbers (0-9)
"""

import gradio as gr
import torch
import pickle
from PIL import Image
import torchvision.transforms as transforms
from src.models.classifier import ASLClassifier
import numpy as np

# Global variables for model
model = None
class_names = None
device = None

def load_model():
    """Load the trained model"""
    global model, class_names, device

    print("Loading model...")

    # Load class names
    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    checkpoint = torch.load(
        'checkpoints/asl_efficientnet_b0/best_model.pth',
        map_location=device
    )

    model = ASLClassifier(
        num_classes=len(class_names),
        model_name='efficientnet_b0',
        dropout_rate=0.4,
        pretrained=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    val_acc = checkpoint.get('val_acc', 'N/A')
    print(f"Model loaded! Validation Accuracy: {val_acc:.2f}%")
    print(f"Using device: {device}")
    print(f"Classes: {len(class_names)} ({class_names[0]}-{class_names[-1]})")

    return model, class_names, device


def predict_image(image):
    """
    Predict ASL sign from image

    Args:
        image: PIL Image or numpy array

    Returns:
        tuple: (predicted_class, confidence_dict)
    """
    if image is None:
        return "Please upload an image!", {}

    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Transform and add batch dimension
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probs = probabilities.cpu().numpy()[0]

    # Get top 5 predictions
    top_5_idx = probs.argsort()[-5:][::-1]

    # Create results dictionary for Gradio
    results = {}
    for idx in top_5_idx:
        class_label = class_names[idx]
        confidence = float(probs[idx])
        results[class_label] = confidence

    # Get top prediction
    top_class = class_names[top_5_idx[0]]
    top_confidence = probs[top_5_idx[0]]

    # Create prediction text
    prediction_text = f"**{top_class.upper()}**"
    if top_class.isdigit():
        prediction_text += f" (Number {top_class})"
    else:
        prediction_text += f" (Letter {top_class.upper()})"

    prediction_text += f"\n\nConfidence: **{top_confidence*100:.2f}%**"

    return prediction_text, results


def create_gui():
    """Create Gradio interface"""

    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .output-class {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin: 10px 0;
    }
    """

    with gr.Blocks(css=css, title="ASL Recognition") as demo:
        gr.Markdown(
            """
            # 🤟 ASL Sign Language Recognition

            Upload an image of an ASL hand sign and the AI will identify if it's a **letter (A-Z)** or **number (0-9)**

            ### How to use:
            1. Upload an image or drag & drop
            2. Click "Predict" or wait for auto-prediction
            3. See the results instantly!
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input
                image_input = gr.Image(
                    label="Upload ASL Hand Sign Image",
                    type="pil",
                    height=400
                )

                predict_btn = gr.Button(
                    "🔍 Predict Sign",
                    variant="primary",
                    size="lg"
                )

                gr.Markdown(
                    """
                    ### 💡 Tips:
                    - Use clear, well-lit images
                    - Hand should be clearly visible
                    - Avoid cluttered backgrounds
                    - Works with digits 0-9 and letters A-Z
                    """
                )

            with gr.Column(scale=1):
                # Outputs
                prediction_output = gr.Markdown(
                    label="Prediction",
                    value="Upload an image to get started!"
                )

                confidence_output = gr.Label(
                    label="Top 5 Predictions",
                    num_top_classes=5
                )

        # Examples
        gr.Markdown("### 📸 Try these example images:")
        gr.Examples(
            examples=[
                ["img/test.jpg"],
                ["img/test2.jpg"],
            ],
            inputs=image_input,
            outputs=[prediction_output, confidence_output],
            fn=predict_image,
            cache_examples=False
        )

        # Model info
        with gr.Accordion("ℹ️ Model Information", open=False):
            gr.Markdown(
                f"""
                - **Model**: EfficientNet-B0
                - **Classes**: 36 (0-9 digits + A-Z letters)
                - **Image Size**: 128x128
                - **Device**: {device}
                - **Status**: ✅ Ready
                """
            )

        # Connect button
        predict_btn.click(
            fn=predict_image,
            inputs=image_input,
            outputs=[prediction_output, confidence_output]
        )

        # Auto-predict on image upload
        image_input.change(
            fn=predict_image,
            inputs=image_input,
            outputs=[prediction_output, confidence_output]
        )

    return demo


if __name__ == "__main__":
    # Load model first
    load_model()

    # Create and launch GUI
    print("\n" + "="*60)
    print("Starting ASL Recognition GUI...")
    print("="*60 + "\n")

    demo = create_gui()

    # Launch with share=False for local only, share=True to get public link
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
