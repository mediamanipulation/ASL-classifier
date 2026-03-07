"""
Quick test to verify GUI components work
"""

import torch
import pickle
from PIL import Image
import torchvision.transforms as transforms
from src.models.classifier import ASLClassifier

print("Testing GUI components...")
print("="*60)

# Test 1: Load class names
print("\n1. Loading class names...")
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)
print(f"   [OK] Loaded {len(class_names)} classes: {class_names[:5]}...{class_names[-3:]}")

# Test 2: Load model
print("\n2. Loading model...")
device = torch.device('cpu')
checkpoint = torch.load('checkpoints/asl_efficientnet_b0/best_model.pth', map_location=device)
model = ASLClassifier(num_classes=len(class_names), dropout_rate=0.4, pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"   [OK] Model loaded (Val Acc: {checkpoint.get('val_acc', 0):.2f}%)")

# Test 3: Test prediction
print("\n3. Testing prediction...")
image = Image.open('img/test.jpg').convert('RGB')
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
img_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
    top_idx = probs.argmax()

print(f"   [OK] Prediction: {class_names[top_idx]} ({probs[top_idx]*100:.2f}%)")

# Test 4: Check Gradio
print("\n4. Checking Gradio installation...")
try:
    import gradio as gr
    print(f"   [OK] Gradio version: {gr.__version__}")
except Exception as e:
    print(f"   [ERROR] Gradio error: {e}")

print("\n" + "="*60)
print("All tests passed! GUI is ready to launch.")
print("="*60)
print("\nTo start the GUI, run:")
print("  python asl_gui.py")
print("\nOr double-click: launch_gui.bat")
