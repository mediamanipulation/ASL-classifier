import torch
import pickle
from PIL import Image
import torchvision.transforms as transforms
from src.models.classifier import ASLClassifier

# Load class names
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('checkpoints/asl_efficientnet_b0/best_model.pth', map_location=device)

model = ASLClassifier(num_classes=36, model_name='efficientnet_b0', dropout_rate=0.2, pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Transform (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Predict
image = Image.open('your_image.jpg').convert('RGB')
with torch.no_grad():
    probs = torch.softmax(model(transform(image).unsqueeze(0).to(device)), dim=1)[0]

predicted = class_names[probs.argmax()]
confidence = probs.max().item()
print(f"{predicted}: {confidence*100:.1f}%")
