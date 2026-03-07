# 🤟 ASL Recognition GUI - User Guide

## Quick Start

### Launch the GUI

**Method 1: Double-click**
```
📁 launch_gui.bat
```

**Method 2: Command line**
```bash
cd "e:\miniconda\envs\asl-env\app"
python asl_gui.py
```

### What Happens Next

1. A terminal window opens showing:
   ```
   Loading model...
   Model loaded! Validation Accuracy: 99.92%

   Running on local URL:  http://127.0.0.1:7860
   ```

2. Your web browser automatically opens the GUI

3. If browser doesn't open, manually go to: **http://127.0.0.1:7860**

---

## Using the GUI

### Main Interface

```
┌─────────────────────────────────────────────────┐
│  🤟 ASL Sign Language Recognition              │
├─────────────────┬───────────────────────────────┤
│                 │                               │
│  [Upload Area]  │  **A** (Letter A)            │
│                 │  Confidence: 98.5%            │
│  Drag & Drop    │                               │
│  or Click       │  Top 5 Predictions:           │
│                 │  ████████████ A (98.5%)       │
│  🔍 Predict     │  ██ B (1.2%)                  │
│                 │  █ C (0.3%)                   │
│                 │                               │
└─────────────────┴───────────────────────────────┘
```

### Step-by-Step

1. **Upload an Image**
   - Click "Upload ASL Hand Sign Image"
   - OR drag and drop an image
   - Supported formats: JPG, PNG, JPEG

2. **Automatic Prediction**
   - Prediction happens instantly!
   - No need to click "Predict" (but you can)

3. **View Results**
   - **Main Prediction**: Shows letter/number in large text
     - Example: "**A** (Letter A)" or "**7** (Number 7)"
   - **Confidence**: How sure the AI is (0-100%)
   - **Top 5 Bar Chart**: Shows alternative predictions

---

## Features

### ✨ What You Can Do

- **Recognize 36 Classes**:
  - Letters: A-Z (26 letters)
  - Numbers: 0-9 (10 digits)

- **Instant Results**:
  - Auto-prediction on upload
  - Less than 1 second processing

- **Confidence Scores**:
  - See how confident the AI is
  - View top 5 alternative predictions

- **Easy to Use**:
  - Drag & drop images
  - Click examples to test
  - No technical knowledge needed

---

## Tips for Best Results

### ✅ Good Images

- Clear, well-lit photos
- Hand clearly visible
- Plain or simple background
- Fingers in correct position
- Close-up of hand

### ❌ Avoid

- Blurry or dark images
- Hand partially hidden
- Cluttered background
- Multiple hands in frame
- Very small hand in image

---

## Example Workflow

### Testing Letter "A"

1. Launch GUI → `python asl_gui.py`
2. Click "Upload Image"
3. Select image of ASL letter A
4. See result: **"A (Letter A) - 99.8%"**

### Testing Number "7"

1. Upload image of ASL number 7
2. Instantly see: **"7 (Number 7) - 95.4%"**
3. Check top 5 to see alternatives

---

## Troubleshooting

### GUI Won't Start

**Problem**: Error when running `python asl_gui.py`

**Solution**:
```bash
# Test components first
python test_gui.py

# If test fails, check:
1. Model file exists: checkpoints/asl_efficientnet_b0/best_model.pth
2. Class names exist: class_names.pkl
3. Gradio installed: pip install gradio
```

### Browser Doesn't Open

**Problem**: Terminal shows "Running" but no browser

**Solution**:
- Manually open: http://127.0.0.1:7860
- Try different browser
- Check if port 7860 is available

### Low Confidence Predictions

**Problem**: All predictions below 50%

**Solution**:
- Use clearer image
- Ensure correct hand position
- Try different lighting
- Check if hand sign is in training data

### Wrong Predictions

**Problem**: Predicts wrong letter/number

**Solution**:
- Model accuracy is 99.8% but not perfect
- Check if hand position matches ASL standards
- Try multiple images of same sign
- Ensure fingers are clearly visible

---

## Model Information

### Performance
- **Validation Accuracy**: 99.92%
- **Training Accuracy**: 98.13%
- **Classes**: 36 (0-9, a-z)
- **Model**: EfficientNet-B0
- **Image Size**: 128x128

### Files Used
- **Model**: `checkpoints/asl_efficientnet_b0/best_model.pth`
- **Classes**: `class_names.pkl`
- **Config**: `config.yaml`

---

## Keyboard Shortcuts (in browser)

- **Ctrl+V**: Paste image from clipboard
- **F5**: Refresh page
- **Ctrl+W**: Close tab

---

## Advanced Options

### Share with Others (Public URL)

Edit `asl_gui.py` line 232:
```python
demo.launch(
    share=True,  # Change to True
    server_port=7860
)
```

This creates a public URL you can share (expires in 72 hours)

### Custom Port

Change port 7860 to another:
```python
demo.launch(server_port=8080)  # Use port 8080 instead
```

### Save All Predictions

Add logging to track all predictions (for analysis)

---

## Stopping the GUI

### Method 1: Terminal
- Press `Ctrl+C` in the terminal window
- Wait for "Keyboard interrupt received" message

### Method 2: Close Window
- Close the terminal window
- GUI stops automatically

---

## File Locations

```
e:\miniconda\envs\asl-env\app\
├── asl_gui.py              ← Main GUI file
├── launch_gui.bat          ← Quick launcher
├── test_gui.py             ← Test if working
├── class_names.pkl         ← Class labels
├── checkpoints/            ← Saved models
│   └── asl_efficientnet_b0/
│       └── best_model.pth
└── img/                    ← Example images
    ├── test.jpg
    └── test2.jpg
```

---

## Need Help?

### Test Everything Works
```bash
python test_gui.py
```

### Check Model
```bash
python -c "import torch; print(torch.load('checkpoints/asl_efficientnet_b0/best_model.pth').keys())"
```

### Reinstall Dependencies
```bash
pip install --upgrade gradio torch torchvision timm
```

---

## Have Fun! 🎉

The GUI is ready to recognize ASL signs!

Try the example images first, then upload your own!

**Model Accuracy: 99.92%** - Enjoy! 🤟
