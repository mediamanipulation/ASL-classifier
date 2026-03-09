# ASL Hand Sign Recognition

Real-time American Sign Language recognition — 36 classes (A-Z + 0-9) powered by EfficientNet-B0.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)
![Accuracy](https://img.shields.io/badge/accuracy-99.96%25-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## What It Does

Upload a photo of a hand sign. Get back the letter or number with a confidence score. Under one second.

```
Input: Photo of hand signing "L"
Output: L (100.0% confidence)
```

Three Gradio interfaces included — a basic predictor, an enhanced version with image preprocessing, and a full testing dashboard for model evaluation.

## The Numbers

| Metric | Score |
|--------|-------|
| Test accuracy | **99.96%** |
| Real-world letter accuracy | **100%** |
| Real-world letter confidence | **96-100%** |
| Classes | **36** (A-Z + 0-9) |
| Training time | **~10 min** (RTX 4090) |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
python train.py

# Predict
python asl_gui.py
# → open http://127.0.0.1:7860
```

That's it. Upload an image or point your webcam at a hand sign.

## Three Interfaces

| Interface | Port | Use Case |
|-----------|------|----------|
| `python asl_gui.py` | 7860 | Quick predictions — upload and go |
| `python asl_gui_enhanced.py` | 7861 | Tough images — adds preprocessing and background handling |
| `python asl_test_ui.py` | 7862 | Model evaluation — batch testing, confidence analysis, training curves |

## How It Works

```
Photo (224x224 RGB)
  → EfficientNet-B0 (ImageNet pretrained)
  → Dropout(0.2)
  → FC(1280 → 36)
  → Softmax → Prediction + Confidence
```

Everything is config-driven. Edit `config.yaml` to change architecture, augmentation, learning rate, batch size — no code changes needed.

## The Generalization Story

The original model hit 99.92% on the test set. Sounded great. Then we tested on real-world images:

**11.3% accuracy. 29.4% average confidence.**

The model had memorized the training set's specific backgrounds and lighting instead of learning hand shapes. The fix wasn't a bigger model or fancier augmentation — it was **30 diverse images per class** from a different source.

| | Before | After |
|---|--------|-------|
| Real-world accuracy | 11.3% | **100%** |
| Average confidence | 29.4% | **99.9%** |
| L sign confidence | 17.2% | **100%** |

**Lesson:** Data diversity beats model complexity. Every time.

## Training

All configuration lives in `config.yaml`:

```yaml
model:
  architecture: efficientnet_b0
  pretrained: true
  dropout_rate: 0.2

training:
  batch_size: 64
  learning_rate: 0.0005
  num_epochs: 40
  early_stopping:
    patience: 7

augmentation:
  rotation_degrees: 10
  random_perspective: true
  random_erasing: true
  color_jitter: {brightness: 0.3, contrast: 0.3}

mixed_precision:
  enabled: true  # 2-3x speedup on modern GPUs
```

Training produces:
- Best model checkpoint (`checkpoints/`)
- Training curves and confusion matrix (`experiments/`)
- TensorBoard logs (`tensorboard --logdir experiments/*/tensorboard`)
- Per-class metrics and classification report

## Dataset

```
input/
  train/   # 36 folders (a-z, 0-9), 70-100 images each
  valid/   # Validation split
  test/    # Held-out test set
```

**Letters A-Z:** 100 images/class — original 70 + 30 diverse images from the [ASL Alphabet Test](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test) dataset (CC0, varied backgrounds and skin tones).

**Digits 0-9:** 70 images/class from the original training set.

## CLI Inference

```bash
# Single image
python inference.py \
  --checkpoint checkpoints/asl_efficientnet_b0/best_model.pth \
  --image photo.jpg \
  --class_names class_names.pkl

# Batch — entire folder
python inference.py \
  --checkpoint checkpoints/asl_efficientnet_b0/best_model.pth \
  --image path/to/folder/ \
  --class_names class_names.pkl
```

## Project Structure

```
app/
├── train.py                    # Training pipeline
├── inference.py                # CLI inference
├── asl_gui.py                  # Prediction UI
├── asl_gui_enhanced.py         # Enhanced UI with preprocessing
├── asl_test_ui.py              # Testing dashboard
├── config.yaml                 # All training configuration
├── class_names.pkl             # Class label mapping
├── src/
│   ├── models/classifier.py    # EfficientNet-B0 wrapper
│   ├── data/dataset.py         # Dataset + augmentation transforms
│   └── utils/                  # Metrics and logging
├── input/                      # Training data
├── checkpoints/                # Saved models
└── experiments/                # Logs, curves, confusion matrices
```

## Requirements

- Python 3.11+
- PyTorch 2.4+ with CUDA
- NVIDIA GPU recommended (trained on RTX 4090, works on any CUDA GPU)
- ~2GB disk for dataset, ~50MB for model checkpoint
- article code [github ASL-classifier](https://github.com/mediamanipulation/ASL-classifier)
## Acknowledgments

- [TIMM](https://github.com/huggingface/pytorch-image-models) — PyTorch Image Models
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019
- [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — Original training dataset
- [ASL Alphabet Test](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test) — Diverse validation images (CC0)
- Big shout out to Rob Mulla for this code and instruction, I used to adapt for the ASL classifier 
-  Build Your First Pytorch Model In Minutes! [Tutorial + Code](https://www.youtube.com/watch?v=tHL5STNJKag&t=269s)
- this is the link to his youtube channel (556) Rob Mulla -[youtube channel](https://www.youtube.com/@robmulla) 

## License

MIT
