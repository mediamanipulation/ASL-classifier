# CLAUDE.md - ASL Alphabet Recognition System

## Project Overview
ASL hand sign recognition system: 36 classes (A-Z letters + 0-9 digits) using EfficientNet-B0 via PyTorch/timm.

## Environment
- **Conda env**: `e:\miniconda\envs\asl-env`
- **Project root**: `e:\miniconda\envs\asl-env\app`
- **Python**: Use `e:\miniconda\envs\asl-env\python.exe` (system Python is different)
- **GPU**: NVIDIA RTX 4090 (24GB VRAM), CUDA 12.6, PyTorch 2.4.1+cu124
- **Mixed precision**: Enabled (FP16, config.yaml)

## Key Commands
```bash
# Training
python train.py

# GUIs
python asl_gui.py              # Basic GUI - port 7860
python asl_gui_enhanced.py     # Enhanced with preprocessing - port 7861
python asl_test_ui.py          # Testing dashboard - port 7862

# Inference
python inference.py --checkpoint checkpoints/asl_efficientnet_b0/best_model.pth --image img/test.jpg --class_names class_names.pkl
```

## Project Structure
```
app/
  train.py                  # Training pipeline (config-driven)
  inference.py              # CLI inference (single/batch)
  asl_gui.py                # Gradio prediction GUI
  asl_gui_enhanced.py       # Enhanced GUI with preprocessing
  asl_test_ui.py            # Testing dashboard (batch test, confidence analysis)
  preprocess_image.py       # Image preprocessing utilities
  analyze_dataset.py        # Dataset diversity analysis
  config.yaml               # Training configuration
  class_names.pkl           # 36 class names (pickled list)
  src/
    models/classifier.py    # ASLClassifier (canonical definition)
    data/dataset.py         # ASLDataset + get_transforms()
    utils/metrics.py        # evaluate_model(), compute_accuracy()
    utils/logging.py        # Logging utilities
  input/train|valid|test/   # Dataset (ImageFolder format, ~70 images/class)
  checkpoints/              # Model checkpoints (.pth)
  experiments/              # Training logs, TensorBoard, confusion matrices
```

## Critical Conventions

### ImageNet Normalization
ALL transform pipelines MUST include ImageNet normalization after ToTensor():
```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```
This matches EfficientNet's pretrained expectations. Forgetting this breaks inference.

### ASLClassifier is duplicated
The model class exists in 3 places (must stay in sync):
- `src/models/classifier.py` (canonical - GUIs import from here)
- `train.py` (inline copy)
- `inference.py` (inline copy)

### Checkpoint format
Checkpoints save a dict with keys: `model_state_dict`, `optimizer_state_dict`, `epoch`, `val_loss`, `val_acc`, `config`.

### .gitignore notes
`models/` pattern in .gitignore catches `src/models/` — use `git add -f` for files under `src/models/` and `src/data/`.
`*.txt` is ignored — `requirements.txt` needs `git add -f`.

## Training Configuration
Config is in `config.yaml`. Key settings:
- `model.freeze_backbone`: Freeze EfficientNet layers initially (two-phase training)
- `model.unfreeze_after_epochs`: When to unfreeze backbone (uses 1/10th LR)
- `augmentation.*`: Heavy augmentation enabled (perspective, affine, blur, erasing)
- `training.early_stopping.patience`: 7 epochs
- `mixed_precision.enabled`: true (RTX 4090)

## Git
- `main` branch: baseline code
- Feature branches for changes (e.g., `feature/improve-generalization`)
- Do not commit .pth, .keras, .pkl, images, or log files (gitignored)

## Known Issues
- Old TensorFlow was removed (conflicted with NumPy 2.x) — do not reinstall it
- Dataset is small (~70 images/class) and homogeneous — real-world accuracy gap exists
- `marker-pdf` has a numpy<2 constraint warning (ignorable, not used by this project)
