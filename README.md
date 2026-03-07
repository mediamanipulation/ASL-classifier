# ASL Alphabet Recognition System

A deep learning system for recognizing American Sign Language (ASL) alphabet hand signs using PyTorch and EfficientNet.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎨 Web GUI Interface

![ASL Recognition GUI](uifrnt.jpg)

**Upload any image and get instant ASL letter/number recognition!**

---

## Features

- **🎨 Beautiful Web GUI**: Easy-to-use interface for instant predictions (99.92% accuracy!)
- **State-of-the-art Architecture**: Uses EfficientNet-B0 with proper dropout regularization
- **Comprehensive Training Pipeline**: Includes learning rate scheduling, early stopping, and model checkpointing
- **Configuration Management**: YAML-based configuration for easy experimentation
- **Extensive Logging**: TensorBoard integration and detailed training logs
- **Modular Design**: Clean separation of concerns with organized code structure
- **Evaluation Metrics**: Per-class metrics, confusion matrices, and classification reports
- **Inference Support**: Single image and batch prediction capabilities

## Project Structure

```
asl-env/app/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py          # Dataset and transforms
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py       # Model architecture
│   └── utils/
│       ├── __init__.py
│       ├── logging.py          # Logging utilities
│       └── metrics.py          # Evaluation metrics
│
├── input/                      # Dataset directory
│   ├── train/
│   ├── valid/
│   └── test/
│
├── experiments/                # Training logs and results
├── checkpoints/               # Saved model checkpoints
├── old_scripts/               # Legacy training scripts
│
├── train.py                   # Main training script
├── inference.py               # Inference script
├── asl_gui.py                 # Web GUI application 🎨
├── launch_gui.bat             # Quick GUI launcher
├── config.yaml                # Training configuration
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
├── GUI_GUIDE.md              # GUI user guide
└── uifrnt.jpg                # GUI screenshot
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Conda or virtualenv

### Setup

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd e:\miniconda\envs\asl-env\app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
   ```

## 🚀 Quick Start - GUI

The easiest way to use the ASL recognition system is through the web GUI:

### Launch the GUI

**Method 1: Quick Launch (Windows)**
```bash
launch_gui.bat
```
Double-click the `launch_gui.bat` file in the app folder.

**Method 2: Command Line**
```bash
python asl_gui.py
```

### Using the GUI

1. **Launch** - Your browser will automatically open to `http://127.0.0.1:7860`
2. **Upload** - Drag & drop an ASL hand sign image or click to upload
3. **Predict** - Get instant results showing:
   - **Letter or Number** (e.g., "A" or "7")
   - **Confidence Score** (e.g., 98.5%)
   - **Top 5 Predictions** with probability bars

### GUI Features

- ✅ **Instant Recognition**: < 1 second processing time
- ✅ **36 Classes**: Recognizes digits 0-9 and letters A-Z
- ✅ **99.92% Accurate**: Trained model with high accuracy
- ✅ **Top 5 Results**: See alternative predictions with confidence scores

For detailed GUI instructions, see [GUI_GUIDE.md](GUI_GUIDE.md)

## Dataset

### Structure

The dataset should be organized in the following structure:

```
input/
├── train/
│   ├── A/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── B/
│   └── ...
├── valid/
│   └── [same structure as train]
└── test/
    └── [same structure as train]
```

Each subfolder represents a class (A-Z for ASL alphabet).

### Data Augmentation

The training pipeline includes:
- Random horizontal flips
- Random rotation (±10 degrees)
- Color jitter (brightness, contrast, saturation, hue)

## Usage

### Training

#### Basic Training

Train with default configuration:

```bash
python train.py
```

#### Custom Configuration

1. Copy and modify `config.yaml`:
   ```bash
   cp config.yaml my_config.yaml
   # Edit my_config.yaml with your settings
   ```

2. Train with custom config:
   ```bash
   python train.py --config my_config.yaml
   ```

#### Configuration Options

Edit `config.yaml` to customize:

- **Model**: Architecture, dropout rate, pretrained weights
- **Data**: Paths, image size, batch size, augmentation
- **Training**: Learning rate, epochs, optimizer, scheduler
- **Logging**: TensorBoard, checkpoint frequency

Example configuration snippet:

```yaml
model:
  architecture: "efficientnet_b0"
  dropout_rate: 0.4

training:
  batch_size: 64
  num_epochs: 25
  learning_rate: 0.0005
```

### Inference

#### Single Image Prediction

```bash
python inference.py \
    --checkpoint checkpoints/asl_efficientnet_b0/best_model.pth \
    --image img/test.jpg \
    --class_names class_names.pkl \
    --output_dir predictions/
```

#### Batch Prediction

```bash
python inference.py \
    --checkpoint checkpoints/asl_efficientnet_b0/best_model.pth \
    --image_dir input/test/A/ \
    --class_names class_names.pkl \
    --output_dir predictions/ \
    --top_k 5
```

### Monitoring Training

#### TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir experiments/asl_efficientnet_b0/tensorboard
```

Then open http://localhost:6006 in your browser.

#### Training Logs

Logs are saved to `experiments/<experiment_name>/training_YYYYMMDD_HHMMSS.log`

## Model Architecture

### ASLClassifier

- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Input**: RGB images, 128×128 pixels
- **Features**: 1280-dimensional feature vector
- **Dropout**: 0.4 (applied before final classifier)
- **Output**: Softmax probabilities over 26 classes

```
Input (3, 128, 128)
    ↓
EfficientNet-B0 (pretrained)
    ↓
Features (1280)
    ↓
Dropout (0.4)
    ↓
Linear (1280 → 26)
    ↓
Output (26 classes)
```

## Training Details

### Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Weight Decay | 0.0001 |
| Batch Size | 64 |
| Epochs | 25 |
| Scheduler | ReduceLROnPlateau |
| Early Stopping Patience | 4 epochs |

### Best Practices

1. **GPU Utilization**: Use `num_workers=4` for optimal data loading
2. **Mixed Precision**: Enable in config for 2-3x speedup on modern GPUs
3. **Learning Rate**: Start with 0.0005, reduce if loss plateaus
4. **Dropout**: 0.3-0.5 range works well for most datasets
5. **Batch Size**: Use largest that fits in GPU memory

## Results

### Evaluation Metrics

After training, the following are saved to `experiments/<experiment_name>/`:

- **Training curves** (`training_curves.png`)
- **Confusion matrix** (`confusion_matrix.png`)
- **Per-class metrics** (`per_class_metrics.txt`)
- **Classification report** (`test_metrics.txt`)

### Sample Output

```
Test Accuracy: 98.50%

Per-Class Metrics:
Class                Precision     Recall   F1-Score    Support
--------------------------------------------------------------------
A                      0.9850     0.9900     0.9875        100
B                      0.9800     0.9850     0.9825        100
...
```

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` in config.yaml
- Reduce `image_size`
- Use `num_workers: 0` if system memory is limited

### Poor Training Performance

- Check data augmentation isn't too aggressive
- Verify learning rate (try 0.001 or 0.0001)
- Increase training epochs
- Check for class imbalance

### Slow Data Loading

- Increase `num_workers` (try 4, 8, or 12)
- Enable `pin_memory: true` for GPU training
- Use SSD for dataset storage

## Advanced Usage

### Fine-tuning

To fine-tune on a new dataset:

1. Load pretrained weights:
   ```python
   checkpoint = torch.load('checkpoints/best_model.pth')
   model.load_state_dict(checkpoint['model_state_dict'])
   ```

2. Freeze early layers (optional):
   ```python
   for param in model.base_model.parameters():
       param.requires_grad = False
   ```

3. Train with lower learning rate

### Experimenting with Models

Try different architectures in `config.yaml`:

```yaml
model:
  architecture: "efficientnet_b1"  # or b2, b3, resnet50, vit_base_patch16_224
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- **TIMM**: PyTorch Image Models library
- **EfficientNet**: Tan & Le, 2019
- **ASL Dataset**: [Add dataset source/citation]

## Citation

If you use this code, please cite:

```bibtex
@software{asl_classifier_2024,
  title = {ASL Alphabet Recognition System},
  year = {2024},
  author = {[Your Name]},
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

---

**Last Updated**: February 2026
