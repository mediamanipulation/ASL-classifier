# Quick Start Guide

## Installation (2 minutes)

```bash
# Navigate to project directory
cd e:\miniconda\envs\asl-env\app

# Install dependencies
pip install -r requirements.txt
```

## Training (5 minutes to start)

### Basic Training
```bash
python train.py
```

### Monitor Progress
```bash
# In another terminal
tensorboard --logdir experiments/asl_efficientnet_b0/tensorboard
# Open http://localhost:6006 in browser
```

### Custom Training
```bash
# Edit config.yaml first, then:
python train.py --config config.yaml
```

## Inference (1 minute)

### Single Image
```bash
python inference.py \
    --checkpoint checkpoints/asl_efficientnet_b0/best_model.pth \
    --image img/test.jpg \
    --class_names class_names.pkl
```

### Batch Prediction
```bash
python inference.py \
    --checkpoint checkpoints/asl_efficientnet_b0/best_model.pth \
    --image_dir input/test/ \
    --output_dir predictions/
```

## Configuration Quick Reference

Edit `config.yaml` to change:

```yaml
# Model
model:
  architecture: "efficientnet_b0"  # Try: efficientnet_b1, resnet50
  dropout_rate: 0.4                # Try: 0.3-0.5

# Training
training:
  batch_size: 64                   # Reduce if OOM error
  num_epochs: 25
  learning_rate: 0.0005            # Try: 0.001 or 0.0001

# Data
data:
  image_size: 128                  # Try: 224 for better accuracy
  num_workers: 4                   # Increase for faster loading
```

## Common Issues

### CUDA Out of Memory
```yaml
training:
  batch_size: 32  # or 16
```

### Slow Training
```yaml
data:
  num_workers: 8  # increase

mixed_precision:
  enabled: true   # if GPU supports it
```

## Where to Find Results

- **Models**: `checkpoints/asl_efficientnet_b0/best_model.pth`
- **Logs**: `experiments/asl_efficientnet_b0/training_*.log`
- **Metrics**: `experiments/asl_efficientnet_b0/test_metrics.txt`
- **Plots**: `experiments/asl_efficientnet_b0/*.png`

## Key Files

- `train.py` - Training script
- `inference.py` - Prediction script
- `config.yaml` - Configuration
- `README.md` - Full documentation
- `IMPROVEMENTS.md` - What was improved

## Help

```bash
python train.py --help
python inference.py --help
```

For detailed documentation, see [README.md](README.md)
