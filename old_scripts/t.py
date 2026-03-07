import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"OpenCV version: {cv2.__version__}")

# Check if GPU is available
print(f"Is GPU available: {tf.config.list_physical_devices('GPU')}")
