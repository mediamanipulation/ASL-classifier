import tensorflow as tf

# Check if TensorFlow can access the GPU
print("Available GPUs:")
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

# Check if TensorFlow is running on GPU
print("Is TensorFlow running on GPU?", tf.test.is_built_with_cuda())
