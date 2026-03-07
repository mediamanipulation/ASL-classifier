import tensorflow as tf

# 1. tf.constant: Creating a constant tensor
def test_tf_constant():
    print("1. tf.constant:")
    constant_tensor = tf.constant([1, 2, 3, 4])
    print(constant_tensor)
    print("Shape:", constant_tensor.shape)
    print("Data Type:", constant_tensor.dtype)
    print("-" * 30)

# 2. tf.zeros: Creating a tensor filled with zeros
def test_tf_zeros():
    print("2. tf.zeros:")
    zero_tensor = tf.zeros(shape=(3, 3))
    print(zero_tensor)
    print("Shape:", zero_tensor.shape)
    print("Data Type:", zero_tensor.dtype)
    print("-" * 30)

# 3. tf.ones: Creating a tensor filled with ones
def test_tf_ones():
    print("3. tf.ones:")
    ones_tensor = tf.ones(shape=(2, 2))
    print(ones_tensor)
    print("Shape:", ones_tensor.shape)
    print("Data Type:", ones_tensor.dtype)
    print("-" * 30)

# 4. tf.eye: Creating an identity matrix
def test_tf_eye():
    print("4. tf.eye:")
    identity_tensor = tf.eye(3)
    print(identity_tensor)
    print("Shape:", identity_tensor.shape)
    print("Data Type:", identity_tensor.dtype)
    print("-" * 30)

# 5. tf.fill: Creating a tensor filled with a scalar value
def test_tf_fill():
    print("5. tf.fill:")
    filled_tensor = tf.fill([2, 3], 7)
    print(filled_tensor)
    print("Shape:", filled_tensor.shape)
    print("Data Type:", filled_tensor.dtype)
    print("-" * 30)

# 6. tf.linspace: Generating a sequence of evenly spaced values
def test_tf_linspace():
    print("6. tf.linspace:")
    linspace_tensor = tf.linspace(start=0.0, stop=1.0, num=5)
    print(linspace_tensor)
    print("Shape:", linspace_tensor.shape)
    print("Data Type:", linspace_tensor.dtype)
    print("-" * 30)

# 7. tf.range: Generating a sequence of numbers
def test_tf_range():
    print("7. tf.range:")
    range_tensor = tf.range(start=0, limit=10, delta=2)
    print(range_tensor)
    print("Shape:", range_tensor.shape)
    print("Data Type:", range_tensor.dtype)
    print("-" * 30)

# Running all tests
def run_all_tests():
    print("TensorFlow Known Values Creation: Test and Learning Tool\n" + "="*60)
    test_tf_constant()
    test_tf_zeros()
    test_tf_ones()
    test_tf_eye()
    test_tf_fill()
    test_tf_linspace()
    test_tf_range()
    print("All tests completed successfully!")

# Run the test functions
if __name__ == "__main__":
    run_all_tests()
