import tensorflow as tf

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

print(sess.run(hello))

from tensorflow.python.client import device_lib

print(tf.test.is_gpu_available())
