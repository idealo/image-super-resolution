import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
from .predictor import Predictor
