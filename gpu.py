import sys
import tensorflow as tf
#import tensorflow_metal
from tensorflow.python import keras
#import tensorflow.keras
import pandas as pd
#import sklearn as sk
#import scipy as sp
import tensorflow as tf
import platform
print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
#print(f"Tensorflow-metal version:{tensorflow-metal.__version__}")
print(tf.config.list_physical_devices())
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
#print(f"Scikit-Learn {sk.__version__}")
#print(f"SciPy {sp.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")    