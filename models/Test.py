import tensorflow as tf

print(tf.test.is_gpu_available())
print(tf.__version__)
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
import platform
import keras

print("Platform: {}".format(platform.platform()))
print("Tensorflow version: {}".format(tf.__version__))
print("Keras version: {}".format(keras.__version__))

# 返回当前设备索引

# import torch
# torch.cuda.current_device()
# #返回GPU的数量
# torch.cuda.device_count()
# #返回gpu名字，设备索引默认从0开始
# torch.cuda.get_device_name(0)
# #cuda是否可用
# torch.cuda.is_available()
