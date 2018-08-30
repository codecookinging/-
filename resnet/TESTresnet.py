from tensorflow.python import pywrap_tensorflow
import tensorflow.contrib.slim as slim
import cv2


pretrained_model='tmp/resnet_v1_50.ckpt'

reader=pywrap_tensorflow.NewCheckpointReader(pretrained_model)

var_to_shape_map=reader.get_variable_to_shape_map()


print(var_to_shape_map)



