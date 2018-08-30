import os,sys

import numpy as np
import cv2
import tensorflow as tf
import os
import datetime
from xuelang.vgg.model import VggNetModel
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#from xuelang.utils.preprocessor import BatchPreprocessor
from xuelang.utils.testdata import BatchPreprocessor

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.app.flags.DEFINE_integer('num_epochs', 10, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', 'fc8,fc7', 'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string('multi_scale', '', 'As preprocessing; scale the image randomly between 2 numbers and crop randomly at network\'s input size')
tf.app.flags.DEFINE_string('training_file', '../data/train_list.txt', 'Training dataset file')
tf.app.flags.DEFINE_string('val_file', '../data/val_list.txt', 'Validation dataset file')
tf.app.flags.DEFINE_string('tensorboard_root_dir', '../training', 'Root directory to put the training logs and weights')
tf.app.flags.DEFINE_integer('log_step', 10, 'Logging period in terms of iteration')

FLAGS = tf.app.flags.FLAGS





train_preprocessor = BatchPreprocessor(dataset_file_path=FLAGS.training_file, num_classes=FLAGS.num_classes,
                                       output_size=[224, 224], horizontal_flip=True, shuffle=True,
                                       multi_scale=None)
val_preprocessor = BatchPreprocessor(dataset_file_path=FLAGS.val_file, num_classes=FLAGS.num_classes,
                                     output_size=[224, 224])
print(type(train_preprocessor.images))
print(len(train_preprocessor.images))
print(len(train_preprocessor.labels))


images,labels=val_preprocessor.next_batch(2)
print(labels)

#print(images[0])

#print(images[8])

print(val_preprocessor.labels)

print(type(train_preprocessor.labels))
#np.savetxt("resut.txt",train_preprocessor.images)

#img=cv2.imread('./round1_train_part2/正常/J01_2018.06.13 14_38_37.jpg')
#print(img)