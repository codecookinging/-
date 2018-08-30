import os,sys


import numpy as np
import tensorflow as tf
import datetime
import pandas as pd
from model import VggNetModel
import cv2
#lib_path = os.path.abspath(os.path.join('..'))
#sys.path.append(lib_path)
#print(lib_path)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#from xuelang.utils.preprocessor import BatchPreprocessor
#from utils.testdata import  BatchPreprocessor
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for adam optimizer')

tf.app.flags.DEFINE_integer('pointer', 0, '')

tf.app.flags.DEFINE_float('dropout_keep_prob', 0.6, 'Dropout keep probability')
tf.app.flags.DEFINE_integer('num_epochs', 50, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', 'fc8,fc7,fc6,conv5_3', 'Finetuning layers seperated by commas')
tf.app.flags.DEFINE_string('multi_scale', '', 'As preprocessing; scale the image randomly between 2 numbers and crop randomly at network\'s input size')
tf.app.flags.DEFINE_string('training_file', '../data/train_list.txt', 'Training dataset file')
tf.app.flags.DEFINE_string('val_file', '../data/val_list.txt', 'Validation dataset file')
tf.app.flags.DEFINE_string('test_file', 'test_list.txt', 'test dataset file')
tf.app.flags.DEFINE_string('tensorboard_root_dir', '../training', 'Root directory to put the training logs and weights')
tf.app.flags.DEFINE_integer('log_step', 10, 'Logging period in terms of iteration')

FLAGS = tf.app.flags.FLAGS
images=[]
#test_preprocessor = BatchPreprocessor(dataset_file_path=FLAGS.test_file, num_classes=FLAGS.num_classes, output_size=[224, 224])



def softmax(x):
    return np.exp(x[:,1])/np.sum(np.exp(x),axis=0)


test_file = open(FLAGS.test_file)

testpath='round1_test_a'
test_img_name=os.listdir(testpath)


lines = test_file.readlines()
for line in lines:
    line=line.strip()
    images.append(line)

ckpt_filename = '../training/vggnet_20180726_093139/checkpoint/model_epoch4.ckpt'
mean_color=[132.2766, 139.6506, 146.9702]
#print(images[0])




def next_batch(images):


    if FLAGS.pointer + FLAGS.batch_size > len(images):



        paths = images[FLAGS.pointer:len(images)]
        # print(len(paths))

    else:


        paths = images[FLAGS.pointer:FLAGS.pointer + FLAGS.batch_size]
        FLAGS.pointer += FLAGS.batch_size
        print(FLAGS.pointer)



    image = np.ndarray([len(paths), 224, 224, 3])






    for i in range(len(paths)):
        img = cv2.imread(paths[i])
        img=cv2.resize(img,(224,224))
        img=img.astype(np.float32)
        img -= np.array(mean_color)
        image[i] = img
    return image








img_input = tf.placeholder(tf.float32, shape=[None,224, 224, 3])



score=VggNetModel(num_classes=FLAGS.num_classes, dropout_keep_prob=1).inference(img_input,False)  #get score

sess=tf.Session()
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
#
# saver.restore(sess, ckpt_filename)


print(tf.global_variables())

print(tf.trainable_variables())

sess.close()




weights = np.load('vgg16_weights.npz',encoding="latin1")

print(weights.keys())


#keys = sorted(weights.keys())

#print(keys)

# for i, name in enumerate(keys):
#     parts = name.split('_')
#     layer = '_'.join(parts[:-1])
#
#     # if layer in skip_layers:
#     #     continue
#
#     if layer == 'fc8' and self.num_classes != 1000:
#         continue
#
#     with tf.variable_scope(layer, reuse=True):
#         if parts[-1] == 'W':
#             var = tf.get_variable('weights')
#             session.run(var.assign(weights[name]))
#         elif parts[-1] == 'b':
#             var = tf.get_variable('biases')
#             session.run(var.assign(weights[name]))

















