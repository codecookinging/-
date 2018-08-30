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
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)




iternum=np.floor(len(images) / FLAGS.batch_size).astype(np.int16)

print(iternum)

def savecsv(result):


    df = pd.DataFrame({'filename':test_img_name,'probability':np.char.mod('%.8f', result)})
    df.to_csv('pred.csv',index=None, header=None)





result=[]
for i in range(iternum+1):
    A=sess.run([score],feed_dict={img_input:next_batch(images)})
    A=np.squeeze(A)

    #print(A.shape)


    result1=sess.run(tf.nn.softmax(A)[:,1])


    result.extend(result1)

print(len(result))

print(len(test_img_name))
savecsv(result)







sess.close()











#print(softmax(A))
#score=np.squeeze(A)
#
# prob=(softmax(score))
# print(prob)


























