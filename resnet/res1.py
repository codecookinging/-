

#this file test to load  the ckpt model

# -*- coding: utf-8 -*-
__author__ = 'saijunz'

import  model
import os
import sys
import tensorflow as tf
from tensorflow.contrib import slim
import datetime
weight_decay=1e-5
import numpy as np
from nets import resnet_v1
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

from utils.processor512 import BatchPreprocessor
tf.app.flags.DEFINE_string('pretrained_model_path', 'tmp/resnet_v1_50.ckpt', '')
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for adam optimizer')

tf.app.flags.DEFINE_integer('num_epochs', 50, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Batch size')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
tf.app.flags.DEFINE_string('training_file', '../data/train_list.txt', 'Training dataset file')

tf.app.flags.DEFINE_string('val_file', '../data/val_list.txt', 'Validation dataset file')

tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpath', 'path')


FLAGS = tf.app.flags.FLAGS


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
FLAGS=tf.app.flags.FLAGS

print(FLAGS.pretrained_model_path)
#saver = tf.train.Saver(tf.global_variables())

#saver = tf.train.Saver(tf.global_variables())

#images =np.random.normal(size=(1,512,512,3))

#images=images.astype(np.float32)


# Placeholders
x = tf.placeholder(tf.float32, [FLAGS.batch_size, 512, 512, 3])
y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])


dropout_keep_prob = tf.placeholder(tf.float32)

loss = model.loss(x,y)

# 定义global step
global_step = tf.Variable(0, trainable=False)
FLAGS.learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=2000, decay_rate=0.5,
                                                 staircase=True)
def optimize(learning_rate,global_step=0):
    var_list = slim.get_trainable_variables()
    return tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=var_list,global_step=global_step)


# optimzer the network
train_op =optimize(FLAGS.learning_rate, global_step)


#get the model
logits=model.model(x, weight_decay=1e-5, is_training=True,dropout=0.5)

logits=tf.squeeze(logits,[1,2])

# Training accuracy of the model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))



accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


train_preprocessor = BatchPreprocessor(dataset_file_path=FLAGS.training_file, num_classes=FLAGS.num_classes,
                                       output_size=[512, 512], horizontal_flip=True, shuffle=True, multi_scale=None)

val_preprocessor = BatchPreprocessor(dataset_file_path=FLAGS.val_file, num_classes=FLAGS.num_classes, output_size=[512, 512])

train_batches_per_epoch = np.floor(len(train_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(len(val_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)



'''
加了 可变学习率 
'''




var_list = slim.get_trainable_variables()  # get trainable varlist
#print('var_list:',var_list)







variable_restore_op = slim.assign_from_checkpoint_fn('tmp/resnet_v1_50.ckpt', var_list,ignore_missing_vars=True)

saver = tf.train.Saver()
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    variable_restore_op(sess)

    print("{} Start training...".format(datetime.datetime.now()))

    for epoch in range(FLAGS.num_epochs):
        print("{} Epoch number: {}".format(datetime.datetime.now(), epoch + 1))
        step = 1
        '''
        if FLAGS.learning_rate>0.000001:
            FLAGS.learning_rate=FLAGS.learning_rate/10
        '''

        # Start training
        while step < train_batches_per_epoch:
            batch_xs, batch_ys = train_preprocessor.next_batch(FLAGS.batch_size)
            sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob: FLAGS.dropout_keep_prob})
            trainacc = sess.run(accuracy,
                                feed_dict={x: batch_xs, y: batch_ys, dropout_keep_prob: FLAGS.dropout_keep_prob})

            print('trainaccuary', trainacc)


            step += 1

        # Epoch completed, start validation
        print("{} Start validation".format(datetime.datetime.now()))
        test_acc = 0.
        test_count = 0

        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_preprocessor.next_batch(FLAGS.batch_size)
            acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, dropout_keep_prob: 1.})
            test_acc += acc
            test_count += 1
            print("{} one_batch Accuracy = {:.4f}".format(datetime.datetime.now(), acc))
        test_acc /= test_count



        print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))

        # Reset the dataset pointers
        val_preprocessor.reset_pointer()
        train_preprocessor.reset_pointer()

        print("{} Saving checkpoint of model...".format(datetime.datetime.now()))

        # save checkpoint of the model
        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model_epoch' + str(epoch + 1) + '.ckpt')

        save_path = saver.save(sess, checkpoint_path)

        print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_path))