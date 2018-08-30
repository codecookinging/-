import os
from tensorflow.python import pywrap_tensorflow
import random

from dataprocess import BatchPreprocessor
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import cv2
import numpy as np
import tensorflow as tf

from nets import resnet_v1
# from nets import resnet_utils
# resnet_arg_scope = resnet_utils.resnet_arg_scope

from tensorflow.contrib import slim

import cv2 as cv
from tqdm import tqdm
import multiprocessing
train_path=['../round1_answer_a','../round1_answer_b','../round1_train_part1','../round1_train_part2','../round1_train_part3']

img_path=[]
lbl_list=[]
class_list=[]


for part_index in train_path:
    class_path_list=os.listdir(part_index)
    for class_index in class_path_list:
        img_path_list=os.listdir(os.path.join(part_index,class_index))
        for img_index in img_path_list:
            if img_index[-3:]=='jpg':
                img_path.append(os.path.join(part_index,class_index,img_index))
                if class_index=='正常':
                    lbl_list.append([1,0,0,0,0,0,0,0,0,0,0])
                    class_list.append(0)
                if class_index=='扎洞':
                    lbl_list.append([0,1,0,0,0,0,0,0,0,0,0])
                    class_list.append(1)
                if class_index=='毛斑':
                    lbl_list.append([0,0,1,0,0,0,0,0,0,0,0])
                    class_list.append(2)
                if class_index=='擦洞':
                    lbl_list.append([0,0,0,1,0,0,0,0,0,0,0])
                    class_list.append(3)
                if class_index=='毛洞':
                    lbl_list.append([0,0,0,0,1,0,0,0,0,0,0])
                    class_list.append(4)
                if class_index=='织稀':
                    lbl_list.append([0,0,0,0,0,1,0,0,0,0,0])
                    class_list.append(5)
                if class_index=='吊经':
                    lbl_list.append([0,0,0,0,0,0,1,0,0,0,0])
                    class_list.append(6)
                if class_index=='缺经':
                    lbl_list.append([0,0,0,0,0,0,0,1,0,0,0])
                    class_list.append(7)
                if class_index == '跳花':
                    lbl_list.append([0, 0, 0, 0, 0, 0, 0, 0, 1,0,0])
                    class_list.append(8)
                if class_index == '油渍' or class_index == '污渍':
                    lbl_list.append([0, 0, 0, 0, 0, 0, 0, 0, 0,1,0])
                    class_list.append(9)
                elif class_index not in ['正常', '扎洞', '毛斑', '擦洞', '织稀', '吊经', '缺经', '跳花', '油渍', '污渍', '毛洞']:
                    lbl_list.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
                    class_list.append(10)
print('positive number:',class_list.count(0))

print(' number:',class_list.count(1))
print(' number:',class_list.count(2))
print(' number:',class_list.count(3))
print(' number:',class_list.count(4))
print(' number:',class_list.count(5))
print(' number:',class_list.count(6))
print(' number:',class_list.count(7))
print(' number:',class_list.count(8))
print(' number:',class_list.count(9))
print(' number:',class_list.count(10))

print('image number:',len(img_path))
print('label number:',len(lbl_list))

n=len(img_path)
width=512

index_list=list(range(n))
random.shuffle(index_list)
img_path_shuf=[]
lbl_list_shuf=[]
class_list_shuf=[]

for i,index in enumerate(index_list):
    img_path_shuf.append(img_path[index])
    lbl_list_shuf.append(lbl_list[index])
    class_list_shuf.append(class_list[index])


img_path=img_path_shuf
lbl_list=lbl_list_shuf
class_list=class_list_shuf
lbl_list=np.array(lbl_list)


batch_size=8
num_epochs=50


def read_img(index):


    return index, cv.resize(cv.imread(img_path[index]),(width,width),interpolation=cv.INTER_AREA)

img_list = np.zeros((n, width, width, 3), dtype=np.float32)
with multiprocessing.Pool(16) as pool:
    with tqdm(pool.imap_unordered(read_img, range(n)), total=n) as pbar:
        for i, img in pbar:
            img_list[i] = img[:,:,::-1]
            img_list[i]-=[132.2766, 139.6506, 146.9702]

n_train = int(n*0.95)

X_train = img_list[:n_train]
X_valid = img_list[n_train:]
y_train = lbl_list[:n_train]
y_valid = lbl_list[n_train:]
print(len(X_train),len(X_valid))
print(len(y_train),len(y_valid))

train_preprocessor = BatchPreprocessor(x_batch=X_train,y_batch=y_train)


val_preprocessor = BatchPreprocessor(x_batch=X_valid,y_batch=y_valid)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(len(y_train) //batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(len(y_valid) // batch_size).astype(np.int16)

# Placeholders
x = tf.placeholder(tf.float32, [batch_size, width, width, 3])
y = tf.placeholder(tf.float32, [None,11])



learning_rate=0.1

# tf.reset_default_graph()
# img1 = cv2.imread('../round1_test_a/J01_2018.06.13 13_22_11.jpg')
#
# img1 = img1.astype(np.float32)
# img1 -= [102.9801, 115.9465, 122.7717]
#
# img1 = cv2.resize(img1, (512, 512))
#
# img1 = tf.convert_to_tensor(img1)
# img1 = tf.expand_dims(img1, 0)
#
# print(img1.shape)

with slim.arg_scope(resnet_v1.resnet_arg_scope()) as sc:
    net, end = resnet_v1.resnet_v1_50(x, 11, is_training=True)



y_predict=end['predictions']

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))


def focal_loss_softmax(labels,logits,gamma=2):

    y_pred=tf.nn.softmax(logits,dim=-1)

    L = -labels * ((1 - y_pred) ** gamma) * tf.log(y_pred)
    L = tf.reduce_sum(L, axis=1)

    return L







global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=200, decay_rate=0.5,
                                                 staircase=True)

# admam to minimize loss
# var list is right?



import datetime

correct_pred = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



    # conv6=tf.get_variable('conv6',[16,16,2048,2048])

    # conv7=tf.get_variable('conv7',[1,1,2018,2048])
    #input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input')

    # for var in tf.trainable_variables():
    #     print(var.name)

    #print('################      global        ############## ')

    # for var in tf.global_variables():
    #     print(var.name)

pretrained_model='resnet_v1_50.ckpt'
reader = pywrap_tensorflow.NewCheckpointReader(pretrained_model)  # 这个可以通过 字典知道 ckpt原来是个字典
variable_assign=[]
var_to_shape_map = reader.get_variable_to_shape_map()
print('################      key of ckpt         ############## ')
for key in var_to_shape_map:

    if key=='resnet_v1_50/logits/weights'or 'resnet_v1_50/logits/biases':
        continue
    variable_assign.append(key)



    print("tensor_name", key)
        #print(reader.get_tensor(key).shape)

    # for var in variable_assign:
    #     print(var)

exclude=['resnet_v1_50/logits/weights','resnet_v1_50/logits/biases']

variable_to_restore=slim.get_variables_to_restore(exclude=exclude)

print('assign value........',len(variable_to_restore))


print(len(tf.trainable_variables()))
print(len(tf.global_variables()))

#adam will increase the number of parameter pre:267 now:562
for var in variable_to_restore:
    print(var)





saver = tf.train.Saver(variable_to_restore)

train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=tf.trainable_variables(),global_step=global_step)

checkpoint_dir='checkpoint'

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, 'resnet_v1_50.ckpt')
    print(net.shape)

    for epoch in range(num_epochs):
        print("{} Epoch number: {}".format(datetime.datetime.now(), epoch + 1))
        step = 1
        # Start training
        while step < train_batches_per_epoch:
            batch_xs, batch_ys = train_preprocessor.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
            sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})
            trainacc = sess.run(accuracy,feed_dict={x: batch_xs, y: batch_ys})

            step += 1
        print('trainaccuary', trainacc)
        print('loss',loss)



        # Epoch completed, start validation
        print("{} Start validation".format(datetime.datetime.now()))
        test_acc = 0.
        test_count = 0

        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_preprocessor.next_batch(batch_size)
            sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})

            acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty})
            test_acc += acc
            test_count += 1
            #print("{} one_batch Accuracy = {:.4f}".format(datetime.datetime.now(), acc))
        test_acc /= test_count
        print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))
        print("{} valloss = {}".format(datetime.datetime.now(), loss))
        # Reset the dataset pointers
        val_preprocessor.reset_pointer()
        train_preprocessor.reset_pointer()

        print("{} Saving checkpoint of model...".format(datetime.datetime.now()))

        # save checkpoint of the model
        checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_path)


    #print(end)
    #print(sess.run(end['predictions']))
