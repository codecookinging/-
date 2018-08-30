import tensorflow as tf
import cv2
import os
import numpy as np

from PIL import Image
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('test_file', 'test_list.txt', 'test dataset file')

test_file = open(FLAGS.test_file)

testpath='round1_test_a'

test_img_name=os.listdir(testpath)

images=[]

test_file = open(FLAGS.test_file)

testpath='round1_test_a'


test_img_name=os.listdir(testpath)




lines = test_file.readlines()
for line in lines:
    line=line.strip()
    images.append(line)



for i in range(len(images)):

    img = cv2.imread(images[i])

    #cv2.GaussianBlur(img, (5, 5), 1)  #高斯滤波器，使得图像平滑
    img = cv2.resize(img,(224, 224),interpolation=cv2.INTER_AREA) ##这个插值方法 抗混叠

    img = img.astype(np.float32)


    cv2.imwrite(os.path.join('imageblur',str(i)+'.jpg'),img)









