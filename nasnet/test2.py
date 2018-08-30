import random
import numpy as np
import pandas as pd
from collections import Counter
import cv2 as cv
from tqdm import tqdm
import multiprocessing

import keras.backend as K
import tensorflow as tf

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import l2
from keras.preprocessing.image import *
from keras import backend as K
from keras.utils import multi_gpu_model

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

train_path=['../round1_train_part2','../round1_train_part3']

val_path=['../round1_train_part1']
img_path=[]
lbl_list=[]
class_list=[]

img_path1=[]
lbl_list1=[]
class_list1=[]

for part_index in train_path:
    class_path_list=os.listdir(part_index)
    for class_index in class_path_list:
        img_path_list=os.listdir(os.path.join(part_index,class_index))
        for img_index in img_path_list:
            if img_index[-3:]=='jpg':
                img_path.append(os.path.join(part_index,class_index,img_index))
                if class_index=='正常':
                    lbl_list.append([1,0])
                    class_list.append(0)
                else:
                    lbl_list.append([0,1])
                    class_list.append(1)


for part_index1 in val_path:
    val_list=os.listdir(part_index1)
    for val_index in val_list:
        img_path_list1=os.listdir(os.path.join(part_index1,val_index))
        for img_index1 in img_path_list1:
            if img_index1[-3:]=='jpg':
                img_path1.append(os.path.join(part_index1,val_index,img_index1))
                if val_index=='正常':
                    lbl_list1.append([1,0])
                    class_list1.append(0)
                else:
                    lbl_list1.append([0,1])
                    class_list1.append(1)




print('positive number:',class_list.count(0))
print('negative number:',class_list.count(1))
print('image number:',len(img_path))
print('label number:',len(lbl_list))

n=len(img_path)
n1=len(img_path1)


width=331


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






def read_img(index):
    return index, cv.resize(cv.imread(img_path[index]),(width,width),interpolation=cv.INTER_AREA)

def read_img1(index):
    return index, cv.resize(cv.imread(img_path1[index]),(width,width),interpolation=cv.INTER_AREA)
img_list = np.zeros((n, width, width, 3), dtype=np.uint8)
with multiprocessing.Pool(16) as pool:
    with tqdm(pool.imap_unordered(read_img, range(n)), total=n) as pbar:
        for i, img in pbar:
            img_list[i] = img[:,:,::-1]

img_list1= np.zeros((n1, width, width, 3), dtype=np.uint8)
with multiprocessing.Pool(16) as pool:
    with tqdm(pool.imap_unordered(read_img1, range(n1)), total=n1) as pbar:
        for i, img in pbar:
            img_list1[i] = img[:,:,::-1]


lbl_list1=np.array(lbl_list1)
X_train = img_list
X_valid = img_list1
y_train = lbl_list
y_valid = lbl_list1
print(len(X_train),len(X_valid))
print(len(y_train),len(y_valid))

class Generator():
    def __init__(self, X, y, batch_size=8, aug=False):
        def generator():
            idg = ImageDataGenerator(horizontal_flip=True,
                                     vertical_flip=True,
                                    )
            while True:
                for i in range(0, len(X), batch_size):
                    X_batch = X[i:i+batch_size].copy()
                    y_barch = y[i:i+batch_size].copy()
                    if aug:
                        for j in range(len(X_batch)):
                            X_batch[j] = idg.random_transform(X_batch[j])
                    yield X_batch, y_barch
        self.generator = generator()
        self.steps = len(X) // batch_size + 1

gen_train = Generator(X_train, y_train, batch_size=8, aug=True)

def acc(y_true, y_pred):
    index = tf.reduce_any(y_true > 0.5, axis=-1)
    res = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    index = tf.cast(index, tf.float32)
    res = tf.cast(res, tf.float32)
    return tf.reduce_sum(res * index) / (tf.reduce_sum(index) + 1e-7)

base_model = NASNetLarge(weights='imagenet', input_shape=(width, width, 3), include_top=False, pooling='avg')

input_tensor = Input((width, width, 3))
x = input_tensor
x = Lambda(nasnet.preprocess_input)(x)
x = base_model(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax', name='softmax')(x)

model = Model(input_tensor, x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=[acc])

model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=5, validation_data=(X_valid, y_valid))



model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=[acc])
model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=5, validation_data=(X_valid, y_valid))
model.compile(optimizer=Adam(1e-6), loss='categorical_crossentropy', metrics=[acc])
model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=5, validation_data=(X_valid, y_valid))
model.compile(optimizer=Adam(1e-7), loss='categorical_crossentropy', metrics=[acc])
#model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=20, validation_data=(X_valid, y_valid))
#model.compile(optimizer=Adam(1e-8), loss='categorical_crossentropy', metrics=[acc])
#model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=5, validation_data=(X_valid, y_valid))

y_pred = model.predict(X_valid, batch_size=32, verbose=1)
print(y_pred)

pred=y_pred.argmax(axis=-1)
label=y_valid.argmax(axis=-1)
print(pred)
print(label)

cnt=0
for i,lbl in enumerate(label):
    if lbl==pred[i]:
        cnt=cnt+1
acc=cnt/len(label)
print('Valid acc:',acc)

test_path='../round1_test_b'
test_img_name=os.listdir(test_path)
n_test=len(test_img_name)

def read_img_test(index):
    return index, cv.resize(cv.imread(os.path.join(test_path,test_img_name[index])),(width,width),interpolation=cv.INTER_AREA)
test_img_list = np.zeros((n_test, width, width, 3), dtype=np.uint8)
with multiprocessing.Pool(12) as pool:
    with tqdm(pool.imap_unordered(read_img_test, range(n_test)), total=n_test) as pbar:
        for i, img in pbar:
            test_img_list[i] = img[:,:,::-1]

y_pred = model.predict(test_img_list, batch_size=32, verbose=1)
print(y_pred)

df = pd.DataFrame({'filename':test_img_name,'probability':np.char.mod('%.8f', y_pred[:,1])})
df.to_csv('../pred11.csv',index=None, header=None)
print('csv saved!')

model.save('model_nasnet11.h5')


print('model saved!')
