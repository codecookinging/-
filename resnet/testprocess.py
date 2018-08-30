import os
from tensorflow.python import pywrap_tensorflow
import random

from dataprocess import BatchPreprocessor
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
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
train_path=['../round1_train_part3']
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

# print(type(img_list))
#
# print(type(X_train))
#
# print(X_train.shape)
#
# x=np.ndarray([batch_size, 512, 512, 3])

# for i in range(0,8):
#     x[i]=X_train[i]
#
# print(x)


#train_preprocessor = BatchPreprocessor(X_train,y_train)

#print(train_preprocessor.)



train_preprocessor = BatchPreprocessor(x_batch=X_train,y_batch=y_train)


val_preprocessor = BatchPreprocessor(X_valid,y_valid)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(len(y_train) /batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(len(y_valid) / batch_size).astype(np.int16)

# i=0
#
x,y=train_preprocessor.next_batch(batch_size)
#
