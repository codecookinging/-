import numpy as np

from keras.models import load_model
import cv2 as cv
from tqdm import tqdm
import multiprocessing
import datetime
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
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

width=331
base_model = NASNetLarge(weights='imagenet', input_shape=(width, width, 3), include_top=False, pooling='avg')

input_tensor = Input((331, 331, 3))
x = input_tensor
x = Lambda(nasnet.preprocess_input)(x)
x = base_model(x)
x = Dropout(0.5)(x)
x = Dense(11, activation='softmax', name='softmax')(x)
model = Model(input_tensor, x)


model = load_model('round2/model_nasnet.h5')

test_path='../xuelang_round2_test_a'
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
pred=y_pred.argmax(axis=-1)

y_pred=max(y_pred)

classes =['norm','defect_1','defect_2','defect_3','defect_4','defect_5','defect_6','defect_7','defect_8','defect_9','defect_10']

class_to_id=dict(zip(list(range(11)),classes))

classname=[]
for i in pred:
    classname.append(class_to_id[i])

with open('round2/1.txt','w') as f:

    for i in range(len(test_img_name)):
        f.write(test_img_name[i])
        f.write('|')
        f.write(classname[i])
        f.write('\t\t')
        f.write(str(y_pred[i]))
        f.write('\n')

