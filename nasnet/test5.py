import datetime


train_path=['../round1_train_part1','../round1_train_part2','../round1_train_part3']
img_path=[]
lbl_list=[]
class_list=[]
import os
import numpy as np
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
                elif class_index not in ['正常','扎洞','毛斑','擦洞','织稀','吊经','缺经','跳花','油渍','污渍','毛洞']:
                    lbl_list.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1])
                    class_list.append(10)


print(len(img_path))



print('positive number:',class_list.count(0))

print(' number:',class_list.count(2))
print(' number:',class_list.count(3))
print(' number:',class_list.count(4))
print(' number:',class_list.count(5))
print(' number:',class_list.count(6))


print(' number:',class_list.count(7))
print(' number:',class_list.count(8))
print(' number:',class_list.count(9))
print(' number:',class_list.count(10))

#print(class_list)
print(len(class_list))


# a=np.array([12,15,2,4])
#
# print(a.argmax(axis=-1))
classes =['norm','defect_1','defect_2','defect_3','defect_4','defect_5','defect_6','defect_7','defect_8','defect_9','defect_10']

class_to_id=dict(zip(list(range(11)),classes))

import pandas as pd

print(class_to_id)

print(class_to_id[0])



pred=[0,1,2,4]
test_img_name=['1.jpg','2.jpg','3.jpg','4.jpg']
y_pred=[0.1,0.2,0.3,0.4]
classname=[]



test_path='../xuelang_round2_test_a'
test_img_name=os.listdir(test_path)

print(len(test_img_name))

print(test_img_name[:4])



