# -*- coding: utf-8 -*-

import os,glob

def get_images(data_dir):
    files = []
    for data_path in data_dir:
        for ext in ['jpg', 'png', 'jpeg', 'JPG']:
            files.extend(glob.glob(
                os.path.join(data_path,'*.{}'.format(ext))))
    return files

imgs = get_images(['./round1_test_a/'])

print(len(imgs))
with open('test_list.txt','a+') as fw:
    for img in imgs:

        fw.write('{}\n'.format(img))


'''
print(len(imgs))
with open('train_list.txt','a+') as fw:
    for img in imgs:
        tag = img.split('/')[-2]
        if tag == '正常':
            fw.write('{}|{}\n'.format(img,0))
        else:
            fw.write('{}|{}\n'.format(img,1))

'''



