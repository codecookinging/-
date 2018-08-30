import shutil
import os
import glob

path=['round1_test_b']

def get_images(path):
    files = []
    for ext in ['jpg', 'png', 'jpeg']:
        files.extend(glob.glob(
            os.path.join(path,'*.{}'.format(ext))))
    return files



a=get_images('../round1_test_b')

print(a)
print(len(a))





files=[]

for ext in ['xml']:
    files.extend(glob.glob(
        os.path.join('../data/round1_answer_b','*','*.{}'.format(ext))))



for i in range(len(a)):

    imagename = a[i].split('/')[-1]
    imagename = imagename[:-4]
    #print(imagename)
    for j in files:
        if imagename in j:
            p, f = os.path.split(j)
            print(p)

            shutil.copy(a[i], p)
