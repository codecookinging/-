import numpy as np

weights = np.load('VGG_imagenet.npy',encoding="latin1")

print(weights)

for i in range(10):
    print(i)