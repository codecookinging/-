import numpy as np
import cv2


class BatchPreprocessor(object):

    def __init__(self, mean_color=[132.2766, 139.6506, 146.9702],x_batch=None,y_batch=None):


        self.mean_color = mean_color
        self.pointer = 0
        self.x_batch = x_batch
        self.y_batch = y_batch


    def reset_pointer(self):
        self.pointer = 0

    def gety(self):
        return self.y_batch


    def next_batch(self, batch_size):
        x = np.ndarray([8, 512, 512, 3])
        y = np.ndarray([8, 11])
        #y= np.ndarray([None, 11])
        # Get next batch of image (path) and labels
        # x = self.x_batch[self.pointer:self.pointer + batch_size]
        # y = self.y_batch[self.pointer:self.pointer + batch_size]
        for i in range(batch_size):
            x[i]=self.x_batch[i+self.pointer]
            y[i]=self.y_batch[i+self.pointer]

        # Update pointer
        self.pointer += batch_size


        return x,y


