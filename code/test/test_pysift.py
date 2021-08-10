#!/usr/bin/env python

import cv2

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter
import pysift

if __name__ != '__main__':
    print('You cannot import this file!')
    exit(1)

image = cv2.imread('../data/cat/1.jfif')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kp, des = pysift.computeKeypointsAndDescriptors(image)
print(kp)
print(des)
