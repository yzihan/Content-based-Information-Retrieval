#!/usr/bin/env python
import numpy as np
from balltree import BallTree
import random

if __name__ != '__main__':
    print('You cannot import this file!')
    exit(1)

x = [[1,2],[2,3],[3,4],[1,12],[2,13],[3,14],[11,0],[12,1],[13,-1]]
random.shuffle(x)
balltree = BallTree(np.array(x, dtype=np.float))
print(balltree.search([2,2], 3))
print(balltree.search([2,10], 3))
print(balltree.search([10,2], 3))

x = [1.0 * x for x in [-1,1,2,3,101,102,103,104,-202,-201,-203,-204] ]
x.extend([ 0.01 * x for x in range(-100, 100, 1) ])
x.extend([ 0.01 * x for x in range(9900, 10000, 1) ])
x.extend([ 0.01 * x for x in range(-20100, -20001, 1) ])
random.shuffle(x)

balltree = BallTree(np.array(x, dtype=np.float))
print(balltree.search(np.array([0.])))
print(balltree.search(np.array([100.])))
print(balltree.search(np.array([-200.])))
