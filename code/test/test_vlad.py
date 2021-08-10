#!/usr/bin/env python
import os
from vlad import VladPrediction
from dataset import DataSet
import time

if __name__ != '__main__':
    print('You cannot import this file!')
    exit(1)

DATASET = '../data'
dataset = DataSet(DATASET, '.jfif')
print('Dataset length={}'.format(len(dataset)))
DISTANCE = 'L2'

QUERY = 'test.jfif'

vlad_class = VladPrediction(dataset, DATASET, QUERY)

kmeans_clusters, vlad_descriptors = vlad_class.get_clusters_vlad_descriptors(dataset)


print('Describing...')
a = time.time()

prepared = vlad_class.prepare_query()

print('Predicting...')
b = time.time()

class_pred = vlad_class.get_prediction(kmeans_clusters, vlad_descriptors, DISTANCE, prepared, mode=1)

c = time.time()
print('Describing time cost: {}', b - a)
print('Predicting time cost: {}', c - b)

print(class_pred)
