#!/usr/bin/env python
import os
from densenet import DenseNetPrediction
from dataset import DataSet
from balltree import BallTree
from collections import Counter
import time

if __name__ != '__main__':
    print('You cannot import this file!')
    exit(1)

DATASET = '../../data'
dataset = DataSet(DATASET, '.jfif')
dataset.shuffle()
print('Dataset length={}'.format(len(dataset)))
DISTANCE = 'L2'

overall_test = 0
overall_correct = 0

K_FOLD=6
KNN_K=3

for pos in range(K_FOLD):
    test_pos = pos
    test_split = 1. / K_FOLD

    descriptor_class = DenseNetPrediction()

    ta = time.time()

    descriptors, tests = descriptor_class.get_clusters_descriptors(dataset, test_split = test_split, test_pos = test_pos)

    print('Setting up accelerator...')
    tb = time.time()

    tree = BallTree([ descriptors[i]['feature_vector'] for i in range(len(descriptors)) ])

    print('Describing...')
    a = time.time()

    prepared = descriptor_class.prepare_query(tests.get_data())

    print('Predicting...')
    b = time.time()

    res_top = descriptor_class.get_prediction(descriptors, DISTANCE, prepared, mode=2, accelerator=tree, nearby_count=KNN_K)

    c = time.time()
    print('Dataset descriptor calculation time cost: {}'.format(tb - ta))
    print('Classification accelerator calculation time cost: {}'.format(a - tb))
    print('Describing time cost: {}'.format(b - a))
    print('Predicting time cost: {}'.format(c - b))

    predicted = []

    for pred_result, correct in zip(res_top, list(tests.get_data()['class_image'])):
        res_count = Counter([ descriptors[x[0]]['class_image'] for x in pred_result ])

        class_pred, pred_count = min(res_count.items(), key=lambda x: (-x[1], x[0]))
        if pred_count == 1: # not reliable result
            class_pred = descriptors[pred_result[0][0]]['class_image']
            # print('override to', class_pred)

        print(pred_result)

        predicted.append(class_pred)

    res = tests.get_data().copy()
    res['predicted_class'] = predicted
    res['correct'] = res['predicted_class'] == res['class_image']
    overall_correct += res['correct'].sum()
    overall_test += len(res)
    print(res)

print(overall_correct, '/', overall_test)
