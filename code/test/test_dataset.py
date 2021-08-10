#!/usr/bin/env python
import dataset

if __name__ != '__main__':
    print('You cannot import this file!')
    exit(1)

dataset = dataset.DataSet('../data', '.jfif')
print(len(dataset))
print(dataset.get_labels())
print(dataset.get_data())
