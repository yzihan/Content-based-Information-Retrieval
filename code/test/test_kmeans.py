#!/usr/bin/env python
from kmeans import KMeans
import numpy as np
import pickle

if __name__ != '__main__':
    print('You cannot import this file!')
    exit(1)

km = KMeans(2)
X = np.array([[1,2],[2,3],[3,4],[4,5],[5,6]], dtype=np.float)
clusters = km.fit(X)
print(clusters)
print(clusters.n_clusters)
print(clusters.cluster_centers_)
print(clusters.predict(np.array([[1,2],[2,3],[3,4],[4,5],[5,6]])))