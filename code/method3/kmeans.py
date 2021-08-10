import numpy as np
from typing import *
import time
import matplotlib.pyplot as plt

# calculate Euclidean distance (batch mode)
def euclDistance(vector1: np.ndarray, vector2: np.ndarray):
	diff = vector2 - vector1
	return np.sqrt(np.sum(np.power(diff, 2), 1))

class KMeansCluster():
	# init centroids with random samples
	def initCentroids(self, dataSet: np.ndarray, k: int):
		n_samples, dim = dataSet.shape
		centroids = np.zeros((k, dim))
		for i in range(k):
			index = int(np.random.uniform(0, n_samples))
			centroids[i, :] = dataSet[index, :]
		return centroids

	# k-means cluster
	def __init__(self, samples: np.ndarray, n_clusters: int, distance = euclDistance):
		self.n_clusters = n_clusters
		self.distance = distance

		n_samples = samples.shape[0]
		# first column stores which cluster this sample belongs to,
		# second column stores the error between this sample and its centroid
		clusterAssessment = np.mat(np.zeros((n_samples, 2)))
		clusterChanged = True

		## step 1: init centroids
		centroids = self.initCentroids(samples, n_clusters)
		self.cluster_centers_ = centroids

		while clusterChanged:
			# clusterChanged = False
			# ## for each sample
			# for i in range(n_samples):
			# 	minDist  = 100000.0
			# 	minIndex = 0
			# 	## for each centroid
			# 	## step 2: find the centroid who is closest
			# 	for j in range(n_clusters):
			# 		dist = distance(np.stack([centroids[j, :]]), np.stack([X[i, :]]))[0]
			# 		if dist < minDist:
			# 			minDist = dist
			# 			minIndex = j

			# 	## step 3: update its cluster
			# 	if clusterAssessment[i, 0] != minIndex:
			# 		clusterChanged = True
			# 		clusterAssessment[i, :] = minIndex, minDist**2

			indices, minDist = self._predict(samples, centroids, distance)
			updated = clusterAssessment.A[:, 0] != indices
			clusterChanged = updated.any()
			if clusterChanged:
				clusterAssessment[updated] = np.stack((indices[updated], np.power(minDist[updated], 2)), 1)

			## step 4: update centroids
			for j in range(n_clusters):
				pointsInCluster = samples[np.nonzero(clusterAssessment[:, 0].A == j)[0]]
				centroids[j, :] = np.mean(pointsInCluster, axis = 0)

	def _predict(self, X: np.ndarray, centroids: np.ndarray, distance) -> Tuple[np.ndarray, np.ndarray]:
		indices = None
		minDist = None
		for i in range(self.n_clusters):
			dist = distance(centroids[i], X)
			if minDist is None:
				minDist = dist
				indices = np.zeros((X.shape[0]), dtype=np.int32)
			else:
				updateIndices = dist < minDist
				indices[updateIndices] = i
				minDist[updateIndices] = dist[updateIndices]
		return indices, minDist

	def predict(self, X: np.ndarray) -> np.ndarray:
		return self._predict(X, self.cluster_centers_, self.distance)[0]

class KMeans():
	def __init__(self, n_clusters: int, distance = euclDistance):
		self.n_clusters = n_clusters
		self.distance = distance

	def fit(self, X: np.ndarray):
		return KMeansCluster(X, self.n_clusters, self.distance)
