# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import Counter
import time
from typing import Tuple,List

EPSILON = 1e-4
ZERO_RADIUS = 1e-100

class Ball():
    def __init__(self, center, radius, points, left, right):
        self.center = center      
        self.radius = radius
        self.left = left
        self.right = right
        self.points = points

# calculate Euclidean distance (one-by-one)
def euclDistance_nonbatch(vector1: np.ndarray, vector2: np.ndarray):
	diff = vector2 - vector1
	return np.sqrt(np.sum(np.power(diff, 2)))

# TODO: this class is not thread safe
class BallTree():
    def __init__(self, values: np.ndarray, indices: np.ndarray = None, distance = euclDistance_nonbatch):
        self.data_length = len(values)
        if self.data_length == 0:
            raise Exception('Data For Ball-Tree Must Be Not empty.')

        if indices is None:
            indices = list(range(self.data_length))
        elif indices.shape[0] != len(indices):
            raise Exception('Indices must be one dimentional')
        data = np.column_stack([values, indices])
        self.distance = distance
        self.max_dist = np.inf

        self.root = self._build_tree(data)

    def _build_tree(self, data: np.ndarray):
        if self.data_length == 0:
            return None
        if self.data_length == 1:
            return Ball(data[0, :-1], ZERO_RADIUS, data, None, None)

        data_disloc = np.row_stack([data[1:], data[0]])
        if np.sum(data_disloc - data) == 0:
            return Ball(data[0, :-1], ZERO_RADIUS, data, None, None)

        distance = self.distance
        cur_center = np.mean(data[:, :-1],axis=0)     # Center of the current ball
        dists_with_center = np.array([ distance(cur_center, point) for point in data[:, :-1] ])     # The distance from the current data point to the center of the ball
        # Take the point farthest from the center to prepare the next two subballs, which is also the radius of the current ball
        max_dist_index = np.argmax(dists_with_center)       
        max_dist = dists_with_center[max_dist_index]
        root = Ball(cur_center, max_dist, data, None, None)

        point1 = data[max_dist_index]
        dists_with_point1 = np.array([ distance(point1[:-1], point) for point in data[:, :-1] ])
        max_dist_index2 = np.argmax(dists_with_point1)
        # Take the point farthest from Point1, so as to find the two subballs of the next level
        point2 = data[max_dist_index2]           
        dists_with_point2 = np.array([ distance(point2[:-1], point) for point in data[:, :-1] ])
        
        assign_point1 = dists_with_point1 < dists_with_point2

        root.left = self._build_tree(data[assign_point1])
        root.right = self._build_tree(data[~assign_point1])
        return root    #This is a Ball

    def search(self, target: np.ndarray, K: int = 3):
        if self.root is None:
            raise Exception('Ball-Tree Must Be Not empty.')
        if K > self.data_length:
            raise ValueError("K in KNN Must Be Greater Than Length of data")
        if len(target) != len(self.root.center):
            raise ValueError("Target Must Has Same Dimension With Data")
        search_result = [(None, self.max_dist)]
        self._search(self.root, target, K, search_result)
        return [ (int(node[0][-1]),node[1]) for node in search_result ]
        # print("calu_dist_nums:",self.nums)

    #root is a Ball
    def _search(self, root_ball: Ball, target: np.ndarray, K: int, search_result: List[Tuple[Ball or None, float]]):
        if root_ball is None:
            return
        # Look for closer data points in a qualified hyperspace, which must be a subspace at the last level
        if root_ball.left is None or root_ball.right is None:
            distance = self.distance
            for node in root_ball.points:
                is_duplicate = []
                for item in search_result:
                    if item[0] is not None:
                        dist = distance(node[:-1], item[0][:-1])
                        nodedist = abs(node[-1] - item[0][-1])
                        is_duplicate.append(dist < EPSILON and nodedist < EPSILON)
                if np.array(is_duplicate, np.bool).any():
                    continue
                dist = distance(target, node[:-1])
                if(len(search_result) < K):
                    if search_result[0][0] is None:
                        search_result[0] = (node, dist)
                    else:
                        search_result.append((node, dist))
                        search_result.sort(key=lambda x: x[1])
                elif dist < search_result[-1][1]:
                    del search_result[-1]
                    search_result.insert(0, (node, dist))
                    search_result.sort(key=lambda x: x[1])
        if self.distance(root_ball.center, target) <= root_ball.radius + search_result[0][1]: #or len(search_result) < K
            self._search(root_ball.left, target, K, search_result)
            self._search(root_ball.right, target, K, search_result)
