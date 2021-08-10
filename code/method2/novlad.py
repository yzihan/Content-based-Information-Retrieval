import os
import cv2
import itertools
import numpy as np
# from kmeans import KMeans
from sklearn.cluster import KMeans
from collections import Counter
from six.moves import cPickle
import pandas as pd

import torch
from pytorch_sift import SIFTNet

DIR = 'metadata'


class SiftDescriptorPrediction(object):
    def __init__(self, query = None):
        self.name = os.path.join(DIR, 'sift_descriptors.pickle')
        self.query_path = query
        self.sift_net = SIFTNet(256)

    def describe_SIFT(self, image, nfeatures=1000):
        '''
        :param image: image path
        :param nfeatures: Number of key-points

        Calculate SIFT descriptor with nfeatures key-points
        '''

        # Converting from BGR to GRAY
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Scale to 256x256
        image = cv2.resize(image, (256, 256))

        with torch.no_grad():
            vec = self.sift_net(torch.tensor(image.astype(np.float32)).view(1,1,256,256))

        return vec.view(128).numpy()

    def get_SIFT_descriptors(self, dataset):
        '''
        :param dataset: Object of DataSet

        Calculate set of SIFT descriptors of all images in dataset
        '''
        descriptors = []

        # Get DataFrame with images paths
        df_data = dataset.get_data()

        # Compute SIFT descriptors
        print('Start computing descriptors using SIFT. ', end='', flush=True)
        for image_temp in df_data.itertuples():
            im = cv2.imread(image_temp.image_path)
            des = self.describe_SIFT(im)
            descriptors.append({'image_path': image_temp.image_path,
                                'class_image': image_temp.class_image,
                                'feature_vector': des
                                })

        cPickle.dump(descriptors, open(self.name, "wb", True))

        print('Complete.')
        return descriptors

    def prepare_query(self, query=None):
        if query is None:
            query = self.query_path
        
        if isinstance(query, pd.DataFrame):
            return [ self.prepare_query(q) for q in query['image_path'] ]
        
        if isinstance(query, list):
            return [ self.prepare_query(q) for q in query ]

        im = cv2.imread(query)
        return self.describe_SIFT(im)

    def get_prediction(self, train_descriptors, distance, prepared_query=None, mode=0, accelerator=None, nearby_count = 3):
        '''
        :param kmeans_clusters: Object of Kmeans (sklearn)
        :param train_descriptors: Set of descriptions

        Calculate VLAD vector for query image, and then
        get best distances in dataset.
        '''
        list_res = []
        if prepared_query is None:
            prepared_query = self.prepare_query()

        if isinstance(prepared_query, list):
            return [ self.get_prediction(train_descriptors, distance, query, mode, accelerator, nearby_count) for query in prepared_query ]

        # compute descriptor for query
        v = prepared_query

        if accelerator is None:
            # brute force
            # Get distances between query VLAD and dataset VLADs descriptors
            for i in range(len(train_descriptors)):
                temp_vec = train_descriptors[i]['feature_vector']
                if distance == 'L1':
                    dist = np.linalg.norm((temp_vec - v), ord=1)
                else:
                    dist = np.linalg.norm(temp_vec - v)
                # list_res.append({'i': i,
                #                 'dist': dist,
                #                 'class': train_descriptors[i]['class_image'],
                #                 'image_path': train_descriptors[i]['image_path']
                #                 })
                list_res.append((i, dist))
            res_ = sorted(list_res, key=lambda x: x[1])
            res_top = res_[:nearby_count]
        else:
            res_top = accelerator.search(v, nearby_count)

        if mode == 2:
            return res_top

        # Get most frequent class in (nearby_count) first classes
        res_count = Counter([ train_descriptors[x[0]]['class_image'] for x in res_top ])
        res = min(res_count.items(), key=lambda x: (-x[1], x[0]))[0]

        if mode == 1:
            return res

        print('\nPredicted class for query image: {}.'.format(res))

    def get_clusters_descriptors(self, dataset, k=64, test_split = None, test_pos = 0):
        '''
        :param dataset: main dataset
        :param k: number os clusters to determine (default=64)

        Load computed clusters and vectors or compute SIFT descriptors, then
        compute clusters for these descriptors, then calculate VLAD descriptors
        and save.
        '''
        # if we have computed clusters and vlad vectors
        if os.path.exists(self.name) and test_split is None:
            descriptors = cPickle.load(open(self.name, "rb", True))
            tests = None
        else:
            if test_split is not None:
                dataset, tests = dataset.split(test_split, test_pos)
            else:
                tests = None
            descriptors = self.get_SIFT_descriptors(dataset)

        if tests is not None:
            return descriptors, tests
        else:
            return descriptors
