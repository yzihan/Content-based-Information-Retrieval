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
from torchvision.models import densenet161
import torchvision.transforms as transforms

NET_NAME='densenet161'

DIR = 'metadata'


class DenseNetPrediction(object):
    def __init__(self, query = None):
        self.name = os.path.join(DIR, 'densenet_descriptors.pickle')
        self.query_path = query
        self.dense_net = densenet161(pretrained=True)
        self.dense_net.eval()
        self.normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                              std = [0.229, 0.224, 0.225])

    def describe(self, image, nfeatures=1000):
        # Calculate descriptor using densenet

        image = torch.FloatTensor(image).permute(2, 0, 1)

        image = self.normalize(image)

        with torch.no_grad():
            vec = self.dense_net(image.unsqueeze(dim=0))

        return vec.view(-1).numpy()

    def get_descriptors(self, dataset):
        '''
        :param dataset: Object of DataSet

        Calculate set of descriptors of all images in dataset
        '''
        descriptors = []

        # Get DataFrame with images paths
        df_data = dataset.get_data()

        print('Start computing descriptors using ' + NET_NAME + '. ', end='', flush=True)
        for image_temp in df_data.itertuples():
            im = cv2.imread(image_temp.image_path)
            des = self.describe(im)
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
        return self.describe(im)

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
            descriptors = self.get_descriptors(dataset)

        if tests is not None:
            return descriptors, tests
        else:
            return descriptors
