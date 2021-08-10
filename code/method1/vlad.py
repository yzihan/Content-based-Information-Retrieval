import os
import cv2
import itertools
import numpy as np
# from kmeans import KMeans
from sklearn.cluster import KMeans
from collections import Counter
from six.moves import cPickle
import pandas as pd
import pysift

DIR = 'metadata'


class VladPrediction(object):
    def __init__(self, query = None):
        self.name = os.path.join(DIR, 'clusters_and_vlad_descriptors.pickle')
        self.query_path = query

    def describe_SIFT(self, image, nfeatures=1000):
        '''
        :param image: image path
        :param nfeatures: Number of key-points

        Calculate SIFT descriptor with nfeatures key-points
        '''

        # Converting from BGR to GRAY
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # # Compute key-points and descriptors for each key-points
        # # nfeatures = number of key-points for each image
        # sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
        # _keypoints, descriptors = sift.detectAndCompute(image, None)
        _keypoints, descriptors = pysift.computeKeypointsAndDescriptors(image)
        return descriptors

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
            if os.path.exists(image_temp.image_path + '.sift'):
                des = cPickle.load(open(image_temp.image_path + '.sift', "rb", True))
            else:
                im = cv2.imread(image_temp.image_path)
                des = self.describe_SIFT(im)
                cPickle.dump(des, open(image_temp.image_path + '.sift', "wb", True))
            if des.all() != None:
                descriptors.append(des)

        # Union all descriptors for each key-point to list
        descriptors = np.array(
            list(itertools.chain.from_iterable(descriptors)))
        cPickle.dump(descriptors, open(os.path.join(
            'metadata', 'sift_vectors_{}'.format(os.path.basename(dataset.dataset_dir))), "wb", True))
        print('Complete.')
        return descriptors

    def get_clusters(self, descriptors, k):
        '''
        :param descriptors: Set of all SIFT descriptors in dataset
        :param k: number of clusters to compute

        Get k number of clusters
        '''
        print('Start clustering descriptors. ', end='', flush=True)

        # Compute k clusters
        clusters = KMeans(n_clusters=k).fit(descriptors)
        print('Complete.')
        return clusters

    def compute_vlad_descriptor(self, descriptors, kmeans_clusters):
        '''
        :param descriptor: SIFT descriptor of image
        :param kmeans_clusters: Object of Kmeans (sklearn)

        First we need to predict clusters fot key-points of image (row in
        input descriptor). Then for each cluster we get descriptors, which belong to it,
        and calculate sum of residuals between descriptor and centroid (cluster center)
        '''
        # Get SIFT dimension (default: 128)
        sift_dim = descriptors.shape[1]

        # Predict clusters for each key-point of image
        labels_pred = kmeans_clusters.predict(descriptors)

        # Get centers fot each cluster and number of clusters
        centers_cluster = kmeans_clusters.cluster_centers_
        numb_cluster = kmeans_clusters.n_clusters
        vlad_descriptors = np.zeros([numb_cluster, sift_dim])

        # Compute the sum of residuals (for belonging x for cluster) for each cluster
        for i in range(numb_cluster):
            if np.sum(labels_pred == i) > 0:

                # Get descritors which belongs to cluster and compute residuals between x and centroids
                x_belongs_cluster = descriptors[labels_pred == i, :]
                vlad_descriptors[i] = np.sum(
                    x_belongs_cluster - centers_cluster[i], axis=0)

        # Create vector from matrix
        vlad_descriptors = vlad_descriptors.flatten()

        # Power and L2 normalization
        vlad_descriptors = np.sign(vlad_descriptors) * \
            (np.abs(vlad_descriptors)**(0.5))
        vlad_descriptors = vlad_descriptors / \
            np.sqrt(vlad_descriptors @ vlad_descriptors)
        return vlad_descriptors

    def get_vlad_descriptors(self, kmeans_clusters, dataset):
        '''
        :param kmeans_clusters: Object of Kmeans (sklearn)
        :param dataset: Object of DataSet

        Calculate VLAD descriptors for dataset
        '''
        vlad_descriptors = []

        # Get DataFrame with paths classes fro images
        df_data = dataset.get_data()

        print('Start computing vlad vectors. ', end='', flush=True)
        for image_temp in df_data.itertuples():

            # Compute SIFT descriptors
            im = cv2.imread(image_temp.image_path)
            
            if os.path.exists(image_temp.image_path + '.sift'):
                descriptor = cPickle.load(open(image_temp.image_path + '.sift', "rb", True))
            else:
                im = cv2.imread(image_temp.image_path)
                descriptor = self.describe_SIFT(im)
                cPickle.dump(des, open(image_temp.image_path + '.sift', "wb", True))

            if descriptor.all() != None:

                # Compute VLAD descriptors
                vlad_descriptor = self.compute_vlad_descriptor(
                    descriptor, kmeans_clusters)
                vlad_descriptors.append({'image_path': image_temp.image_path,
                                         'class_image': image_temp.class_image,
                                         'feature_vector': vlad_descriptor
                                         })
        print('Complete.')
        return vlad_descriptors

    def prepare_query(self, query=None):
        if query is None:
            query = self.query_path
        
        if isinstance(query, pd.DataFrame):
            return [ self.prepare_query(q) for q in query['image_path'] ]
        
        if isinstance(query, list):
            return [ self.prepare_query(q) for q in query ]

        if os.path.exists(query + '.sift'):
            des = cPickle.load(open(query + '.sift', "rb", True))
        else:
            im = cv2.imread(query)
            des = self.describe_SIFT(im)
            cPickle.dump(des, open(query + '.sift', "wb", True))

        return des

    def get_prediction(self, kmeans_clusters, vlad_descriptors, distance, prepared_query=None, mode=0, accelerator=None, nearby_count = 3):
        '''
        :param kmeans_clusters: Object of Kmeans (sklearn)
        :param vlad_descriptors: Set of VLAD descriptions

        Calculate VLAD vector for query image, and then
        get best distances in dataset.
        '''
        list_res = []
        if prepared_query is None:
            prepared_query = self.prepare_query()
        descriptor = prepared_query

        if isinstance(descriptor, list):
            return [ self.get_prediction(kmeans_clusters, vlad_descriptors, distance, des, mode, accelerator, nearby_count) for des in descriptor ]

        # compute VLAD descriptor for query
        v = self.compute_vlad_descriptor(descriptor, kmeans_clusters)

        if accelerator is None:
            # brute force
            # Get distances between query VLAD and dataset VLADs descriptors
            for i in range(len(vlad_descriptors)):
                temp_vec = vlad_descriptors[i]['feature_vector']
                if distance == 'L1':
                    dist = np.linalg.norm((temp_vec - v), ord=1)
                else:
                    dist = np.linalg.norm(temp_vec - v)
                # list_res.append({'i': i,
                #                 'dist': dist,
                #                 'class': vlad_descriptors[i]['class_image'],
                #                 'image_path': vlad_descriptors[i]['image_path']
                #                 })
                list_res.append((i, dist))
            res_ = sorted(list_res, key=lambda x: x[1])
            res_top = res_[:nearby_count]
        else:
            res_top = accelerator.search(v, nearby_count)

        if mode == 2:
            return res_top

        # Get most frequent class in (nearby_count) first classes
        res_count = Counter([ vlad_descriptors[x[0]]['class_image'] for x in res_top ])
        res = min(res_count.items(), key=lambda x: (-x[1], x[0]))[0]

        if mode == 1:
            return res

        print('\nPredicted class for query image: {}.'.format(res))

    def get_clusters_vlad_descriptors(self, dataset, k=64, test_split = None, test_pos = 0):
        '''
        :param dataset: main dataset
        :param k: number os clusters to determine (default=64)

        Load computed clusters and vectors or compute SIFT descriptors, then
        compute clusters for these descriptors, then calculate VLAD descriptors
        and save.
        '''
        # if we have computed clusters and vlad vectors
        if os.path.exists(self.name) and test_split is None:
            with open(self.name, 'rb') as file:
                kmeans_clusters, vlad_descriptors = cPickle.load(file)
            tests = None
        else:
            if test_split is not None:
                dataset, tests = dataset.split(test_split, test_pos)
            else:
                tests = None

            # Get list of all SIFT descriptors
            descriptors = self.get_SIFT_descriptors(dataset)

            # Get Kmeans object with k clusters
            kmeans_clusters = self.get_clusters(descriptors, k)

            # Compute VLAD descriptors
            vlad_descriptors = self.get_vlad_descriptors(
                kmeans_clusters, dataset)

            if test_split is None:
                # Save results
                with open(self.name, 'wb') as file:
                    cPickle.dump([kmeans_clusters, vlad_descriptors], file)
        if tests is not None:
            return kmeans_clusters, vlad_descriptors, tests
        else:
            return kmeans_clusters, vlad_descriptors
