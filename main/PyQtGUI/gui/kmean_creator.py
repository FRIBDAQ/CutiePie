#!/usr/bin/env python
import sys, os
sys.path.append(os.getcwd())

import numpy as np
from sklearn.cluster import KMeans
from itertools import compress
from matplotlib.patches import Circle

import algo_factory  # noqa: F401 - keep import so the factory can discover this module

# KMean algo parameters (sklearn.cluster.KMeans)
# n_clusters=8,                             // The number of clusters to form as well as the number of centroids to generate.
# init='k-means++',                         // Method for initialization: ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
# See section Notes in k_init for more details. ‘random’: choose n_clusters
# observations (rows) at random from data for the initial centroids.
# n_init=10,                                // Number of time the k-means algorithm will be run with different centroid seeds.
#                                              The final results will be the best output of n_init consecutive runs in terms of inertia.
# max_iter=300,                             // Maximum number of iterations of the k-means algorithm for a single run.
# tol=0.0001,                               // Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
#                                              It’s not advised to set tol=0 since convergence might never be declared due to rounding errors. Use a very small number instead.
# precompute_distances='auto', DEPRECATED   // Precompute distances (faster but takes more memory).
# ‘auto’: do not precompute distances if n_samples * n_clusters > 12 million.
# This corresponds to about 100MB overhead per job using double precision.
# verbose=0,                                // Verbosity mode.
# random_state=None,                        // Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
# copy_x=True,                              // When pre-computing distances it is more numerically accurate to center the data first. If copy_x is True (default), then the original data is not modified.
# If False, the original data is modified, and put back before the function
# returns, but small numerical differences may be introduced by subtracting
# and then adding the data mean. Note that if the original data is not
# C-contiguous, a copy will be made even if copy_x is False.
# n_jobs='None',   DEPRECATED               // The number of OpenMP threads to use for the computation. Parallelism is sample-wise on the main cython loop which assigns each sample to its closest center.
# algorithm='auto'                          // K-means algorithm to use. The classical EM-style algorithm is “full”.
# The “elkan” variation is more efficient on data with well-defined clusters,
# by using the triangle inequality. However it’s more memory intensive due to
# the allocation of an extra array of shape (n_samples, n_clusters).

class KMeanAlgo:
    def __init__(self, n_clusters, init, n_init, max_iter, tol, precompute_distances, verbose,
                 random_state, algorithm):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.verbose = verbose
        self.random_state = random_state
        self.algorithm = algorithm

    # implementation of the algorithm, the argument are mandatory even if not used
    def start(self, data, weigths, nclusters, axis, figure=None):
        # create kmeans object
        kmeans = KMeans(nclusters, self.init, self.n_init, self.max_iter, self.tol, self.precompute_distances, self.verbose, self.random_state, self.algorithm)
        # fit kmeans object to data
        kmeans.fit(data, sample_weight=weigths)

        cluster_center = kmeans.cluster_centers_        
        confidPerc = self.soft_clustering_weights(data,cluster_center)

        prob_dict = {}
        for i in range(len(confidPerc[0])):
            prob_dict[i] = self.extract(confidPerc, i)

        # CL 90%/CL 95%
        bool_dict_90 = {}
        bool_dict_95 = {}
        w_90 = {}
        w_95 = {}
        sum90 = 0
        sum95 = 0
        for i in range(len(cluster_center)):
            self.addPoint(axis, [cluster_center[i][0], cluster_center[i][1]])
            bool_dict_90[i] = [True if x>0.9 else False for i, x in enumerate(prob_dict[i])]
            bool_dict_95[i] = [True if x>0.95 else False for i, x in enumerate(prob_dict[i])]
            w_90[i] = list(compress(weigths, bool_dict_90[i]))
            w_95[i] = list(compress(weigths, bool_dict_95[i]))
            sum90 += sum(w_90[i])
            sum95 += sum(w_95[i])

        print("#########################################")
        print("# Results of K-Mean clustering analysis #")
        print("#########################################")
        for i in range(len(cluster_center)):
            print("Cluster", i," with center (x,y)=(",cluster_center[i][0],",",cluster_center[i][1],")")
        print("Confidence Level 90% -->", 100*sum90/sum(weigths),"%")
        print("Confidence Level 95% -->", 100*sum95/sum(weigths),"%")
        print("#########################################")

    def extract(self, lst, index):
        return [item[index] for item in lst]

    def addPoint(self, axis, new_point):
        patch = Circle((new_point[0], new_point[1]), radius=10, color="red")
        axis.add_patch(patch)
        
    # soft clustering confidence method
    def soft_clustering_weights(self, data, cluster_centers, **kwargs):
        """Function to calculate the weights from soft k-means data: Array of
        data. shape = N x F, for N data points and F Features cluster_centers:
        Array of cluster centres."""
        # Fuzziness parameter m>=1. Where m=1 => hard segmentation
        m = 2
        if 'm' in kwargs:
            m = kwargs['m']

        Nclusters = cluster_centers.shape[0]
        Ndp = len(data)

        # Get distances from the cluster centres for each data point and each cluster
        EuclidDist = np.zeros((Ndp, Nclusters))
        for i in range(Nclusters):
            EuclidDist[:,i] = np.sum((data-np.matlib.repmat(cluster_centers[i], Ndp, 1))**2,axis=1)

        # Denominator of the weight from wikipedia:
        invWeight = EuclidDist**(2/(m-1))*np.matlib.repmat(np.sum((1./EuclidDist)**(2/(m-1)),axis=1).reshape(-1,1),1,Nclusters)
        Weight = 1./invWeight

        return Weight

        
class KMeanAlgoBuilder:
    def __call__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0,
                 random_state=None, algorithm='auto', **_ignored):
        return KMeanAlgo(n_clusters, init, n_init, max_iter, tol, precompute_distances, verbose, random_state, algorithm)
