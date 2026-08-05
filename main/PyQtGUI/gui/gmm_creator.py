#!/usr/bin/env python
import sys, os
sys.path.append(os.getcwd())

import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

import algo_factory  # noqa: F401 - keep import so the factory can discover this module
        
# Gaussian-mixture parameters, passed straight to
# sklearn.mixture.GaussianMixture; see its documentation for the full set.

class GMMAlgo:
    def __init__(self, n_components, covariance_type, tol, reg_covar, max_iter, n_init, init_params, weights_init, means_init,
                 precisions_init, random_state, warm_start, verbose, verbose_interval):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    # implementation of the algorithm, the argument are mandatory even if not used
    def start(self, data, weigths, nclusters, axis, figure=None):
        model = GaussianMixture(n_components=nclusters)

        # fit GM object to data
        model.fit(data)
        cluster_center = model.means_

        # draw ellipses
        self.addEllipse(axis, model.means_, model.covariances_, model.weights_)

        print("###################################################")
        print("# Results of Gaussian Mixture clustering analysis #")
        print("###################################################")
        for i in range(len(cluster_center)):
            print("Cluster",i,"with center (x,y)=(",cluster_center[i][0],",",cluster_center[i][1],")")
        print("###################################################")
        

    def addEllipse(self, axis, mean, cov, weight):
        w_factor = 0.2 / weight.max()
        for pos, covar, w in zip(mean, cov, weight):
            self.draw_ellipse(pos, covar, axis, color="red", alpha=w * w_factor)

    def draw_ellipse(self, position, covariance, axis, **kwargs):
        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 4):
            axis.add_patch(Ellipse(position, nsig * width, nsig * height,
                                   angle, **kwargs))

        
class GMMAlgoBuilder:
    def __call__(self, n_components=1, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10, **_ignored):
        return GMMAlgo(n_components, covariance_type, tol, reg_covar, max_iter, n_init, init_params, weights_init, means_init,
                       precisions_init, random_state, warm_start, verbose, verbose_interval)

