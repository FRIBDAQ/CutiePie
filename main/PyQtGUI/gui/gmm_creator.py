#!/usr/bin/env python
import sys, os
sys.path.append(os.getcwd())

import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

import algo_factory  # noqa: F401 - keep import so the factory can discover this module
        
# GMM algo parameters (sklearn.mixture.GaussianMixture)
# n_components=1,        // The number of mixture components.
# covariance_type='full' // The number of mixture components. Options are: {‘full’ (default), ‘tied’, ‘diag’, ‘spherical’}.
# tol=0.001              // The convergence threshold. EM iterations will stop when the lower bound average gain is below this threshold.
# reg_covar=1e-06        // Non-negative regularization added to the diagonal of covariance. Allows to assure that the covariance matrices are all positive.
# max_iter=100           // The number of EM iterations to perform.
# n_init=1               // The number of initializations to perform. The best results are kept.
# init_params='kmeans'   // The method used to initialize the weights, the means and the precisions. Must be one of:
#                           1) 'kmeans' : responsibilities are initialized using kmeans
#                           2) 'random' : responsibilities are initialized randomly.
# weights_init=None      // The user-provided initial weights, defaults to None. If it None, weights are initialized using the init_params method.
# means_init=None        // The user-provided initial means, defaults to None, If it None, means are initialized using the init_params method.
# precisions_init=None   // The user-provided initial precisions (inverse of the covariance matrices), defaults to None.
#                           If it None, precisions are initialized using the ‘init_params’ method. The shape depends on ‘covariance_type’:
#                           (n_components,)                        if 'spherical',
#                           (n_features, n_features)               if 'tied',
#                           (n_components, n_features)             if 'diag',
#                           (n_components, n_features, n_features) if 'full'
# random_state=None      // Controls the random seed given to the method chosen to initialize the parameters (see init_params).
#                           In addition, it controls the generation of random samples from the fitted distribution (see the method sample).
#                           Pass an int for reproducible output across multiple function calls.
# warm_start=False       // If ‘warm_start’ is True, the solution of the last fitting is used as initialization for the next call of fit().
#                           This can speed up convergence when fit is called several times on similar problems.
#                           In that case, ‘n_init’ is ignored and only a single initialization occurs upon the first call.
# verbose=0              // Enable verbose output. If 1 then it prints the current initialization and each iteration step.
#                           If greater than 1 then it prints also the log probability and the time needed for each step.
# verbose_interval=10    // Number of iteration done before the next print.

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

