#!/usr/bin/env python
import sys, os
sys.path.append(os.getcwd())

import cv2
import algo_factory  # noqa: F401 - keep import so the factory can discover this module
from ImgSegPlot import ImgSegPlot

# KMean algo parameters (cv.kmeans())
# samples,             // It should be of np.float32 data type, and each feature should be put in a single column.
# nclusters(K),        // Number of clusters required at end
# criteria,            // It is the iteration termination criteria. When this criteria is satisfied, algorithm iteration stops.
#                      // Actually, it should be a tuple of 3 parameters. They are `( type, max_iter, epsilon )`:
#                      //  1) type of termination criteria. It has 3 flags as below:
#                             cv.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached.
#                             cv.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
#                             cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met.
#                          2) max_iter - An integer specifying maximum number of iterations.
#                          3) epsilon - Required accuracy
# attempts,            // Flag to specify the number of times the algorithm is executed using different initial labellings.
#                         The algorithm returns the labels that yield the best compactness. This compactness is returned as output.
#flags,                // This flag is used to specify how initial centers are taken. Normally two flags are used for this : cv.KMEANS_PP_CENTERS and cv.KMEANS_RANDOM_CENTERS.


class ImgSegAlgo:
    def __init__(self, nclusters, criteria, attempts, flags):
        self.nclusters = nclusters
        self.criteria = criteria
        self.attempts = attempts
        self.flags = flags

        self.imgSegPopup = ImgSegPlot()
        
    # implementation of the algorithm, the argument are mandatory even if not used
    def start(self, data, weigths, nclusters, axis, figure=None):
        xmin, xmax = axis.get_xlim()
        ymin, ymax = axis.get_ylim()

        #create picture for clustering analysis
        filename = 'imgSeg.jpg'
        extent = axis.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
        figure.savefig(filename, bbox_inches=extent.expanded(0.8, 0.9))

        self.imgSegPopup.create_clusterChecks(nclusters)
        self.imgSegPopup.show()
        #set all the parameters
        self.imgSegPopup.setConfig(nclusters, self.criteria, self.attempts, self.flags)
        self.imgSegPopup.plot(filename, nclusters, xmin, xmax, ymin, ymax)
        
class ImgSegAlgoBuilder:
    def __call__(self, nclusters, criteria, attempts, flags, **_ignored):
        return ImgSegAlgo(nclusters=5, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
