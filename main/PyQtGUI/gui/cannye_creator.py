#!/usr/bin/env python
# import sys, os
# sys.path.append(os.getcwd())
#
# import algo_factory  # noqa: F401 - keep import so the factory can discover this module
# from CannyEdgePlot import CannyEdgePlot
#
# # Canny edge parameters: sigma and kernel_size control the Gaussian blur that
# # removes noise before the gradient, and the two thresholds decide which edges
# # survive. See the OpenCV Canny documentation for the full description.
#
# class CannyEdgeAlgo:
#     def __init__(self, sigma, kernel_size, lowthreshold, highthreshold, weak_pixel, strong_pixel):
#         self.sigma = sigma
#         self.kernel_size = kernel_size
#         self.lowthreshold = lowthreshold
#         self.highthreshold = highthreshold
#         self.weak_pixel = weak_pixel
#         self.strong_pixel = strong_pixel
#
#         self.cannyEdgePopup = CannyEdgePlot()
#
#     # implementation of the algorithm, the argument are mandatory even if not used
#     def start(self, data, weigths, nclusters, axis, figure=None):
#         xmin, xmax = axis.get_xlim()
#         ymin, ymax = axis.get_ylim()
#
#         #create picture for clustering analysis
#         filename = 'cannyE.jpg'
#         extent = axis.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
#         figure.savefig(filename, bbox_inches=extent.expanded(0.8, 0.9))
#
#         self.cannyEdgePopup.show()
#         self.cannyEdgePopup.setConfig(self.sigma, self.kernel_size, self.lowthreshold, self.highthreshold, self.weak_pixel, self.strong_pixel)
#         self.cannyEdgePopup.plotEdge(filename, xmin, xmax, ymin, ymax)
#
# class CannyEdgeAlgoBuilder:
#     def __call__(self, sigma=1, kernel_size=7, lowthreshold=0.05, highthreshold=0.15, weak_pixel=75, strong_pixel=255, **_ignored):
#         return CannyEdgeAlgo(sigma, kernel_size, lowthreshold, highthreshold, weak_pixel, strong_pixel)
