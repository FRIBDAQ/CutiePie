#!/usr/bin/env python
import io
import re
import sys, os
cwd = os.getcwd()

#  Sadly it seems that environment variables sent in via setenv
# in the standalone script don't actually get propagate into the
# script so we can't add the ../lib dir to the PYTHONPATH there.
# We figure out where we're installed and add that to the path."

mydir = os.path.dirname(__file__)
sys.path.append(mydir)
libdir = mydir + '/../lib'
sys.path.append(libdir)

# use preprocessor macro __file__ to get the installation directory
# caveat : expects a particular format of installation directory (N.NN-NNNN)
instPath = ""
fileDir = os.path.dirname(os.path.abspath(__file__))
subDirList = fileDir.split("/")
for subDir in subDirList:
    if subDir != "":
        instPath += "/"+subDir
        if re.search(r'\d+\.\d{2}-\d{3}', subDir ):
            # specVersion = subDir
            break


# Check if the USERDIR environment variable is set.
if "USERDIR" in os.environ:
    sys.path.append(os.environ["USERDIR"])
    print("The .py user-based files are in ",os.environ["USERDIR"])
# Check if the user-based files are in the current working directory.
elif os.path.exists(os.path.join(cwd, "fit_skel_creator.py")) and os.path.exists(os.path.join(cwd, "algo_skel_creator.py")):
    print("The .py user-based files are in the current working directory.")
    sys.path.append(cwd)

#Script is append after USERDIR or cwd so if those are defined the priority is to them not to Script
sys.path.append(instPath + "/Script")

from PyQt5 import QtCore
from PyQt5.QtWidgets import *

import cv2

# GUI graphical skeleton
from GUI import MainWindow

import fit_factory
# skeleton for user-based FIT implementation
import fit_skel_creator
# already implemented examples
import fit_gaus_creator
import fit_exp_creator
import fit_p1_creator
import fit_p2_creator
import fit_gp1_creator
import fit_gp2_creator

import algo_factory
# skeleton for user-based ML implementation
import algo_skel_creator
# already implemented examples
import kmean_creator
import gmm_creator
import imgseg_creator
import cannye_creator

# print("Check import : ",fit_skel_creator.__file__)

#######################################
##  Fitting
#######################################
fitfactory = fit_factory.FitFactory()
# Configurable parameters for Skel (for the full param list, please see skel_creator.py)
config_fit_skel = {
    'param_1': 1,
    'param_2': 1,
    'param_3': 10
}

# Configurable parameters for Gaussian fit - NB technically they are not needed as the initial values are extracted from the plot itself
config_fit_gaus = {
    'amplitude': 1000,
    'mean': 100,
    'standard_deviation': 10
}

# Configurable parameters for Exponential fit
config_fit_exp = {
    'a': 1,
    'b': 5,
    'c': -1
}

# Configurable parameters for 1st order Polynomial fit
config_fit_p1 = {
    'p0': 100,
    'p1': 10
}

# Configurable parameters for 2nd order Polynomial fit
config_fit_p2 = {
    'p0': 100,
    'p1': 10,
    'p2': 1,
}

# Configurable parameters for Gaussian + 1st order Polynomial fit - NB technically they are not needed as the initial values are extracted from the plot itself
config_fit_gp1 = {
    'amplitude': 1000,
    'mean': 100,
    'standard_deviation': 10,
    'p0': 100,
    'p1': 10,
    'f': 0.9
}

# Configurable parameters for Gaussian + 2st order Polynomial fit - NB technically they are not needed as the initial values are extracted from the plot itself
config_fit_gp2 = {
    'amplitude': 1000,
    'mean': 100,
    'standard_deviation': 10,
    'p0': 100,
    'p1': 10,
    'p2': 1,
    'f': 0.9
}


#######################################
##  ML
#######################################
algofactory = algo_factory.AlgoFactory()
# Configurable parameters for Skel (for the full param list, please see skel_creator.py)
config_algo_skel = {
    'param_1': 5,
    'param_2': 'test',
    'param_3': 100
}

# Configurable parameters for KMean (for the full param list, please see kmean_creator.py)
config_algo_kmean = {
    'n_clusters': 3,
    'n_init': 10
}

# Configurable parameters for Gaussian Mixture Model (for the full param list, please see gmm_creator.py)
config_algo_gmm = {
    'n_components': 4,
}
'''
# Configurable parameters for Image Segmentation (for the full param list, please see imgseg_creator.py)
config_algo_img = {
    'nclusters': 5,
    'criteria' : (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
    'attempts' :10,
    'flags': cv2.KMEANS_RANDOM_CENTERS
}

# Configurable parameters for Canny Edge (for the full param list, please see cannye_creator.py)
config_algo_canny = {
    'sigma': 1,
    'kernel_size': 7,
    'lowthreshold': 0.05,
    'highthreshold': 0.15,
    'weak_pixel': 75,
    'strong_pixel': 255
}
'''
# Fitting function registration
fitfactory.register_builder('Skeleton', fit_skel_creator.SkelFitBuilder(), config_fit_skel)
fitfactory.register_builder('Gauss', fit_gaus_creator.GausFitBuilder(), config_fit_gaus)
fitfactory.register_builder('Exp', fit_exp_creator.ExpFitBuilder(), config_fit_exp)
fitfactory.register_builder('Pol1', fit_p1_creator.Pol1FitBuilder(), config_fit_p1)
fitfactory.register_builder('Pol2', fit_p2_creator.Pol2FitBuilder(), config_fit_p2)
fitfactory.register_builder('G+Pol1', fit_gp1_creator.GPol1FitBuilder(), config_fit_gp1)
fitfactory.register_builder('G+Pol2', fit_gp2_creator.GPol2FitBuilder(), config_fit_gp2)

# ML Algorithm registration
algofactory.register_builder('Skeleton', algo_skel_creator.SkelAlgoBuilder(), config_algo_skel)
algofactory.register_builder('KMean', kmean_creator.KMeanAlgoBuilder(), config_algo_kmean)
algofactory.register_builder('Gaussian MM', gmm_creator.GMMAlgoBuilder(), config_algo_gmm)
#algofactory.register_builder('Image Segmentation', imgseg_creator.ImgSegAlgoBuilder(), config_algo_img)
#algofactory.register_builder('Canny Edge', cannye_creator.CannyEdgeAlgoBuilder(), config_algo_canny)

app = QApplication(sys.argv)
gui = MainWindow(algofactory, fitfactory)
gui.show()
sys.exit(app.exec_())

