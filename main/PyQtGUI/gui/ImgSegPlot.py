import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PyQt5.QtWidgets import (QCheckBox, QDialog, QGridLayout, QGroupBox, QPushButton, QVBoxLayout)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import cv2

class ImgSegPlot(QDialog):
    def __init__(self, *args, **kwargs):
        super(ImgSegPlot, self).__init__(*args, **kwargs)

        self.nclust = 0
        self.criteria = 0
        self.attempt = 0
        self.flags = 0
        self.isChecked = {}

        # a figure instance to plot on
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to `Update` method
        self.button = QPushButton('Update')
        self.button.clicked.connect(self.updateClusterImg)

        # set the layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def loadFigure(self, filename):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def setConfig(self, clusters, criteria, attempt, flags):
        self.nclust = clusters
        self.criteria = criteria
        self.attempt = attempt
        self.flags = flags
    
    def plot(self, filename, nclusters, xmin, xmax, ymin, ymax):

        self.image = self.loadFigure(filename)

        # reshape the image to a 2D array of pixels and 3 color values (RGB)
        pixel_values = self.image.reshape((-1, 3))
        # convert to float
        pixel_values = np.float32(pixel_values)

        # number of clusters (K)
        _, self.labels, (centers) = cv2.kmeans(pixel_values, nclusters, None, self.criteria, self.attempt, self.flags)

        # convert back to 8 bit values
        centers = np.uint8(centers)

        # flatten the labels array
        self.labels = self.labels.flatten()

        # convert all pixels to the color of the centroids
        segmented_image = centers[self.labels.flatten()]

        # reshape back to the original image dimension
        segmented_image = segmented_image.reshape(self.image.shape)
        # create an axis
        self.ax = self.figure.add_subplot(111)
        # discards the old graph
        self.ax.clear()
        self.ax.imshow(segmented_image, aspect='auto')
        #self.ax.set_xlim(xmin,xmax)
        #self.ax.set_ylim(ymin,ymax)        
        
        # refresh canvas
        self.canvas.draw_idle()

    def create_clusterChecks(self, nclusters):
        self.pCheck = QGroupBox("Cluster Selection")

        self.nclust = nclusters
        row = 0
        col = 0
        counter = 0
        self.clust_cbox = []
        deflayout = QGridLayout()
        for i in range(nclusters):
            title = "Cluster "+str(i+1)
            self.clust_cbox.append(QCheckBox(title,self))
            if (i%10 == 0) and i != 0:
                row += 1
                col = 0
                counter += 1
            else:
                col = i-counter*10
            deflayout.addWidget(self.clust_cbox[i], row, col)

        self.pCheck.setLayout(deflayout)
        self.layout.addWidget(self.pCheck)

        # create signals for checkbox
        self.create_cluster_signals()
        
    def create_cluster_signals(self):
        print("inside create_cluster_signals for ",self.nclust, " clusters")
        for i in range(self.nclust):
            self.isChecked[i] = False
            self.clust_cbox[i].stateChanged.connect(self.filterCluster)
            self.clust_cbox[i].setChecked(True)

    def filterCluster(self):
        for i, btn in enumerate(self.clust_cbox):
            if btn.isChecked() == False:
                try:
                    self.isChecked[i] = False
                except Exception:
                    pass
            else:
                if self.isChecked[i] == False:
                    self.isChecked[i] = True

    def updateClusterImg(self):
        # disable only the cluster number 2 (turn the pixel into black)
        masked_image = np.copy(self.image)
        # convert to the shape of a vector of pixel values
        masked_image = masked_image.reshape((-1, 3))
        index_list = [i for i in range(len(self.isChecked)) if self.isChecked[i] == False]
        for i in index_list:
            # masked with white
            masked_image[self.labels == i] = [255, 255, 255]
            
        # convert back to original shape
        masked_image = masked_image.reshape(self.image.shape)
        # show the image
        self.ax.clear()
        self.ax.imshow(masked_image, aspect='auto')

        # refresh canvas
        self.canvas.draw()
        
