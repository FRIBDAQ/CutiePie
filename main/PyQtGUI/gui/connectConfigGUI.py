import os
import getpass
import time
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import CPyConverter as cpy

class ConnectConfiguration(QWidget):
    def __init__(self, *args, **kwargs):
            super(ConnectConfiguration, self).__init__(*args, **kwargs)

            self.setWindowTitle("Connection configuration")
            self.resize(300, 150);

            try:
                    host = os.environ["RESThost"]
                    port = os.environ["RESTport"]
                    mirrorport = os.environ["MIRRORport"]
            except:
                    os.environ['RESThost'] = "localhost"
                    os.environ["RESTport"] = "<port>"
                    os.environ["MIRRORport"] = "<mirror>"
                    host = os.environ["RESThost"]
                    port = os.environ["RESTport"]
                    mirrorport = os.environ["MIRRORport"] 

            self.serverLabel = QLabel()
            self.serverLabel.setText('Server')
            self.serverLabel.setFixedWidth(50) 
            self.server = QLineEdit()
            self.server.setFixedWidth(200) 
            self.server.setText(host)

            self.userLabel = QLabel("User")
            self.userLabel.setFixedWidth(40) 
            self.user = QLineEdit()                
            self.user.setText(getpass.getuser())
            self.user.setFixedWidth(200) 

            self.restLabel = QLabel()
            self.restLabel.setText('REST port')
            self.restLabel.setFixedWidth(70) 
            self.rest = QLineEdit()
            self.rest.setText(port)
            self.rest.setFixedWidth(80) 

            self.mirrorLabel = QLabel("Mirror port")
            self.mirrorLabel.setFixedWidth(75) 
            self.mirror = QLineEdit()
            self.mirror.setText(mirrorport)
            self.mirror.setFixedWidth(80) 
            
            self.ok = QPushButton("Ok", self)
            self.cancel = QPushButton("Cancel", self)            


            layout = QGridLayout()
            layout.addWidget(self.serverLabel, 1, 1, QtCore.Qt.AlignLeft)
            layout.addWidget(self.server, 1, 2, 1, 1)
            layout.addWidget(self.userLabel, 2, 1, QtCore.Qt.AlignLeft)
            layout.addWidget(self.user, 2, 2, 1, 1)            
            layout.addWidget(self.restLabel, 3, 1, QtCore.Qt.AlignLeft)    
            layout.addWidget(self.rest, 3, 2, 1, 1)    
            layout.addWidget(self.mirrorLabel, 4, 1, QtCore.Qt.AlignLeft)    
            layout.addWidget(self.mirror, 4, 2, 1, 1)    
            lay5 = QHBoxLayout()
            lay5.addWidget(self.ok)            
            lay5.addWidget(self.cancel)
            layout.addLayout(lay5, 5, 1, 1, 2)    

            self.setLayout(layout)

