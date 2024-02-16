import os
import getpass
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class Configuration(QGridLayout):
    def __init__(self, *args, **kwargs):
            super(Configuration, self).__init__(*args, **kwargs)


# -------- beginning of what was previously in MenuGUI -----------
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
            self.serverLabel.setText('   Server')
            self.serverLabel.setFixedWidth(50) 
            self.server = QLineEdit()
            self.server.setFixedWidth(180) 
            self.server.setText(host)

            self.restLabel = QLabel()
            self.restLabel.setText('   REST port')
            self.restLabel.setFixedWidth(70) 
            self.rest = QLineEdit()
            self.rest.setText(port)
            self.rest.setFixedWidth(80) 
            
            self.userLabel = QLabel("   User")
            self.userLabel.setFixedWidth(40) 
            self.user = QLineEdit()                
            self.user.setText(getpass.getuser())
            self.user.setFixedWidth(180) 

            self.mirrorLabel = QLabel("   Mirror port")
            self.mirrorLabel.setFixedWidth(75) 

            self.mirror = QLineEdit()
            self.mirror.setText(mirrorport)
            self.mirror.setFixedWidth(80) 

            self.exitButton = QPushButton("Exit")
            self.exitButton.setStyleSheet("background-color:rgb(252, 48, 3);")
            self.exitButton.setFixedWidth(85) 

            self.connectButton = QPushButton("Connect")
            self.connectButton.setStyleSheet("background-color:#7ec0ee;")
            self.connectButton.setFixedWidth(85) 

            self.saveButton = QPushButton("Save Geometry")
            self.saveButton.setStyleSheet("background-color:#ffd700;")
            self.saveButton.setFixedWidth(100) 

            self.loadButton = QPushButton("Load Geometry")
            self.loadButton.setStyleSheet("background-color:#ffd700;")
            self.loadButton.setFixedWidth(100) 
# -------- end of what was previously in MenuGUI -----------


            self.isDrag = False
            self.isEdit = False            

            self.histo_list_label = QLabel("   Spectrum")
            self.histo_list_label.setFixedWidth(70)
            self.histo_list = QComboBox()
            self.histo_list.setFixedWidth(200)

            self.histo_geo_label = QLabel("Geometry")
            self.histo_geo_label.setFixedWidth(63)
            self.histo_geo_row = QComboBox()
            self.histo_geo_row.setFixedWidth(50)
            self.histo_geo_col = QComboBox()
            self.histo_geo_col.setFixedWidth(50)

            self.histo_geo_add = QPushButton("Add")
            self.histo_geo_add.setFixedWidth(85)
            self.histo_geo_add.setStyleSheet("background-color:#bcee68;")

            self.histo_geo_update = QPushButton("Update")
            self.histo_geo_update.setFixedWidth(85)
            self.histo_geo_update.setStyleSheet("background-color:#bcee68;")
             
            self.button1D = QRadioButton("1D")
            # self.button1D.setFixedWidth(40)
            self.button2D = QRadioButton("2D")
            # self.button2D.setFixedWidth(40)                        
            self.button2D_option = QComboBox()
            self.button2D_option.addItem("Light")
            self.button2D_option.addItem("Dark")

            #Gate menu
            # self.gate_list_label = QLabel("   Gate")            
            # self.gate_list_label.setFixedWidth(40)            
            # self.listGate = QComboBox()
            # self.listGate.setFixedWidth(125) 
            # self.listGate_type_label = QLabel("  ")
            # self.listGate_type_label.setFixedWidth(20)
            # self.listGate_type = ["s", "c", "b", "gs", "gc", "gb"]

            # self.listGate_type.addItem("s")
            # self.listGate_type.addItem("c")
            # self.listGate_type.addItem("b")
            # self.listGate_type.addItem("gs")
            # self.listGate_type.addItem("gc")
            # self.listGate_type.addItem("gb")
            # self.listGate_type.setCurrentIndex(-1)

            self.createGate = QPushButton("Gate")
            self.createGate.setFixedWidth(85)
            self.createGate.setStyleSheet("background-color:#ffc7fd;")       

            self.createSumRegionButton = QPushButton("Sum. Region")
            self.createSumRegionButton.setFixedWidth(85)
            self.createSumRegionButton.setStyleSheet("background-color:#ffc7fd;") 

            # self.editGate = QPushButton("Modify")
            # self.editGate.setFixedWidth(85)
            # self.menu = QMenu()
            # #For 1d: Edit, for 2d: Drag and edit 
            # self.menu.addAction('Drag', self.drag)
            # self.menu.addAction('Edit', self.edit)
            # self.editGate.setMenu(self.menu)
            # self.editGate.setStyleSheet("background-color:#ffc7fd;")  

            self.integrateGateAndRegion = QPushButton("Integrate")
            self.integrateGateAndRegion.setFixedWidth(85)
            self.integrateGateAndRegion.setStyleSheet("background-color:#9f79ee;")

            self.extraButton = QPushButton("Extra")
            self.extraButton.setFixedWidth(85)
            self.extraButton.setStyleSheet("background-color:#ffd700;")

            
            for i in range(1,10):
                self.histo_geo_row.addItem(str(i))
                self.histo_geo_col.addItem(str(i))


            #line organized in several blocks 
            connectLayout = QHBoxLayout()
            geoLayout = QHBoxLayout()
            spectrumLayout = QHBoxLayout()
            gateLayout = QHBoxLayout()
            othersLayout = QHBoxLayout()

            connectLayout.addWidget(self.connectButton)

            geoLayout.addWidget(self.histo_geo_label)
            geoLayout.addWidget(self.histo_geo_row)
            geoLayout.addWidget(self.histo_geo_col)
            geoLayout.addWidget(self.loadButton)
            geoLayout.addWidget(self.saveButton)

            spectrumLayout.addWidget(self.histo_list_label)
            spectrumLayout.addWidget(self.histo_list)
            spectrumLayout.addWidget(self.histo_geo_add)
            spectrumLayout.addWidget(self.histo_geo_update)

            gateLayout.addWidget(self.createGate)
            gateLayout.addWidget(self.createSumRegionButton)
            gateLayout.addWidget(self.integrateGateAndRegion)

            othersLayout.addWidget(self.extraButton)
            othersLayout.addWidget(self.exitButton)

            self.addLayout(connectLayout, 0, 1, 0, 1)
            self.addLayout(geoLayout, 0, 2, 0, 1)
            self.addLayout(spectrumLayout, 0, 3, 0, 1)
            self.addLayout(gateLayout, 0, 4, 0, 1)
            self.addLayout(othersLayout, 0, 5, 0, 1)

            
    def drag(self):
        self.isDrag = True
        self.isEdit = False
        
    def edit(self):
        self.isDrag = False        
        self.isEdit = True

