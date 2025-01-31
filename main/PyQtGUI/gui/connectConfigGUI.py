import os,sys,re
import socket
import getpass
import psutil
import time
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import CPyConverter as cpy


# Check DAQ version, need Tkinter module for the port manager
def checkVersionNumber(string):
        # Usual version
        match = re.search(r'daq/(\d{2,}\.\d{1,}-\d{3,})', string)
        # Pre version
        matchPre = re.search(r'daq/(\d{2,}\.\d{1,}-pre\d{1,})', string)
        if match:
                versionNumber = match.group(1)
                major, minor_patch = versionNumber.split('.')
                minor, patch = minor_patch.split('-')
                major, minor, patch = int(major), int(minor), int(patch)

                if major >= 11 and minor >= 3 and patch >= 29:
                        return True
                else:
                        return False
        elif matchPre:
                versionNumber = matchPre.group(1)
                major, minor_patch = versionNumber.split('.')
                minor, patch = minor_patch.split('-pre')
                major, minor, patch = int(major), int(minor), int(patch)

                if major >= 12 and minor >= 0 and patch >= 0:
                        return True
                else:
                        return False
        else:
                return False


# check if DAQROOT defined, to know if can use 3python portManager to get REST/MIRROR ports number
daqroot = os.environ.get('DAQROOT', '')
if daqroot :
        # Check that port manager can be used with that daq version
        if checkVersionNumber(daqroot):
                # Dirty trick with sys.argv... __init__ of "/usr/lib/python3.7/tkinter/__init__.py" uses sys.argv[0], and here argv is empty
                if len(sys.argv) == 0: sys.argv = ["blabla"]
                import nscldaq.portmanager.PortManager as mPM   
                if sys.argv[0] == "blabla": sys.argv = []
        else:
                print("Your DAQ version is too old (<11.3-029) to benefit from the port manager features.")
                daqroot = ''



class ConnectConfiguration(QWidget):
        def __init__(self, loggerMain):
                super(ConnectConfiguration, self).__init__()

                self.setWindowModality(True)

                self.logger = loggerMain

                self.thisCutiePiePID = psutil.Process().pid
                self.isOpennedWithSpecTcl = psutil.Process(self.thisCutiePiePID).parent().name == 'SpecTcl'

                self.setWindowTitle("Connection configuration")
                self.resize(300, 150)

                self.serverLabel = QLabel()
                self.serverLabel.setText('Server')
                self.serverLabel.setFixedWidth(50) 
                self.server = QLineEdit()
                self.server.setFixedWidth(200) 

                self.userLabel = QLabel("User")
                self.userLabel.setFixedWidth(40) 
                self.user = QLineEdit()                
                self.user.setFixedWidth(200) 

                self.restLabel = QLabel()
                self.restLabel.setText('REST port')
                self.restLabel.setFixedWidth(70) 
                self.rest = QLineEdit()
                self.rest.setFixedWidth(100) 
                self.restLabel.setToolTip("Set explicitly in SpecTclInit.tcl or automatically by a port manager, in the latter case please choose a SpecTcl PID")
                self.rest.setToolTip("Set explicitly in SpecTclInit.tcl or automatically by a port manager, in the latter case please choose a SpecTcl PID")

                self.mirrorLabel = QLabel("Mirror port")
                self.mirrorLabel.setFixedWidth(75) 
                self.mirror = QLineEdit()
                self.mirror.setFixedWidth(100) 
                self.mirrorLabel.setToolTip("Set explicitly in SpecTclInit.tcl or automatically by a port manager, in the latter case please choose a SpecTcl PID")
                self.mirror.setToolTip("Set explicitly in SpecTclInit.tcl or automatically by a port manager, in the latter case please choose a SpecTcl PID")

                self.spectclPIDLabel = QLabel("SpecTcl PID")
                self.spectclPIDLabel.setFixedWidth(75) 
                self.spectclPID = QComboBox()
                self.spectclPID.setFixedWidth(100) 
                if self.canUsePortManger() :
                        self.spectclPIDLabel.setToolTip("Please choose the desired SpecTcl, the associated ports will be set automatically.")
                        self.spectclPID.setToolTip("Please choose the desired SpecTcl, the associated ports will be set automatically.")
                        self.logger.debug('ConnectConfiguration - Init - With port manager')
                else :
                        self.spectclPIDLabel.setToolTip("Obtained via a port manager for non-local SpecTcl, to enable please set $DAQROOT")
                        self.spectclPID.setToolTip("Obtained via a port manager for non-local SpecTcl, to enable please set $DAQROOT")
                        self.spectclPID.setEnabled(False)
                        self.spectclPIDLabel.setStyleSheet("QLabel { color: gray; }")
                        self.logger.debug('ConnectConfiguration - Init - Without port manager')


                # When CutiePie is openned with SpecTcl set fields with process env variables.
                if self.isOpennedWithSpecTcl:
                        self.setFieldsIsOpennedWithSpecTcl()
                        self.logger.debug('ConnectConfiguration - Init - openned with SpecTcl')
                else :
                        self.logger.debug('ConnectConfiguration - Init - openned standalone')
                        # When CutiePie not openned with SpecTcl, check if can use port manager.
                        # If port manager not available set fields to default.
                        default = False
                        if self.canUsePortManger() :
                                self.setPortManager(socket.gethostname())
                        else :
                                default = True
                        self.setFields(socket.gethostname(), getpass.getuser(), default)

                        # CUTIEPIE_CONNECT in '~/.failsafe_cutiepie' facilitates recovery from previous CutiePie connection 
                        # Valid only when pass here in the Init, i.e. for the first time connectConfig popup is openned
                        try:
                                with open('~/.failsafe_cutiepie', "r") as failfile:
                                        for line in failfile:
                                                if "CUTIEPIE_CONNECT=" in line:
                                                        line = line.strip()
                                                        var, info = line.split('=')
                                                        infoFromFailSafe = info.split('::')
                                if len(infoFromFailSafe) == 5:
                                        self.setFieldsFromEnv(infoFromFailSafe)
                                        self.logger.debug('ConnectConfiguration - Init - found CUTIEPIE_CONNECT in ~/.failsafe_cutiepie')
                        except:
                                pass
                
                self.ok = QPushButton("Ok", self)
                self.cancel = QPushButton("Cancel", self)            

                layout = QGridLayout()
                layout.addWidget(self.serverLabel, 1, 1, QtCore.Qt.AlignLeft)
                layout.addWidget(self.server, 1, 2, 1, 1)
                layout.addWidget(self.userLabel, 2, 1, QtCore.Qt.AlignLeft)
                layout.addWidget(self.user, 2, 2, 1, 1)    
                layout.addWidget(self.spectclPIDLabel, 3, 1, QtCore.Qt.AlignLeft)
                layout.addWidget(self.spectclPID, 3, 2, 1, 1)         
                layout.addWidget(self.restLabel, 4, 1, QtCore.Qt.AlignLeft)    
                layout.addWidget(self.rest, 4, 2, 1, 1)    
                layout.addWidget(self.mirrorLabel, 5, 1, QtCore.Qt.AlignLeft)    
                layout.addWidget(self.mirror, 5, 2, 1, 1)    
                lay5 = QHBoxLayout()
                lay5.addWidget(self.ok)            
                lay5.addWidget(self.cancel)
                layout.addLayout(lay5, 6, 1, 1, 2)    
                self.setLayout(layout)


        def portManagerAvailable(self):
                if daqroot:
                        return True
                return False

        def canUsePortManger(self, server=socket.gethostname()):
                self.logger.debug('canUsePortManger')
                if self.portManagerAvailable():
                        try:
                                dummyList = mPM.PortManager(server).listPorts()
                                return True
                        except:
                                self.logger.warning(f'canUsePortManger - issue with port manager on server {server}')
                                self.logger.debug(f'canUsePortManger - issue with port manager on server {server}', exc_info=True)
                                return False
                else :
                        return False


        def setPortManager(self, host):
                success = True
                try:
                        self.portManager = mPM.PortManager(host)
                except:
                        success = False
                self.logger.debug(f'ConnectConfiguration - setPortManager for host: {host}, success: {success}')
                return success


        def setFieldsIsOpennedWithSpecTcl(self):
                self.logger.debug('ConnectConfiguration - setFieldsIsOpennedWithSpecTcl')
                self.server.setText(os.environ["RESThost"])
                self.user.setText(getpass.getuser())
                self.rest.setText(os.environ["RESTport"])
                self.mirror.setText(os.environ["MIRRORport"])
                # Set the PID list
                parentSpecTclPID = psutil.Process(self.thisCutiePiePID).parent().pid
                if self.canUsePortManger(os.environ["RESThost"]) :
                        self.setPortManager(os.environ["RESThost"])
                        self.spectclPID.addItems(self.getServiceInfo(getpass.getuser(), 'mirror', 'pid', 'all'))
                        indexParentSpecTclPID = self.spectclPID.findText(parentSpecTclPID, QtCore.Qt.MatchFixedString)
                        if indexParentSpecTclPID >= 0:
                                self.spectclPID.setCurrentIndex(indexParentSpecTclPID)
                        else:
                                print("WARNING - SpecTcl PID retrieved from psutil lib is not found with port manager.")
                else :
                        self.spectclPID.addItem(parentSpecTclPID)


        def setFields(self, server=socket.gethostname(), user=getpass.getuser(), default=False):
                self.logger.debug(f'ConnectConfiguration - setFields - server: {server}, user: {user}, default: {default}')
                if default:
                        self.server.setText(socket.gethostname())
                        self.user.setText(getpass.getuser())
                        self.rest.setText("")
                        self.mirror.setText("")
                        return
                else:
                        self.server.setText(server)
                        self.user.setText(user)
                        if self.canUsePortManger(server):
                                # with port manager, SpecTcl PID, rest and mirror ports are set by:
                                self.getFieldsFromPortManager(user)


        def updateFields(self, server=socket.gethostname(), user=getpass.getuser(), default=False):
                self.logger.debug(f'ConnectConfiguration - updateFields - server: {server}, user: {user}, default: {default}')
                if default:
                        self.server.setText(socket.gethostname())
                        self.user.setText(getpass.getuser())
                        self.spectclPID.clear()
                        self.rest.setText("")
                        self.mirror.setText("")
                        return

                if server == socket.gethostname() and user == getpass.getuser() and self.isOpennedWithSpecTcl:
                        self.setFieldsIsOpennedWithSpecTcl()
                else:
                        self.server.setText(server)
                        self.user.setText(user)
                        if self.canUsePortManger(server):
                                self.getFieldsFromPortManager(user)


        def setFieldsFromEnv(self, info):
                self.logger.debug(f'ConnectConfiguration - setFieldsFromEnv - info: {info}')
                self.server.setText(info[0])
                self.user.setText(info[1])
                self.spectclPID.clear()
                pid = info[4]
                # if port manager is available, get list of SpecTcl PIDs, else get PID from env var
                if self.canUsePortManger(self.server.text()):
                        self.spectclPID.addItems(self.getServiceInfo(self.user.text(), 'mirror', 'pid', 'all'))
                        index = self.spectclPID.findText(pid, QtCore.Qt.MatchFixedString)
                        if index >= 0:
                                self.spectclPID.setCurrentIndex(index)
                        else:
                                self.logger.warning(f'ConnectConfiguration - setFieldsFromEnv - SpecTcl PID retrieved from CUTIEPIE_CONNECT is no longer in port manager.')
                else:
                        self.spectclPID.addItem(pid)

                self.rest.setText(info[2])
                self.mirror.setText(info[3])


        def getFieldsFromPortManager(self, user):
                self.logger.debug(f'ConnectConfiguration - getFieldsFromPortManager - user: {user}')
                # Fill SpecTcl PID list with all running spectcl for the given host-user pair.
                self.spectclPID.addItems(self.getServiceInfo(user, 'mirror', 'pid', 'all'))
                # Set port fields according to top of spectcl PID list.
                self.rest.setText(str(self.getServiceInfo(user, 'rest', 'port', 'first')))
                self.mirror.setText(str(self.getServiceInfo(user, 'mirror', 'port', 'first')))


        def getServiceInfo(self, host, type, info, whichOne, pid=0):
                self.logger.debug(f'ConnectConfiguration - getServiceInfo - host: {host}, type: {type}, info: {info}, whichOne: {whichOne}, pid: {pid}')
                # Get service name from port manager 
                # type expected: 'rest' or 'mirror'
                # info expected: 'port' or 'pid'
                # if pid not null then select also by pid
                # whichOne expected: 'first', 'last', or 'all' (in case of several services for a given host-user pair)
                serviceList = []
                if type == 'rest':
                        serviceList = self.portManager.find(beginswith="SpecTcl_REST", user=host)
                elif type == 'mirror':
                        serviceList = self.portManager.find(beginswith="SpecTcl_MIRROR", user=host)
                else :
                        print('getServiceInfo - bad type')
                        
                if len(serviceList) == 0:
                        self.logger.debug('ConnectConfiguration - getServiceInfo - serviceList empty')
                        if whichOne == "all":
                                return []
                        return "None"

                if info == 'port':
                        if pid != 0:
                                serviceList = [serviceDict['port'] for serviceDict in serviceList if self.extractNumbers(serviceDict['service'])[-1]==pid]
                        else:
                                serviceList = [serviceDict['port'] for serviceDict in serviceList]
                elif info == 'pid':
                        # Expect pid to follow service name. Take mirror service here.
                        serviceList = [self.extractNumbers(serviceDict['service'])[-1] for serviceDict in serviceList]
                else :
                        self.logger.debug('ConnectConfiguration - getServiceInfo - info unrecognized')
                
                if len(serviceList) == 0:
                        self.logger.debug('ConnectConfiguration - getServiceInfo - serviceList empty')
                        if whichOne == "all":
                                return []
                        return "None"

                if whichOne == 'first':
                        return serviceList[0]
                elif whichOne == 'last':
                        return serviceList[-1]
                elif whichOne == 'all':
                        return serviceList


        # Indicates if should close CutiePie depending if was openned with SpecTcl.
        def shouldCloseCutiePie(self):
                if self.server.text() == socket.gethostname() and self.user.text() == getpass.getuser() \
                   and self.isOpennedWithSpecTcl:
                        parentSpecTclPID = psutil.Process(self.thisCutiePiePID).parent().pid
                        try :
                                psutil.Process(parentSpecTclPID)
                                return False 
                        except :
                                self.logger.debug('ConnectConfiguration - shouldCloseCutiePie True')
                                return True
                else :
                        return False


        # Update SpecTcl PID according to port manager if exist or local SpecTcl
        def updateSpecTclPID(self):
                self.logger.debug('ConnectConfiguration - updateSpecTclPID')
                currentPIDs = [self.spectclPID.itemText(i) for i in range(self.spectclPID.count())]
                if self.canUsePortManger(self.server.text()):
                        # port manager doesn't seem to be updated soon enough
                        if self.server.text() != socket.gethostname() :
                                PIDsRef = self.getServiceInfo(self.user.text(), 'mirror', 'pid', 'all')
                        # if local host, prefer use psutil to check if proc is alive
                        else :
                                PIDsRef = [proc.info['pid'] for proc in psutil.process_iter(['pid', 'name', 'username']) \
                                        if proc.info['name'] == 'SpecTcl' and proc.info['username'] == self.user.text()]
                else :
                        if self.server.text() != socket.gethostname():
                                return 
                        PIDsRef = [proc.info['pid'] for proc in psutil.process_iter(['pid', 'name', 'username']) \
                                if proc.info['name'] == 'SpecTcl' and proc.info['username'] == self.user.text()]

                currentPIDsSet = set(currentPIDs)
                PIDsRefSet = set(PIDsRef)

                # Add missing PIDs from ref to combobox
                for pid in PIDsRefSet - currentPIDsSet:
                        self.spectclPID.insertItem(self.spectclPID.count(), str(pid))

                # Remove missing PIDs from combobox
                for pid in currentPIDsSet - PIDsRefSet:
                        self.spectclPID.removeItem(self.spectclPID.findText(pid))


        # Return list of numbers extracted from the string arg.
        def extractNumbers(self, string):
                numbers = re.findall(r'\d+', string)
                return [num for num in numbers]






