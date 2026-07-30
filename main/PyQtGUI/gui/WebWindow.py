from PyQt5.QtCore import pyqtSlot, QSettings, QTimer, QUrl, Qt
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QDockWidget, QPlainTextEdit, QTabWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView as QWebView

from logger import log

SETTING_GEOMETRY = "geometry"

class LoggerDock(QDockWidget):

    def __init__(self, *args):
        super(LoggerDock, self).__init__(*args)
        self.textview = QPlainTextEdit(self)
        self.textview.setReadOnly(True)
        self.setWidget(self.textview)

    @pyqtSlot(str)
    def log(self, message):
        self.textview.appendPlainText(message)


class CustomWebView(QWebView):

    def __init__(self, mainwindow, main=False):
        super(CustomWebView, self).__init__(None)
        self.parent = mainwindow
        self.tabIndex = -1
        self.main = main
        self.loadedPage = None
        self.loadFinished.connect(self.onpagechange)

    @pyqtSlot(bool)
    def onpagechange(self, ok):
        log("Inside CustomWebView - on page change: %s, %s" % (self.url(), ok))
        if self.loadedPage is not None:
            log("Inside CustomWebView - disconnecting on close signal")
            self.loadedPage.windowCloseRequested.disconnect(self.close)
        self.loadedPage = self.page()
        log("Inside CustomWebView - connecting on close signal")
        self.loadedPage.windowCloseRequested.connect(self.close)
        self.setWindowTitle(self.title())
        if not self.main:
            self.parent.tabs.setTabText(self.tabIndex, self.title())
        if not ok:
            QMessageBox.information(self, "Error", "Error loading page!", QMessageBox.Ok)

        
    def createWindow(self, windowtype):
        return self.parent.createBrowserTab(windowtype, js=True)

    def closeEvent(self, event):
        if self.loadedPage is not None:
            log("Inside closeEvent - disconnecting on close signal")
            self.loadedPage.windowCloseRequested.disconnect(self.close)
        
        if not self.main:
            if self in self.parent.windows:
                self.parent.windows.remove(self)
            log("Inside closeEvent - Window count: %s" % (len(self.parent.windows)+1))
        event.accept()
        
class WebWindow(QMainWindow):

    def __init__(self, parent=None, homepage=None):
        super(WebWindow, self).__init__(parent)
        self.homepage = homepage
        self.windows = []

        self.loggerdock = LoggerDock("Log Message", self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.loggerdock)

        settings = QSettings()
        val = settings.value(SETTING_GEOMETRY, None)
        if val is not None:
            self.restoreGeometry(val)

        self.basewebview = CustomWebView(self, main=True)
        self.windows.append(self.basewebview)
        self.tabs = QTabWidget(self)
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(self.destroyBrowserTab)
        self.basewebview.tabIndex = self.tabs.addTab(self.basewebview, "File Browser")

        self.setCentralWidget(self.tabs)

    def loadmain(self, homepage):
        self.homepage = homepage
        QTimer.singleShot(0, self.initialload)

    def createBrowserTab(self, windowtype, js=True):
        v = CustomWebView(self)
        self.windows.append(v)
        v.tabIndex = self.tabs.addTab(v, "Window %s" % (len(self.windows)+1))
        self.tabs.setCurrentIndex(v.tabIndex)
        return v

    @pyqtSlot(int)
    def destroyBrowserTab(self, which):
        log("Inside destroyBrowserTab")
        closeevent = QCloseEvent()
        win = self.tabs.widget(which)
        if win.main:
            log("Inside destroyBrowserTab -- win.main")
            self.close()
        else:
            win.closeEvent(closeevent)
            if closeevent.isAccepted():
                log("Inside destroyBrowserTab -- !win.main -- closeevent.isAccepted")
                self.tabs.removeTab(which)

    @pyqtSlot()
    def initialload(self):
        if self.homepage:
            self.basewebview.load(QUrl(self.homepage))
        self.show()

    def savefile(self, url):
        pass

    def closeEvent(self, event):
        if len(self.windows) > 1:
            if QMessageBox.Ok == QMessageBox.information(self, "Really Close?",
                                                         "Really close %s tabs?" % (len(self.windows)),
                                                         QMessageBox.Cancel | QMessageBox.Ok):
                for i in reversed(range(len(self.windows))):
                    w = self.windows.pop(i)
                    w.close()
                event.accept()
            else:
                event.ignore()
                return
        else:
            event.accept()

        # save geometry
        settings = QSettings()
        settings.setValue(SETTING_GEOMETRY, self.saveGeometry())
