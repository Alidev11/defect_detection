########################################################################
## IMPORTS
########################################################################
import matplotlib.pyplot as plt
import torch
from utils.dataloader import get_train_test_loaders, get_cv_train_test_loaders
from utils.helper import train, evaluate, predict_localize
from utils.constants import NEG_CLASS
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from PySide2 import *
import psutil
#from multiprocessing import cpu_count
#import datetime
from qt_material import *
import shutil
import platform
import PySide2extn
from time import time, sleep
########################################################################

########################################################################
# IMPORT GUI FILE
from lastGUIapp import *


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        apply_stylesheet(app, theme='dark_cyan.xml')
        #######################################################################
        ## # Remove window tlttle bar
        ########################################################################
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)

        #######################################################################
        ## # Set main background to transparent
        ########################################################################
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        #######################################################################
        ## # Shadow effect style
        ########################################################################
        self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(50)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QtGui.QColor(0, 92, 157, 550))

        #######################################################################
        ## # Appy shadow to central widget
        ########################################################################
        self.ui.centralwidget.setGraphicsEffect(self.shadow)

        #######################################################################
        # Set window Icon
        # This icon and title will not appear on our app main window because we removed the title bar
        #######################################################################
        self.setWindowIcon(QtGui.QIcon(":/icons/feather/airplay.svg"))
        # Set window tittle
        self.setWindowTitle("UTIL Manager")

        #################################################################################
        # Window Size grip to resize window
        #################################################################################
        QtWidgets.QSizeGrip(self.ui.size_grip)

        #######################################################################
        # Minimize window
        self.ui.min_win_button.clicked.connect(lambda: self.showMinimized())
        #######################################################################
        # Close window
        self.ui.close_win_button.clicked.connect(lambda: self.close())
        #######################################################################
        # Restore/Maximize window
        self.ui.restore_win_button.clicked.connect(lambda: self.restore_or_maximize_window())
        #######################################################################
        # STACKED PAGES NAVIGATION/////////////////
        # Using side menu buttons
        #######################################################################

        # navigate to Home page
        self.ui.home_icon_button.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.home_page))
        # navigate to Statistics page
        self.ui.statistic_icon_button.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.statistics_page))
        # navigate to Infos page
        self.ui.infos_icon_button.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.infos_page))

        # Left Menu toggle button (Show hide menu labels)
        self.ui.menu_button.clicked.connect(lambda: self.slideLeftMenu())

        #######################################################################
        # Style clicked menu button
        for w in self.ui.left_menu_frame.findChildren(QPushButton):
            # Add click event listener
            w.clicked.connect(self.applyButtonStyle)

        self.showMaximized()
        if self.isMaximized():
            self.ui.restore_win_button.setIcon(QtGui.QIcon(u":/icons/cil-window-maximize.png"))


        self.file_path = None
        self.path_1 = None
        self.path_2 = None
        self.path_3 = None
        self.path_4 = None
        self.path_5 = None
        self.path_6 = None
        self.path_7 = None

        self.ui.openFile_button.clicked.connect(self.open_folder)
        self.ui.classify_button.clicked.connect(lambda: self.classify())
        print(self.file_path)
        self.show()

    def show_warning_messagebox(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)

        # setting message for Message Box
        msg.setText("Warning: You should select a folder first")

        # setting Message box window title
        msg.setWindowTitle("Warning!")

        # declaring buttons on Message Box
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        # start the app
        msg.exec_()


    def open_folder(self):
        foldername = QFileDialog.getExistingDirectory(
            caption="Select a folder",
            directory="data/mvtec_anomaly_detection/datasetSardine/test/bad"
        )
        self.file_path = foldername
        if self.file_path == "C:/Users/Mon Ordi/Desktop/Visual-Inspection-main-last/data/mvtec_anomaly_detection/datasetSardine/test/bad" or self.file_path == None :
            self.show_warning_messagebox()
            self.open_folder()
            return
        else:
            self.path_1 = self.file_path + "/1.jpg"
            self.path_2 = self.file_path + "/2.jpg"
            self.path_3 = self.file_path + "/3.jpg"
            self.path_4 = self.file_path + "/4.jpg"
            self.path_5 = self.file_path + "/5.jpg"
            self.path_6 = self.file_path + "/6.jpg"
            self.path_7 = self.file_path + "/7.jpg"

            self.ui.photo_1.setPixmap(QtGui.QPixmap(self.path_1).scaled(670, 130, QtCore.Qt.KeepAspectRatio))
            self.ui.photo_2.setPixmap(QtGui.QPixmap(self.path_2).scaled(670, 130, QtCore.Qt.KeepAspectRatio))
            self.ui.photo_3.setPixmap(QtGui.QPixmap(self.path_3).scaled(670, 130, QtCore.Qt.KeepAspectRatio))
            self.ui.photo_4.setPixmap(QtGui.QPixmap(self.path_4).scaled(670, 130, QtCore.Qt.KeepAspectRatio))
            self.ui.photo_5.setPixmap(QtGui.QPixmap(self.path_5).scaled(670, 130, QtCore.Qt.KeepAspectRatio))
            self.ui.photo_6.setPixmap(QtGui.QPixmap(self.path_6).scaled(670, 130, QtCore.Qt.KeepAspectRatio))
            self.ui.photo_7.setPixmap(QtGui.QPixmap(self.path_7).scaled(670, 130, QtCore.Qt.KeepAspectRatio))
            print(self.file_path)
            #self.file_path = path


    def classify(self):
        if self.file_path == "C:/Users/Mon Ordi/Desktop/Visual-Inspection-main-last/data/mvtec_anomaly_detection/datasetSardine/test/bad" or self.file_path == None:
            self.show_warning_messagebox()
            self.open_folder()
            return
        else:
            data_folder = "./data/mvtec_anomaly_detection"
            subset_name = "datasetSardine"
            data_folder = os.path.join(data_folder, subset_name)

            batch_size = 10
            class_weight = [1, 3] if NEG_CLASS == 1 else [3, 1]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            heatmap_thres = 0.7

            train_loader, test_loader = get_train_test_loaders(
                root=data_folder, batch_size=batch_size, test_size=0.2, random_state=42
            )
            model_path = f"./weights/{subset_name}_model.h5"
            model = torch.load(model_path, map_location=device)
            heat=False
            predictionn=predict_localize(
                model, test_loader, device, self.path_2, thres=heatmap_thres, n_samples=1, show_heatmap=False
            )

            #self.ui.photo_1.setStyleSheet("background-color: transparent;")
            self.ui.photo_1.setPixmap(QtGui.QPixmap("./classified/zoo0.png").scaled(700, 170, QtCore.Qt.KeepAspectRatio))
            self.ui.class_label_1.setText(predictionn)

            self.ui.photo_2.setPixmap(QtGui.QPixmap("./classified/zoo0.png").scaled(700, 170, QtCore.Qt.KeepAspectRatio))
            self.ui.class_label_2.setText(predictionn)

            self.ui.photo_3.setPixmap(QtGui.QPixmap("./classified/zoo0.png").scaled(700, 170, QtCore.Qt.KeepAspectRatio))
            self.ui.class_label_3.setText(predictionn)

            self.ui.photo_4.setPixmap(QtGui.QPixmap("./classified/zoo0.png").scaled(700, 170, QtCore.Qt.KeepAspectRatio))
            self.ui.class_label_4.setText(predictionn)

            self.ui.photo_5.setPixmap(QtGui.QPixmap("./classified/zoo0.png").scaled(700, 170, QtCore.Qt.KeepAspectRatio))
            self.ui.class_label_5.setText(predictionn)

            self.ui.photo_6.setPixmap(QtGui.QPixmap("./classified/zoo0.png").scaled(700, 170, QtCore.Qt.KeepAspectRatio))
            self.ui.class_label_6.setText(predictionn)

            self.ui.photo_7.setPixmap(QtGui.QPixmap("./classified/zoo0.png").scaled(700, 170, QtCore.Qt.KeepAspectRatio))
            self.ui.class_label_7.setText(predictionn)

            self.ui.photo_1.adjustSize()


    def restore_or_maximize_window(self):
        if self.isMaximized():
            self.showNormal()
            self.ui.restore_win_button.setIcon(QtGui.QIcon(u":/icons/cil-window-restore.png"))
        else:
            self.showMaximized()
            self.ui.restore_win_button.setIcon(QtGui.QIcon(u":/icons/cil-window-maximize.png"))

    # ###############################################
    # Function to Move window on mouse drag event on the tittle bar
    # ###############################################
    def mousePressEvent(self, event):
        self.oldPosition = event.globalPos()

        # action #2

    def mouseMoveEvent(self, event):
        delta = QPoint(event.globalPos() - self.oldPosition)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPosition = event.globalPos()

    def applyButtonStyle(self):
        # Reset style for other buttons
        for w in self.ui.left_menu_frame.findChildren(QPushButton):
            # If the button name is not equal to clicked button name
            if w.objectName() != self.sender().objectName():
                # Create default style by removing the left border
                # Lets remove the bottom border style

                # Lets also remove the left border style

                # Apply the default style
                w.setStyleSheet("border-bottom: none;")
                #

        # Apply new style to clicked button
        # Sender = clicked button
        # Get the clicked button stylesheet then add new left-border style to it
        # Lets add the bottom border style
        # Apply the new style
        self.sender().setStyleSheet("border-left: 2px solid #42c3ca;")
        #
        return

    def slideLeftMenu(self):
        # Get current left menu width
        width = self.ui.menu_frame.width()

        # If minimized
        if width == 46:
            # Expand menu
            newWidth = 141
            self.ui.menu_button.setIcon(QtGui.QIcon(u":/icons/feather/white/chevron-left.svg"))
        # If maximized
        else:
            # Restore menu
            newWidth = 46
            self.ui.menu_button.setIcon(QtGui.QIcon(u":/icons/feather/white/align-left.svg"))

        # Animate the transition
        self.animation = QPropertyAnimation(self.ui.menu_frame, b"minimumWidth")  # Animate minimumWidht
        self.animation.setDuration(250)
        self.animation.setStartValue(width)  # Start value is the current menu width
        self.animation.setEndValue(newWidth)  # end value is the new menu width
        self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.animation.start()
    #######################################################################


########################################################################
## EXECUTE APP
########################################################################
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
########################################################################
## END===>
########################################################################