import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataloader import get_train_test_loaders, get_cv_train_test_loaders
from utils import model
from utils.model import CustomVGG
from utils.helper import train, evaluate, predict_localize
from utils.constants import NEG_CLASS
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(569, 572)
        MainWindow.setStyleSheet("background-color: rgb(83, 83, 83);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("background-color: rgb(255, 255, 255)")
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(60, 30, 431, 161))
        self.label.setStyleSheet("background-color: rgb(145, 145, 145);")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(40, 300, 481, 251))
        self.label_2.setStyleSheet("background-color: rgb(116, 116, 116);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(210, 270, 141, 20))
        self.label_3.setStyleSheet("text-align: center;\n"
                                   "font: 12pt \"MS Shell Dlg 2\";")
        self.label_3.setObjectName("label_3")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setGeometry(QtCore.QRect(60, 220, 441, 23))
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.pushButton_3 = QtWidgets.QPushButton(self.splitter)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_2 = QtWidgets.QPushButton(self.splitter)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton = QtWidgets.QPushButton(self.splitter)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_4 = QtWidgets.QPushButton(self.splitter)
        self.pushButton_4.setObjectName("pushButton_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate("MainWindow", "Classification Result"))
        self.pushButton_3.setText(_translate("MainWindow", "Left"))
        self.pushButton_2.setText(_translate("MainWindow", "Classify"))
        self.pushButton.setText(_translate("MainWindow", "Right"))
        self.pushButton_4.setText(_translate("MainWindow", "loadFile"))
        self.pushButton_4.clicked.connect(self.pushButton_handler)
        self.pushButton_2.clicked.connect(self.pushButton_handler2)

    def pushButton_handler(self):
        self.open_dialog_box()

    def pushButton_handler2(self):
        data_folder = "./data/mvtec_anomaly_detection"
        subset_name = "serdine2"
        data_folder = os.path.join(data_folder, subset_name)

        batch_size = 10
        lr = 0.0001
        epochs = 50
        class_weight = [1, 3] if NEG_CLASS == 1 else [3, 1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        heatmap_thres = 0.7

        train_loader, test_loader = get_train_test_loaders(
            root=data_folder, batch_size=batch_size, test_size=0.2, random_state=42
        )
        model_path = f"./weights/{subset_name}_model.h5"
        # torch.save(model, model_path)
        model = torch.load(model_path, map_location=device)
        predict_localize(
            model, test_loader, device, thres=heatmap_thres, n_samples=1, show_heatmap=True
        )
        self.label_2.setStyleSheet("background-color: transparent;")
        #qpix=QtGui.QPixmap("./classified/zoo.png")
        self.label_2.setPixmap(QtGui.QPixmap("./classified/zoo0.png").scaled(670, 250, QtCore.Qt.KeepAspectRatio))

    def open_dialog_box(self):
        filename = QFileDialog.getOpenFileName()
        path = filename[0]
        self.show_loaded_image(path)

    def show_loaded_image(self, path):
        self.label.setPixmap(QtGui.QPixmap(path))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


