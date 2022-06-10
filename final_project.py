########################################################################
## IMPORTS
########################################################################
import matplotlib.pyplot as plt
import sys
import time
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar, QLabel, QFrame, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer
import torch
from utils.dataloader import get_train_test_loaders, get_cv_train_test_loaders
from utils.helper import train, evaluate, predict_localize
from utils.constants import NEG_CLASS
from PySide2.QtGui import QPainter
from PyQt5.QtChart import QChart, QChartView, QBarSet, QPercentBarSeries, QBarCategoryAxis
from PyQt5.QtGui import QPainter
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from qt_material import *



# IMPORT GUI FILE
from lastGUIapp import *
shadow_elements = {
    "defected_card_frame",
    "good_card_frame"
}

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SpLash Screen Example')
        self.setFixedSize(700, 400)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.counter = 0
        self.n = 300 # total instance

        self.initUI()

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self.loading())
        self.timer.start(30)

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.frame = QtWidgets.QFrame()
        layout.addWidget(self.frame)

        self.labelTitle = QtWidgets.QLabel(self.frame)
        self.labelTitle.setObjectName('LabelTitle')

        # center labels
        self.labelTitle.resize(self.width() - 10, 150)
        self.labelTitle.move(0, 20) # x, y
        self.labelTitle.setText('Visual Inspection')
        self.labelTitle.setAlignment(QtCore.Qt.AlignCenter)

        self.labelDescription = QLabel(self.frame)
        self.labelDescription.resize(self.width() - 10, 50)
        self.labelDescription.move(0, self.labelTitle.height())
        self.labelDescription.setObjectName('LabelDesc')
        self.labelDescription.setText('<strong>Working on Task #1</strong>')
        self.labelDescription.setAlignment(QtCore.Qt.AlignCenter)

        self.progressBar = QProgressBar(self.frame)
        self.progressBar.resize(self.width() - 200 - 10, 50)
        self.progressBar.move(100, self.labelDescription.y() + 80)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setFormat('%p%')
        self.progressBar.setTextVisible(True)
        self.progressBar.setRange(0, self.n)
        self.progressBar.setValue(20)

        self.labelLoading = QLabel(self.frame)
        self.labelLoading.resize(self.width() - 10, 50)
        self.labelLoading.move(0, self.progressBar.y() + 70)
        self.labelLoading.setObjectName('LabelLoading')
        self.labelLoading.setAlignment(QtCore.Qt.AlignCenter)
        self.labelLoading.setText('loading...')

    def loading(self):
        self.progressBar.setValue(self.counter)

        if self.counter == int(self.n * 0.3):
            self.labelDescription.setText('<strong>Working on Task #2</strong>')
        elif self.counter == int(self.n * 0.6):
            self.labelDescription.setText('<strong>Working on Task #3</strong>')
        elif self.counter >= self.n:
            self.timer.stop()
            self.close()

            time.sleep(1)

            self.myApp = MainWindow()
            self.myApp.show()

        self.counter += 1



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
        self.create_percentage_bar_chart()
        print(self.file_path)
        #self.show()
        for x in shadow_elements:
            #######################################################################
            ## # Shadow effect style
            ########################################################################
            effect = QtWidgets.QGraphicsDropShadowEffect(self)
            effect.setBlurRadius(35)
            effect.setXOffset(10)
            effect.setYOffset(10)
            effect.setColor(QtGui.QColor(0, 0, 0, 100))
            getattr(self.ui, x).setGraphicsEffect(effect)

    def show_warning_messagebox(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Warning: You should select a folder first")
        # setting Message box window title
        msg.setWindowTitle("Warning!")
        # declaring buttons on Message Box
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.exec_()


    def open_folder(self):
        foldername = QFileDialog.getExistingDirectory(
            caption="Select a folder",
            directory="data/mvtec_anomaly_detection/datasetSardine/test"
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
            #print(self.file_path)

###Rename

    def create_percentage_bar_chart(self):

        set0 = QBarSet("Defected")
        set1 = QBarSet("Good")

        set0.append([1, 2, 3,  4, 5, 6, 1, 2, 3,  4, 5, 6])
        set1.append([5, 0, 0,  4, 0, 7, 5, 0, 0,  4, 0, 7])

        series = QPercentBarSeries()
        series.append(set0)
        series.append(set1)

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Defected vs Good cans in 2022")
        chart.setAnimationOptions(QChart.SeriesAnimations)


        categories = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "July", "Aug", "Sep", "Oct", "Nov", "Dec"]
        axis = QBarCategoryAxis()
        axis.append(categories)
        chart.createDefaultAxes()
        chart.setAxisX(axis, series)

        chart.legend().setVisible(True)
        #chart.legend().setAlignment(Qt.AlignBottom)


        self.ui.chart_view = QChartView(chart)
        self.ui.chart_view.setRenderHint(QPainter.Antialiasing)
        self.ui.chart_view.chart().setTheme(QChart.ChartThemeDark)
        # QChart.setTheme(theme)

        # print(self.ui.chart_view.chart().theme())
        # self.ui.chart_view.chart().setBackgroundBrush(QtGui.QColor("gray"))

        # self.setCentralWidget(chart_view)

        # self.lineEdit = QLineEdit(self.percentage_bar_chart_cont)
        # self.lineEdit.setObjectName(u"lineEdit")

        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ui.chart_view.sizePolicy().hasHeightForWidth())
        self.ui.chart_view.setSizePolicy(sizePolicy)
        self.ui.chart_view.setMinimumSize(QSize(0, 300))
        self.ui.percentage_bar_chart_cont.addWidget(self.ui.chart_view, 0, 0,  9, 9)
        self.ui.chart_frame.setStyleSheet(u"background-color: transparent")

    def classify_top(self):
        data_folder = "./data/mvtec_anomaly_detection"
        subset_name = "datasetSardine_top"
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
        heat = False
        bbox = False
        if self.ui.options_checkox_1.isChecked():
            heat = True
        if self.ui.options_checkox_2.isChecked():
            bbox = True
        predictionn = predict_localize(
            model, test_loader, device, self.path_4, bbox, thres=heatmap_thres, n_samples=1, show_heatmap=heat
        )
        self.ui.photo_4.setPixmap(
            QtGui.QPixmap("./classified/zoo0.png").scaled(700, 170, QtCore.Qt.KeepAspectRatio))
        self.ui.photo_4.setScaledContents(True)
        if predictionn == "Good":
            self.ui.class_label_4.setStyleSheet("background-color: #5FD068;")
        else:
            self.ui.class_label_4.setStyleSheet("background-color: #F00C44;")
        self.ui.class_label_4.setText(predictionn)


    def classify_corner(self):
        data_folder = "./data/mvtec_anomaly_detection"
        subset_name = "corner"
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
        heat = False
        bbox = False
        if self.ui.options_checkox_1.isChecked():
            heat = True
        if self.ui.options_checkox_2.isChecked():
            bbox = True

        for i in range(4):
            predictionn = predict_localize(
                model, test_loader, device, self.path_1, bbox, thres=heatmap_thres, n_samples=1, show_heatmap=heat
            )
            if i == 0:
                self.ui.photo_1.setPixmap(
                    QtGui.QPixmap("./classified/zoo0.png").scaled(700, 170, QtCore.Qt.KeepAspectRatio))
                self.ui.photo_1.setScaledContents(True)
                if predictionn == "Good":
                    self.ui.class_label_1.setStyleSheet("background-color: #5FD068;")
                else:
                    self.ui.class_label_1.setStyleSheet("background-color: #F00C44;")
                self.ui.class_label_1.setText(predictionn)
                x = self.path_1
                self.path_1 = self.path_3
            elif i == 1:
                self.ui.photo_3.setPixmap(
                    QtGui.QPixmap("./classified/zoo0.png").scaled(700, 170, QtCore.Qt.KeepAspectRatio))
                self.ui.photo_3.setScaledContents(True)
                if predictionn == "Good":
                    self.ui.class_label_3.setStyleSheet("background-color: #5FD068;")
                else:
                    self.ui.class_label_3.setStyleSheet("background-color: #F00C44;")
                self.ui.class_label_3.setText(predictionn)
                self.path_1 = self.path_5
            elif i == 2:
                self.ui.photo_5.setPixmap(
                    QtGui.QPixmap("./classified/zoo0.png").scaled(700, 170, QtCore.Qt.KeepAspectRatio))
                self.ui.photo_5.setScaledContents(True)
                if predictionn == "Good":
                    self.ui.class_label_5.setStyleSheet("background-color: #5FD068;")
                else:
                    self.ui.class_label_5.setStyleSheet("background-color: #F00C44;")
                self.ui.class_label_5.setText(predictionn)
                self.path_1 = self.path_7
            else:
                self.ui.photo_7.setPixmap(
                    QtGui.QPixmap("./classified/zoo0.png").scaled(700, 170, QtCore.Qt.KeepAspectRatio))
                self.ui.photo_7.setScaledContents(True)
                if predictionn == "Good":
                    self.ui.class_label_7.setStyleSheet("background-color: #5FD068;")
                else:
                    self.ui.class_label_7.setStyleSheet("background-color: #F00C44;")
                self.ui.class_label_7.setText(predictionn)
                self.path_1 = x


###Classify Front And Back
    def classify_front(self):
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
        heat = False
        bbox = False
        if self.ui.options_checkox_1.isChecked():
            heat = True
        if self.ui.options_checkox_2.isChecked():
            bbox = True

        for i in range(2):
            predictionn = predict_localize(
                model, test_loader, device, self.path_2, bbox, thres=heatmap_thres, n_samples=1, show_heatmap=heat
            )
            if i == 0:
                self.ui.photo_2.setPixmap(
                    QtGui.QPixmap("./classified/zoo0.png").scaled(700, 170, QtCore.Qt.KeepAspectRatio))
                self.ui.photo_2.setScaledContents(True)
                if predictionn == "Good":
                    self.ui.class_label_2.setStyleSheet("background-color: #5FD068;")
                else:
                    self.ui.class_label_2.setStyleSheet("background-color: #F00C44;")
                self.ui.class_label_2.setText(predictionn)
                x = self.path_2
                self.path_2 = self.path_6
            else:
                self.ui.photo_6.setPixmap(
                    QtGui.QPixmap("./classified/zoo0.png").scaled(700, 170, QtCore.Qt.KeepAspectRatio))
                self.ui.photo_6.setScaledContents(True)
                if predictionn == "Good":
                    self.ui.class_label_6.setStyleSheet("background-color: #5FD068;")
                else:
                    self.ui.class_label_6.setStyleSheet("background-color: #F00C44;")
                self.ui.class_label_6.setText(predictionn)
                self.path_2 = x




    def classify(self):
        if self.file_path == "C:/Users/Mon Ordi/Desktop/Visual-Inspection-main-last/data/mvtec_anomaly_detection/datasetSardine/test/bad" or self.file_path == None:
            self.show_warning_messagebox()
            self.open_folder()
            return
        else:
            self.classify_front()
            self.classify_corner()
            self.classify_top()

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
    app.setStyleSheet('''
            #LabelTitle {
                font-size: 60px;
                color: #93deed;
            }

            #LabelDesc {
                font-size: 30px;
                color: #c2ced1;
            }

            #LabelLoading {
                font-size: 30px;
                color: #e8e8eb;
            }

            QFrame {
                background-color: #2F4454;
                color: rgb(220, 220, 220);
            }

            QProgressBar {
                background-color: #DA7B93;
                color: rgb(200, 200, 200);
                border-style: none;
                border-radius: 10px;
                text-align: center;
                font-size: 30px;
            }

            QProgressBar::chunk {
                border-radius: 10px;
                background-color: qlineargradient(spread:pad x1:0, x2:1, y1:0.511364, y2:0.523, stop:0 #1C3334, stop:1 #376E6F);
            }
        ''')

    splash = SplashScreen()
    splash.show()
    #window = MainWindow()
    try:
        sys.exit(app.exec_())
    except SystemExit:
        print('Closing Window...')
########################################################################
## END===>
########################################################################