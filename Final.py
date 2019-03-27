import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

sample = {'col0': [1], 'col1': [40], 'col2': [1], 'col3': [4], 'col4': [0], 'col5': [0], 'col6': [1], 'col7': [1],
          'col8': [395], 'col9': [100], 'col10': [70], 'col11': [23], 'col12': [80], 'col13': [70]}

y_re = []

print("original sample: ", sample)

# Importing the dataset
dataset = pd.read_csv("framingham.csv")
dataset.drop(['education'], axis=1, inplace=True)

# Creating matrix of independent features
X = dataset.iloc[:, :-1].values

# Creating dependent variable vector
Y = dataset.iloc[:, 14].values

# Dealing with missing values
imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imputer = imputer.fit(X[:, 0:14])
X[:, 0:14] = imputer.transform(X[:, 0:14])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heart Disease Prediction")
        self.setFixedSize(320, 570)
        self.setWindowIcon(QtGui.QIcon('heart.png'))
        self.setGeometry(100, 100, 50, 50)
        self.init_ui()

    def init_ui(self):
        # LABELS FOR WIDGETS
        self.lb_name = QLabel("Name")
        self.lb_gender = QLabel("Gender")
        self.lb_age = QLabel("Age")
        self.lb_current_smoker = QLabel("Current Smoker")
        self.lb_cigs_per_day = QLabel("Cigarettes Per Day")
        self.lb_BP_meds = QLabel("BP Medication")
        self.lb_prevalent_stroke = QLabel("Prevalent Stroke")
        self.lb_prevalent_hyp = QLabel("Prevalent Hypertension")
        self.lb_diabetic = QLabel("Diabetic")
        self.lb_cholesterol = QLabel("Cholesterol")
        self.lb_sysBP = QLabel("Systolic BP")
        self.lb_diaBP = QLabel("Diastolic BP")
        self.lb_BMI = QLabel("BMI")
        self.lb_heart_rate = QLabel("Heart Rate")
        self.lb_glucose = QLabel("Glucose")
        self.lb_algorithm = QLabel("Algorithm:")
        self.lb_blank = QLabel(" ")

        # WIDGETS
        # Name
        self.wd_name = QLineEdit()
        self.wd_name.setPlaceholderText('Enter name')

        # Gender
        self.wd_gender = QComboBox()
        self.wd_gender.addItem("--SELECT--")
        self.wd_gender.addItem("Male")
        self.wd_gender.addItem("Female")
        self.wd_gender.currentTextChanged.connect(self.gender_selected)

        # Age
        self.wd_age = QLineEdit()
        self.wd_age.setMaxLength(2)
        self.wd_age.setPlaceholderText('Enter Age')

        # Cigarettes per day
        self.wd_cigs_per_day = QLineEdit()
        self.wd_cigs_per_day.setMaxLength(2)
        self.wd_cigs_per_day.setPlaceholderText('Enter cigarettes consumed per day')

        # Current Smoker
        self.wd_current_smoker = QCheckBox()
        self.wd_current_smoker.toggled.connect(self.checkbox_selected)

        # BP Medication
        self.wd_BP_meds = QCheckBox()
        self.wd_BP_meds.toggled.connect(self.checkbox_selected)

        # Prevalent smoker
        self.wd_prevalent_stroke = QCheckBox()
        self.wd_prevalent_stroke.toggled.connect(self.checkbox_selected)

        # Prevalent Hypertension
        self.wd_prevalent_hyp = QCheckBox()
        self.wd_prevalent_hyp.toggled.connect(self.checkbox_selected)

        # Diabetic
        self.wd_diabetic = QCheckBox()
        self.wd_diabetic.toggled.connect(self.checkbox_selected)

        # Cholesterol
        self.wd_cholesterol = QLineEdit()
        self.wd_cholesterol.setMaxLength(3)
        self.wd_cholesterol.setPlaceholderText('Enter Cholesterol')

        # Systolic Blood Pressure
        self.wd_sysBP = QLineEdit()
        self.wd_sysBP.setMaxLength(3)
        self.wd_sysBP.setPlaceholderText('Enter Systolic BP')

        # Diastolic Blood Pressure
        self.wd_diaBP = QLineEdit()
        self.wd_diaBP.setMaxLength(3)
        self.wd_diaBP.setPlaceholderText('Enter Diastolic BP')

        # Body Mass Index
        self.wd_BMI = QLineEdit()
        self.wd_BMI.setMaxLength(3)
        self.wd_BMI.setPlaceholderText('Enter Body Mass Index')

        # Heart Rate
        self.wd_heart_rate = QLineEdit()
        self.wd_heart_rate.setMaxLength(3)
        self.wd_heart_rate.setPlaceholderText('Enter Heart Rate')

        # Glucose
        self.wd_glucose = QLineEdit()
        self.wd_glucose.setMaxLength(3)
        self.wd_glucose.setPlaceholderText('Enter Glucose')

        # WIDGETS FOR ALGORITHMS
        # Support Vector Machine
        self.algo_SVM = QRadioButton("Support Vector Machine")
        self.algo_SVM.toggled.connect(self.radiobtn_selected)

        # K-Nearest Neighbours
        self.algo_KNN = QRadioButton("KNN")
        self.algo_KNN.toggled.connect(self.radiobtn_selected)

        # Logistic Regression
        self.algo_LR = QRadioButton("Logistic Regression")
        self.algo_LR.toggled.connect(self.radiobtn_selected)

        # Naive Bayes Theorem
        self.algo_NBT = QRadioButton("Naives Bayes Tree")
        self.algo_NBT.toggled.connect(self.radiobtn_selected)

        # Decision Tree
        self.algo_DT = QRadioButton("Decision Tree")

        # Random Forest
        self.algo_RF = QRadioButton("Random Forest")

        # ANN
        self.algo_ANN = QRadioButton("Artificial Neural Networks")

        # BUTTONS
        self.bn_check = QPushButton("Check Accuracy")
        self.bn_check.clicked.connect(self.accuracy_clicked)

        self.bn_submit = QPushButton("Submit")
        self.bn_submit.clicked.connect(self.submit_clicked)

        # LAYOUTS
        VBox_main = QVBoxLayout()
        VBox_up = QVBoxLayout()
        HBox_down = QHBoxLayout()
        VBox_left = QVBoxLayout()
        VBox_right = QVBoxLayout()
        HBox_button = QHBoxLayout()
        gridLayout_up = QGridLayout()
        gridLayout_result = QGridLayout()

        # ADDING WIDGETS TO LAYOUT
        gridLayout_up.addWidget(self.lb_name, 0, 0)
        gridLayout_up.addWidget(self.wd_name, 0, 1)
        gridLayout_up.addWidget(self.lb_gender, 1, 0)
        gridLayout_up.addWidget(self.wd_gender, 1, 1)
        gridLayout_up.addWidget(self.lb_age, 2, 0)
        gridLayout_up.addWidget(self.wd_age, 2, 1)
        gridLayout_up.addWidget(self.lb_current_smoker, 3, 0)
        gridLayout_up.addWidget(self.wd_current_smoker, 3, 1)
        gridLayout_up.addWidget(self.lb_cigs_per_day, 4, 0)
        gridLayout_up.addWidget(self.wd_cigs_per_day, 4, 1)
        gridLayout_up.addWidget(self.lb_BP_meds, 5, 0)
        gridLayout_up.addWidget(self.wd_BP_meds, 5, 1)
        gridLayout_up.addWidget(self.lb_prevalent_stroke, 6, 0)
        gridLayout_up.addWidget(self.wd_prevalent_stroke, 6, 1)
        gridLayout_up.addWidget(self.lb_prevalent_hyp, 7, 0)
        gridLayout_up.addWidget(self.wd_prevalent_hyp, 7, 1)
        gridLayout_up.addWidget(self.lb_diabetic, 8, 0)
        gridLayout_up.addWidget(self.wd_diabetic, 8, 1)
        gridLayout_up.addWidget(self.lb_cholesterol, 9, 0)
        gridLayout_up.addWidget(self.wd_cholesterol, 9, 1)
        gridLayout_up.addWidget(self.lb_sysBP, 10, 0)
        gridLayout_up.addWidget(self.wd_sysBP, 10, 1)
        gridLayout_up.addWidget(self.lb_diaBP, 11, 0)
        gridLayout_up.addWidget(self.wd_diaBP, 11, 1)
        gridLayout_up.addWidget(self.lb_BMI, 12, 0)
        gridLayout_up.addWidget(self.wd_BMI, 12, 1)
        gridLayout_up.addWidget(self.lb_heart_rate, 13, 0)
        gridLayout_up.addWidget(self.wd_heart_rate, 13, 1)
        gridLayout_up.addWidget(self.lb_glucose, 14, 0)
        gridLayout_up.addWidget(self.wd_glucose, 14, 1)
        gridLayout_up.addWidget(self.lb_blank, 15, 0)

        VBox_left.addWidget(self.lb_algorithm)
        VBox_left.addWidget(self.algo_SVM)
        VBox_left.addWidget(self.algo_KNN)
        VBox_left.addWidget(self.algo_LR)
        VBox_left.addWidget(self.algo_NBT)
        VBox_left.addWidget(self.algo_RF)
        VBox_left.addWidget(self.algo_DT)
        VBox_left.addWidget(self.algo_ANN)

        HBox_button.addWidget(self.bn_check)
        HBox_button.addWidget(self.bn_submit)

        # Setting Layout
        VBox_main.addLayout(VBox_up)
        VBox_main.addLayout(HBox_down)
        VBox_main.addLayout(HBox_button)
        HBox_down.addLayout(VBox_left)
        HBox_down.addLayout(VBox_right)
        VBox_right.addLayout(gridLayout_result)
        VBox_up.addLayout(gridLayout_up)

        self.setLayout(VBox_main)

        self.show()

    # FUNCTIONS FOR WIDGETS
    algo_number = 0

    def checkbox_selected(self):
        pass

    # wd_gender Combo Box
    def gender_selected(self):
        text = self.wd_gender.currentText()
        if text == 'Male':
            return 1
        elif text == 'Female':
            return 0
        else:
            return None

    def radiobtn_selected(self):
        if self.algo_SVM.isChecked():
            self.algo_number = 1

        elif self.algo_KNN.isChecked():
            self.algo_number = 2

        elif self.algo_LR.isChecked():
            self.algo_number = 3

        elif self.algo_NBT.isChecked():
            self.algo_number = 4

        elif self.algo_RF.isChecked():
            self.algo_number = 5

        elif self.algo_DT.isChecked():
            self.algo_number = 6

        elif self.algo_ANN.isChecked():
            self.algo_number = 7
        else:
            print("I have not programmed that path yet")

    def accuracy_clicked(self):

        if self.algo_number == 1:
            msg = support_vector_machine_algorithm()
            QMessageBox.about(self, "K-Support Vector Machine Accuracy Check", msg[1])

        if self.algo_number == 2:
            msg = knn_algorithm()
            QMessageBox.about(self, "KNN Accuracy Check", msg[1])

        if self.algo_number == 3:
            msg = logistic_regression_algorithm()
            QMessageBox.about(self, "Logistic Regression Accuracy Check", msg[1])

        if self.algo_number == 4:
            msg = naive_bayes_algorithm()
            QMessageBox.about(self, "Naives Bayes Accuracy Check", msg[1])

        if self.algo_number == 5:
            msg = random_forest_algorithm()
            QMessageBox.about(self, "Random Forest Accuracy Check", msg[1])

        if self.algo_number == 6:
            msg = decision_tree_algorithm()
            QMessageBox.about(self, "Decision Tree Accuracy Check", msg[1])

        if self.algo_number == 7:
            # msg = ann_algorithm()
            # QMessageBox.about(self, "ANN Accuracy Check", msg[1])
            print('do this')

    def submit_clicked(self):
        global sample

        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)

        # Adding data to the dict: 'sample'

        if self.gender_selected() is None:
            msgBox.setWindowTitle("Gender not selected!")
            msgBox.setInformativeText("Please select a gender!")
            msgBox.exec_()
        else:
            sample['col0'] = [self.gender_selected()]

        if self.wd_age.text() != '':
            sample['col1'] = [int(self.wd_age.text())]
        else:
            msgBox.setWindowTitle("Age not entered!")
            msgBox.setInformativeText("Please enter your age!")
            msgBox.exec_()

        if self.wd_current_smoker.isChecked():
            sample['col2'] = [1]
            if self.wd_cigs_per_day.text() != '' or self.wd_cigs_per_day.text() != '0':
                sample['col3'] = [self.wd_cigs_per_day.text()]
            else:
                msgBox.setWindowTitle("Invalid Input!")
                msgBox.setInformativeText("Please enter more than 0 number of cigarettes per day!")
                msgBox.exec_()
        else:
            sample['col2'] = [0]
            self.wd_cigs_per_day.setText('0')
            sample['col3'] = [self.wd_cigs_per_day.text()]

        if self.wd_BP_meds.isChecked():
            sample['col4'] = [1]
        else:
            sample['col4'] = [0]

        if self.wd_prevalent_stroke.isChecked():
            sample['col5'] = [1]
        else:
            sample['col5'] = [0]

        if self.wd_prevalent_hyp.isChecked():
            sample['col6'] = [1]
        else:
            sample['col6'] = [0]

        if self.wd_diabetic.isChecked():
            sample['col7'] = [1]
        else:
            sample['col7'] = [0]

        if self.wd_cholesterol.text() != '':
            sample['col8'] = [int(self.wd_cholesterol.text())]
        else:
            msgBox.setWindowTitle("Cholesterol not entered!")
            msgBox.setInformativeText("Please enter your cholesterol level!")
            msgBox.exec_()

        if self.wd_sysBP.text() != '':
            sample['col9'] = [int(self.wd_sysBP.text())]
        else:
            msgBox.setWindowTitle("Systolic BP not entered!")
            msgBox.setInformativeText("Please enter your Systolic Blood Pressure!")
            msgBox.exec_()

        if self.wd_diaBP.text() != '':
            sample['col10'] = [int(self.wd_diaBP.text())]
        else:
            msgBox.setWindowTitle("Cholesterol not entered!")
            msgBox.setInformativeText("Please enter your Diastolic Blood Pressure!")
            msgBox.exec_()

        if self.wd_BMI.text() != '':
            sample['col11'] = [float(self.wd_BMI.text())]
        else:
            msgBox.setWindowTitle("BMI not entered!")
            msgBox.setInformativeText("Please enter your Body Mass Index")
            msgBox.exec_()

        if self.wd_heart_rate.text() != '':
            sample['col12'] = [int(self.wd_heart_rate.text())]
        else:
            msgBox.setWindowTitle("Heart Rate not entered!")
            msgBox.setInformativeText("Please enter your Heart Rate!")
            msgBox.exec_()

        if self.wd_glucose.text() != '':
            sample['col13'] = [int(self.wd_glucose.text())]
        else:
            msgBox.setWindowTitle("Glucose not entered!")
            msgBox.setInformativeText("Please enter your glucose level!")
            msgBox.exec_()

        print('afterwards: ', sample)

        # Giving the prediction

        msg_good = QMessageBox()
        msg_good.setWindowTitle('Prediction Result')
        msg_good.setInformativeText('You are healthy.')
        msg_good.setIcon(QMessageBox.Information)
        msg_good.setWindowIcon(QtGui.QIcon('heart_healthy.png'))

        msg_bad = QMessageBox()
        msg_bad.setWindowTitle('Prediction Result')
        msg_bad.setInformativeText('You \'re at risk.')
        msg_bad.setIcon(QMessageBox.Critical)
        msg_bad.setDetailedText('visit us at healthtips.info')
        msg_bad.setWindowIcon(QtGui.QIcon('heart-rate-monitor.png'))

        if self.algo_number == 1:
            prediction = support_vector_machine_algorithm()

            if prediction[0] == 0:
                print('alive')
                msg_good.exec_()
            elif prediction[0] == 1:
                print('ded')
                msg_bad.exec_()

        if self.algo_number == 2:
            prediction = knn_algorithm()

            if prediction[0] == 0:
                print('alive')
                msg_good.exec_()
            elif prediction[0] == 1:
                print('ded')
                msg_bad.exec_()

        if self.algo_number == 3:
            prediction = logistic_regression_algorithm()

            if prediction[0] == 0:
                print('alive')

                msg_good.exec_()
            elif prediction[0] == 1:
                print('ded')
                msg_bad.exec_()

        if self.algo_number == 4:
            prediction = naive_bayes_algorithm()

            if prediction[0] == 0:
                print('alive')
                msg_good.exec_()
            elif prediction[0] == 1:
                print('ded')
                msg_bad.exec_()

        if self.algo_number == 5:
            prediction = random_forest_algorithm()

            if prediction[0] == 0:
                print('alive')
                msg_good.exec_()
            elif prediction[0] == 1:
                print('ded')
                msg_bad.exec_()

        if self.algo_number == 6:

            prediction = decision_tree_algorithm()

            if prediction[0] == 0:
                print('alive')
                msg_good.exec_()
            elif prediction[0] == 1:
                print('ded')
                msg_bad.exec_()

        if self.algo_number == 7:
            # msgBox = "Accuracy: " + str(ann_algorithm()) + "%"
            # QMessageBox.about(self, "ANN Accuracy Check", msgBox)
            print('do this')


'''
def ann_algorithm():
    
    # Importing the Keras libraries and packages

    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(activation="relu", input_dim=14, units=8, kernel_initializer="uniform"))

    # Adding the second hidden layer
    classifier.add(Dense(activation="relu", units=8, kernel_initializer="uniform"))

    # Adding the output layer
    classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))

    # Compiling the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size=10, epochs=100)

    # Part 3 - Making the predictions and evaluating the model

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)

    global y_re
    sample_re = pd.DataFrame(data=sample)
    sample_re = sc_X.transform(sample_re)
    y_re = classifier.predict(sample_re)
    print(y_re)

    a = int(y_re)
    b = 'Accuracy: ' +  str(sklearn.metrics.accuracy_score(y_test, y_pred) * 100) + "%"
    # Accuracy
    return [a, b]

'''


def random_forest_algorithm():
    # Fitting RFC to the Training set

    classifier = RandomForestClassifier(n_estimators=14, criterion='entropy', max_features=14, random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    global y_re
    sample_re = pd.DataFrame(data=sample)
    sample_re = sc_X.transform(sample_re)
    y_re = classifier.predict(sample_re)
    a = int(y_re)    
    y  = ("{0:.2f}".format(sklearn.metrics.accuracy_score(y_test, y_pred) * 100))    
    b = 'Accuracy: ' + str(y) + "%"
    # Accuracy and Prediction
    return [a, b]


def naive_bayes_algorithm():
    # Fitting Naive Bayes to the Training set

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    global y_re
    sample_re = pd.DataFrame(data=sample)
    sample_re = sc_X.transform(sample_re)
    y_re = classifier.predict(sample_re)
    a = int(y_re)    
    y  = ("{0:.2f}".format(sklearn.metrics.accuracy_score(y_test, y_pred) * 100))    
    b = 'Accuracy: ' + str(y) + "%"
    # Accuracy and Prediction
    return [a, b]

def logistic_regression_algorithm():
    # Fitting Logistic Regression to the Training set

    classifier = LogisticRegression(penalty='l2', solver='newton-cg', max_iter=50, random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    global y_re
    sample_re = pd.DataFrame(data=sample)
    sample_re = sc_X.transform(sample_re)
    y_re = classifier.predict(sample_re)
    a = int(y_re)    
    y  = ("{0:.2f}".format(sklearn.metrics.accuracy_score(y_test, y_pred) * 100))    
    b = 'Accuracy: ' + str(y) + "%"
    # Accuracy and Prediction
    return [a, b]
 

def knn_algorithm():
    # Fitting K-NN to the Training set

    classifier = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='ball_tree', metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Accuracy
    global y_re
    sample_re = pd.DataFrame(data=sample)
    sample_re = sc_X.transform(sample_re)
    y_re = classifier.predict(sample_re)
    a = int(y_re)    
    y  = ("{0:.2f}".format(sklearn.metrics.accuracy_score(y_test, y_pred) * 100))    
    b = 'Accuracy: ' + str(y) + "%"
    # Accuracy and Prediction
    return [a, b]


def support_vector_machine_algorithm():
    # Fitting Kernel SVM to the Training set

    classifier = SVC(C=1, kernel='rbf', gamma=0.3, random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Accuracy
    global y_re
    sample_re = pd.DataFrame(data=sample)
    sample_re = sc_X.transform(sample_re)
    y_re = classifier.predict(sample_re)
    a = int(y_re)    
    y  = ("{0:.2f}".format(sklearn.metrics.accuracy_score(y_test, y_pred) * 100))    
    b = 'Accuracy: ' + str(y) + "%"
    # Accuracy and Prediction
    return [a, b]


def decision_tree_algorithm():
    # Fitting Decision Tree to the Training set

    classifier = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=10, random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    global y_re
    sample_re = pd.DataFrame(data=sample)
    sample_re = sc_X.transform(sample_re)
    y_re = classifier.predict(sample_re)
    a = int(y_re)    
    y  = ("{0:.2f}".format(sklearn.metrics.accuracy_score(y_test, y_pred) * 100))    
    b = 'Accuracy: ' + str(y) + "%"
    # Accuracy and Prediction
    return [a, b]


app = QApplication(sys.argv)
a_window = Window()
sys.exit(app.exec_())
# app.exec_()
