import sys
from time import sleep
import requests
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtWidgets import QRadioButton, QScrollBar, QSplitter, QTableWidgetItem, QTableWidget, QComboBox, QVBoxLayout, QGridLayout, QDialog, QWidget, QPushButton, QApplication, QMainWindow, QAction, QMessageBox, QLabel, QTextEdit, QProgressBar, QLineEdit
from PyQt5.QtCore import QCoreApplication
from threading import Thread
import predict_knn
import predict_logistic_regression
import predict_random_forest
import predict_svm
import predict_keras


class Window(QDialog):
    def __init__(self):
        super().__init__()
        self.flag = 0
        self.method = 'Logistic Regression'
        self.chatTextField = QLineEdit(self)
        self.chatTextField.resize(480, 100)
        self.chatTextField.move(10, 350)
        self.btnSend = QPushButton("Send", self)
        self.btnSend.resize(480, 30)
        self.btnSendFont = self.btnSend.font()
        self.btnSendFont.setPointSize(15)
        self.btnSend.setFont(self.btnSendFont)
        self.btnSend.move(10, 460)
        self.btnSend.setStyleSheet("background-color: #F7CE16")
        self.btnSend.clicked.connect(self.send)
        self.RadioButton1 = QRadioButton('Logistic Regression')
        self.RadioButton1.setChecked(True)
        self.RadioButton2 = QRadioButton('Random Forest')
        self.RadioButton3 = QRadioButton('KNN')
        self.RadioButton4 = QRadioButton('SVM')
        self.RadioButton5 = QRadioButton('Keras Neural Network')
        self.RadioButton1.toggled.connect(self.changeMethod)
        self.RadioButton2.toggled.connect(self.changeMethod)
        self.RadioButton3.toggled.connect(self.changeMethod)
        self.RadioButton4.toggled.connect(self.changeMethod)
        self.RadioButton5.toggled.connect(self.changeMethod)
        
        self.chatBody = QVBoxLayout(self)
        # self.chatBody.addWidget(self.chatTextField)
        # self.chatBody.addWidget(self.btnSend)
        # self.chatWidget.setLayout(self.chatBody)
        splitter = QSplitter(QtCore.Qt.Vertical)

        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        # self.chatLayout=QVBoxLayout()
        # self.scrollBar=QScrollBar(self.chat)
        # self.chat.setLayout(self.chatLayout)

        splitter.addWidget(self.chat)
        splitter.addWidget(self.chatTextField)
        splitter.setSizes([300, 100])

        splitter2 = QSplitter(QtCore.Qt.Vertical)
        splitter2.addWidget(splitter)

        splitter2.addWidget(self.btnSend)
        splitter2.setSizes([100, 10])

        self.chatBody.addWidget(splitter2)
        self.chatBody.addWidget(self.RadioButton1)
        self.chatBody.addWidget(self.RadioButton2)
        self.chatBody.addWidget(self.RadioButton3)
        self.chatBody.addWidget(self.RadioButton4)
        self.chatBody.addWidget(self.RadioButton5)
        

        self.setWindowTitle("Chat Application")
        self.resize(600, 600)
        
    def changeMethod(self):
        radioBtn = self.sender()
        if(radioBtn.isChecked()):
            print('changed to ' + radioBtn.text())
            self.method = radioBtn.text()

    def send(self):
        text = self.chatTextField.text()
        spam = True
        if(self.method == 'Logistic Regression'):
            spam = predict_logistic_regression(text)
            print("the method was logistic")
        elif(self.method == 'KNN'):
            spam = predict_knn(text)
        elif(self.method == 'Random Forest'):
            spam = predict_random_forest(text)
            print("the method was random forest")
        elif(self.method == 'SVM'):
            spam = predict_svm(text)
        elif(self.method == 'Keras Neural Network'):
            spam = predict_keras(text)
            
        if spam:
            font = self.chat.font()
            font.setPointSize(13)
            self.chat.setFont(font)
            self.chat.textColor()
            # textFormatted='{:>80}'.format(text)
            redText = "<span style=\" font-size:8pt; font-weight:600; color:#ff0000;\" >"
            redText = redText + "the message you sent was reported as spam, it was not sent..."
            redText = redText + "</span>"
            self.chat.append(redText)
        else:
            font = self.chat.font()
            font.setPointSize(13)
            self.chat.setFont(font)
            # textFormatted='{:>80}'.format(text)
            self.chat.append('me: {}'.format(text))
            # global conn
            # conn.send(text.encode("utf-8"))
            self.chatTextField.setText("")
            requests.post('http://localhost:5000/sendMessage', {"username": sys.argv[1],
                                                                "reciever": sys.argv[2],
                                                                "content": text})


class Client(Thread):
    def __init__(self, window):
        Thread.__init__(self)
        self.window = window

    def run(self):
        while True:
            # print("Multithreaded Python server : Waiting for connections from TCP clients...")
            # global conn
            # (conn, (ip,port)) = tcpServer.accept()
            # newthread = ClientThread(ip,port,window)
            # newthread.start()
            # threads.append(newthread)
            r = requests.post(
                'http://localhost:5000/recieveMessages', {"username": sys.argv[1]})
            print(r.json())
            for i in r.json()['messages']:
                self.window.chat.append('{user}: {content}'.format(
                    user=i['from'], content=i['content']))
            # if len(r.json.messages) != 0:


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = Window()
    serverThread = Client(window)
    serverThread.start()
    window.exec()

    sys.exit(app.exec_())
