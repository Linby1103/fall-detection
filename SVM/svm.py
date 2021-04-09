# # -*- coding:utf-8 -*-
# """
# @author:Lisa
# @file:svm_Iris.py
# @func:Use SVM to achieve Iris flower classification
# @time:2018/5/30 0030上午 9:58
# """
# from sklearn import svm
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# from loadset import SVMDataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.externals import *
#
# # define converts(字典)
# def Iris_label(s):
#     it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
#     return it[s]
#
#
# # 1.读取数据集
# # path = 'F:/Python_Project/SVM/data/Iris.data'
# # data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: Iris_label})
# # converters={4:Iris_label}中“4”指的是第5列：将第5列的str转化为label(number)
# # print(data.shape)
#
# # 2.划分数据与标签
# traindataloader=SVMDataLoader('/mnt/workspace/code/CPPYOLO/build/fall.txt')
# testdataloader=SVMDataLoader('/mnt/workspace/code/CPPYOLO/build/test.txt')
#
# # testdataloader=SVMDataLoader('/mnt/workspace/code/CPPYOLO/build/train.txt')
# x,y=traindataloader.loaddatafromtxt()
# train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.6,
#                                                                   test_size=0.4)  # sklearn.model_selection.
# # print(train_data.shape)
#
# # 3.训练svm分类器
# classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovo')  # ovr:一对多策略
# classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先
#
#
# tx,ty=testdataloader.loaddatafromtxt()
# output=classifier.predict(tx)
# totalres=len(output)
# counter=0
# for index,i in enumerate(output):
#     if ty[index]==i:
#         counter=counter+1
#     print("prediect :%d   label:  %d\n",i,ty[index])
#
#
# print("Total sample %d , TP %d p %f\n",totalres,counter,counter/totalres)
# # 4.计算svc分类器的准确率
# # print("训练集：", classifier.score(train_data, train_label))
# # print("测试集：", classifier.score(test_data, test_label))
#
#

import numpy as np
from sklearn import datasets
from sklearn import model_selection as ms
from loadset import SVMDataLoader
from sklearn.metrics import confusion_matrix
import cv2

traindataloader=SVMDataLoader('/home/workdir/code/HI3559Av100/ncnn_yolov5/build/fall.txt')
testdataloader=SVMDataLoader('/home/workdir/code/HI3559Av100/ncnn_yolov5/build/test.txt')
x,y=traindataloader.loaddatafromtxt()
x = x.astype(np.float32)
X_train, X_test, y_train, y_test = ms.train_test_split(
    x, y, test_size=0.2, random_state=42
)

svm = cv2.ml.SVM_create()
svm.setKernel(cv2.ml.SVM_RBF)


'''train'''
y_train = y_train.reshape(-1, 1)
# print(y_train)
svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
svm.save("svmtest.mat")
print("Done\n")


'''开始预测'''
svm2 = cv2.ml.SVM_load("svmtest.mat")
tx,ty=testdataloader.loaddatafromtxt()
tx=tx.astype(np.float32)
_, y_pred = svm2.predict(tx)

totalres=len(y_pred)
counter=0
for index,i in enumerate(y_pred):
    if int(i[0])==ty[index]:
        counter = counter + 1
    print("predict :{}   label :{}\n".format(i[0],ty[index]))


'''用scikit-learn的metrics模块计算准确率'''
tn, fp, fn, tp =confusion_matrix(ty, y_pred).ravel()
precision = tp / (tp+fp)  # 查准率
recall = tp / (tp+fn)  # 查全率
print("recall :%f   precision   :%f"%(recall,precision))





