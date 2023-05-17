#!/usr/bin/env pytho
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2023/5/17 14:32
# @File    : comparison.py
# @annotation

import numpy as np

from sklearn import metrics, svm

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

from utils import cal_measure
import warnings

warnings.filterwarnings("ignore")


def SVM_(x_train, y_train, x_test, y_test):
    """predict and test"""
    print('start SVM evaluation...')
    model = svm.SVC(C=1, kernel='rbf', gamma=1 / (2 * x_train.var()), decision_function_shape='ovr', probability=True)
    # clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    print('train accuracy:' + str(metrics.accuracy_score(y_train, pred_train)))
    pred_test = model.predict(x_test)
    print('test accuracy:' + str(metrics.accuracy_score(y_test, pred_test)))
    # Precision, Recall, F1-score
    cal_measure(pred_test, y_test)

    return model


# can be deprecated
def ANN_(x_train, y_train, x_test, y_test):
    """predict and test"""
    print('start ANN evaluation...')
    model = MLPClassifier(hidden_layer_sizes=(32, 32, 16), activation='relu', solver='adam', alpha=0.01,
                          batch_size=32, max_iter=3000)
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    print('Train Accuracy: %f' % accuracy_score(y_train, pred_train))
    pred_test = model.predict(x_test)
    print('Test Accuracy: %f' % accuracy_score(y_test, pred_test))
    # Precision, Recall, F1-score
    cal_measure(pred_test, y_test)
    kappa_value = cohen_kappa_score(pred_test, y_test)
    print('Cohen_Kappa: %f' % kappa_value)

    return model


def RF_(x_train, y_train, x_test, y_test):
    """predict and test"""
    print('start RF evaluation...')
    model = RandomForestClassifier(n_estimators=200, max_depth=None)

    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    # 训练精度
    print('train_Accuracy: %f' % accuracy_score(y_train, pred_train))
    # 测试精度
    print('test_Accuracy: %f' % accuracy_score(y_test, pred_test))
    # pred1 = clf2.predict_proba() # 预测类别概率
    cal_measure(pred_test, y_test)
    kappa_value = cohen_kappa_score(pred_test, y_test)
    print('Cohen_Kappa: %f' % kappa_value)

    return model


def read_f_l_csv(file):
    tmp = np.loadtxt(file, dtype=str, delimiter=",", encoding='UTF-8')
    features = tmp[1:, :-2].astype(np.float32)
    features = features / features.max(axis=0)
    label = tmp[1:, -1].astype(np.float32)
    return features, label


if __name__ == "__main__":
    x, y = read_f_l_csv('data_src/samples.csv')
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.75, test_size=.05, shuffle=True)

    """evaluate and save LSM result"""
    # SVM-based
    model_svm = SVM_(x_train, y_train, x_test, y_test)
    print('done SVM-based LSM prediction! \n')

    # MLP_based
    model_mlp = ANN_(x_train, y_train, x_test, y_test)
    print('done MLP-based LSM prediction! \n')

    # RF-based
    model_rf = RF_(x_train, y_train, x_test, y_test)

    print('done RF-based LSM prediction! \n')
