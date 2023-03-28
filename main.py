#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2023/3/27 17:37
# @File    : main.py
# @annotation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

import xgboost
import shap

from util import cal_measure
from util import pred_LSM




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

    # SHAP
    print('SHAP...')
    # TODO: SHAP for RF
    # SHAP_(model.predict_proba, x_train, x_test, f_names)
    shap.initjs()
    explainer = shap.Explainer(model)
    shap_values = explainer(x_train)
    shap.plots.bar(shap_values[:100, :, 0])  # shap_values(n_samples, features, _prob)
    # 参考meta_LSM中SVM
    return model


def Xgboost_(x_train, y_train, x_test, y_test, f_names):
    """predict and test"""
    print('start RF evaluation...')
    model = xgboost.XGBRegressor().fit(x_train, y_train)

    # pred_train = model.predict(x_train)
    # pred_test = model.predict(x_test)
    # # 训练精度
    # print('train_Accuracy: %f' % accuracy_score(y_train, pred_train))
    # # 测试精度
    # print('test_Accuracy: %f' % accuracy_score(y_test, pred_test))
    # # pred1 = clf2.predict_proba() # 预测类别概率
    # cal_measure(pred_test, y_test)
    # kappa_value = cohen_kappa_score(pred_test, y_test)
    # print('Cohen_Kappa: %f' % kappa_value)

    # SHAP
    print('SHAP...')
    # TODO: SHAP for RF
    # SHAP_(model.predict_proba, x_train, x_test, f_names)
    shap.initjs()
    # SHAP demo are using dataframe instead of nparray
    x_train = pd.DataFrame(x_train)  # 将numpy的array数组x_test转为dataframe格式。
    x_test = pd.DataFrame(x_test)
    x_train.columns = f_names  # 添加特征名称
    x_test.columns = f_names

    explainer = shap.Explainer(model)
    shap_values = explainer(x_train)
    # shap.plots.bar(shap_values[:100, :, 0])  # shap_values(n_samples, features, _prob)
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig('tmp/waterfall_HK.pdf')
    # plt.close()
    # shap.plots.force(shap_values[0])
    # shap.plots.force(shap_values)
    # shap.plots.scatter(shap_values[:, "RM"], color=shap_values)
    shap.plots.beeswarm(shap_values, show=False)
    plt.savefig('tmp/beeswarm.pdf')
    # plt.close()
    shap.plots.bar(shap_values, show=False)
    plt.savefig('tmp/bar.pdf')
    # plt.close()
    return model


if __name__ == "__main__":
    """input data"""
    # positive samples
    p_data = np.loadtxt('./data_src/p_samples.csv', dtype=str, delimiter=",", encoding='UTF-8')
    f_names = p_data[0, :-3].astype(str)
    p_samples = np.hstack((p_data[1:, :-3], p_data[1:, -1].reshape(-1, 1))).astype(np.float32)
    # negative samples
    n_data = np.loadtxt('./data_src/n_samples.csv', dtype=str, delimiter=",", encoding='UTF-8')
    n_samples = np.hstack((n_data[1:, :-3], n_data[1:, -1].reshape(-1, 1))).astype(np.float32)

    # 训练集
    samples = np.vstack((p_samples[:2611, :], n_samples))
    x_train, x_test, y_train, y_test = train_test_split(samples[:, :-1], samples[:, -1], test_size=0.3, shuffle=True)

    # np.random.shuffle(samples)
    # # grid features
    # grid_f = np.loadtxt('./data_src/grid_samples_HK.csv', dtype=str, delimiter=",", encoding='UTF-8')
    # samples_f = grid_f[1:, :-2].astype(np.float32)
    # xy = grid_f[1:, -2:].astype(np.float32)
    # samples_f = samples_f / samples_f.max(axis=0)

    # RF-based
    model_rf = Xgboost_(x_train, y_train, x_test, y_test, f_names)
    # pred_LSM(model_rf, xy, samples_f, 'RF')
    # print('done RF-based LSM prediction! \n')
