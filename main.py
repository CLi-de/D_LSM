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


def Xgboost_(x_train, y_train, x_test, y_test, f_names, savefig_name):
    """predict and test"""
    print('start Xgboost evaluation...')
    model = xgboost.XGBRegressor().fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
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
    # SHAP_(model.predict_proba, x_train, x_test, f_names)
    shap.initjs()
    # SHAP demo are using dataframe instead of nparray
    x_train = pd.DataFrame(x_train)  # 将numpy的array数组x_test转为dataframe格式。
    x_test = pd.DataFrame(x_test)
    x_train.columns = f_names  # 添加特征名称
    x_test.columns = f_names

    explainer = shap.Explainer(model)
    shap_values = explainer(x_train)

    def font_setting(plt, xlabel):
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 14,
                 }
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 18,
                 }
        plt.yticks(fontsize=10, font=font1)
        plt.xlabel(xlabel, fontdict=font2)

    # shap.plots.waterfall(shap_values[0], show=False)
    # plt.savefig('tmp/waterfall_HK.pdf')
    # plt.close()

    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    font_setting(plt, "impact on model output")
    plt.tight_layout()  # keep labels within frame
    plt.savefig('tmp/beeswarm' + savefig_name + '.pdf')
    plt.close()

    # bar plotting
    shap.plots.bar(shap_values, max_display=15, show_data=False, show=False)
    font_setting(plt, "LIF importance")
    plt.tight_layout()  #
    plt.savefig('tmp/bar' + savefig_name + '.pdf')
    plt.close()

    # shap.plots.force(shap_values[0])
    # shap.plots.force(shap_values)
    # shap.plots.scatter(shap_values[:, "RM"], color=shap_values)
    return model


if __name__ == "__main__":
    """input data"""
    # positive samples
    p_data = np.loadtxt('./data_src/p_samples.csv', dtype=str, delimiter=",", encoding='UTF-8-sig')
    p_samples = np.hstack((p_data[1:, :-3], p_data[1:, -1].reshape(-1, 1))).astype(np.float32)
    f_names = p_data[0, :-3].astype(str)

    # negative samples
    n_data = np.loadtxt('./data_src/n_samples.csv', dtype=str, delimiter=",", encoding='UTF-8')
    n_samples = np.hstack((n_data[1:, :-3], n_data[1:, -1].reshape(-1, 1))).astype(np.float32)

    # 样本集 （按时间分三段）
    np.random.shuffle(n_samples)  # shuffle n_samples
    samples1 = np.vstack((p_samples[4696:, :], n_samples[:1200, :]))  # 1964-1989 (1420)
    samples2 = np.vstack((p_samples[2611:4696, :], n_samples[:1800, :]))  # 1990-2007 (2085)
    samples3 = np.vstack((p_samples[:2611, :], n_samples))  # 2008-2019 (2611)


    def Xgboost(samples, filename):
        x_train, x_test, y_train, y_test = train_test_split(samples[:, :-1], samples[:, -1], test_size=0.2,
                                                            shuffle=True)
        Xgboost_(x_train, y_train, x_test, y_test, f_names, filename)


    print("1964-1989: evaluating...")
    Xgboost(samples1, '1')
    print("1990-2007: evaluating...")
    Xgboost(samples2, '2')
    print("2008-2019: evaluating...")
    Xgboost(samples3, '3')

    # # grid features
    # grid_f = np.loadtxt('./data_src/grid_samples_HK.csv', dtype=str, delimiter=",", encoding='UTF-8')
    # samples_f = grid_f[1:, :-2].astype(np.float32)
    # xy = grid_f[1:, -2:].astype(np.float32)
    # samples_f = samples_f / samples_f.max(axis=0)

    # Xgboost-based

    # pred_LSM(model_rf, xy, samples_f, 'RF')
    # print('done RF-based LSM prediction! \n')
