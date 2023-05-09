#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2023/5/8 15:23
# @File    : Xgboost.py
# @annotation

"""
Overall performance evaluation by XGB
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

import xgboost
import shap

from utils import cal_measure, pred_LSM


def Xgboost_(x_train, y_train, x_test, y_test, f_names, savefig_name):
    """predict and test"""
    # print('start Xgboost evaluation...')
    model = xgboost.XGBClassifier().fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)
    # 训练精度
    print('train_Accuracy: %f' % accuracy_score(y_train, pred_train))
    # 测试精度
    print('test_Accuracy: %f' % accuracy_score(y_test, pred_test))
    # pred1 = clf2.predict_proba() # 预测类别概率
    cal_measure(pred_test, y_test)
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
    shap_values = explainer(x_train[:1000])

    def font_setting(plt, xlabel=None):
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 14,
                 }
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 14,
                 }
        plt.yticks(fontsize=10, font=font1)
        plt.xlabel(xlabel, fontdict=font2)

    def save_pic(savename, xlabel=None):
        font_setting(plt, xlabel)
        plt.tight_layout()  # keep labels within frame
        plt.savefig(savename)
        plt.close()

    '''success'''
    # # waterfall
    # shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    # save_pic('tmp/waterfall' + savefig_name + '.pdf',
    #          'Contribution of various LIFs to output in a single sample')

    # bar
    shap.plots.bar(shap_values, max_display=15, show=False)
    save_pic('tmp/bar' + savefig_name + '.pdf', 'LIF importance')

    # violin
    shap.summary_plot(shap_values, max_display=15, show=False, plot_type='violin')
    save_pic('tmp/violin' + savefig_name + '.pdf', 'SHAP values')

    # scatter
    shap.plots.scatter(shap_values, show=False, color='blue')
    # save_pic('tmp/scatter' + savefig_name + '.pdf')
    # font_setting(plt, xlabel)
    plt.tight_layout()  # keep labels within frame
    plt.savefig('tmp/scatter' + savefig_name + '.pdf')
    plt.close()
    # heatmap
    shap.plots.heatmap(shap_values, max_display=15, show=False)
    save_pic('tmp/heatmap' + savefig_name + '.pdf', 'Non-landslide/Landslide samples')

    '''failures'''
    # # force
    # shap.plots.force(shap_values[0], show=False)
    # font_setting(plt)
    # plt.tight_layout()
    # plt.savefig('tmp/force' + savefig_name + '.pdf')
    # plt.close()
    #
    # # forces
    # shap.plots.force(shap_values)
    # shap.plots.force(shap_values[0], show=False)
    # font_setting(plt)
    # plt.tight_layout()  #
    # plt.savefig('tmp/forces' + savefig_name + '.pdf')
    # plt.close()

    # shap.plots.scatter(shap_values[:, "RM"], color=shap_values)



def feature_normalization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma, mu, sigma


if __name__ == "__main__":
    """input data"""
    # positive samples
    p_data = np.loadtxt('./data_src/p_samples.csv', dtype=str, delimiter=",", encoding='UTF-8-sig')
    # p_samples = np.hstack((p_data[1:, :-3], p_data[1:, -1].reshape(-1, 1))).astype(np.float32)
    p_samples = p_data[1:, :-5].astype(np.float32)
    f_names = p_data[0, :-6].astype(str)

    # negative samples
    n_data = np.loadtxt('./data_src/n_samples.csv', dtype=str, delimiter=",", encoding='UTF-8')
    # n_samples = np.hstack((n_data[1:, :-3], n_data[1:, -1].reshape(-1, 1))).astype(np.float32)
    n_samples = n_data[1:, :-3].astype(np.float32)

    samples = np.vstack((p_samples, n_samples))
    samples_f, mean, std = feature_normalization(samples[:, :-1])
    samples = np.hstack((samples_f, samples[:, -1].reshape(-1, 1)))

    x_train, x_test, y_train, y_test = train_test_split(samples[:, :-1], samples[:, -1], test_size=0.2,
                                                        shuffle=True)
    Xgboost_(x_train, y_train, x_test, y_test, f_names, '_XGB')


    print('done Xgboost-based LSM prediction! \n')
