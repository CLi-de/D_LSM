#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2023/3/27 17:38
# @File    : util.py
# @annotation

import numpy as np
import pandas as pd


def cal_measure(pred, y_test):
    TP = ((pred == 1) * (y_test == 1)).astype(int).sum()
    FP = ((pred == 1) * (y_test == 0)).astype(int).sum()
    FN = ((pred == 0) * (y_test == 1)).astype(int).sum()
    TN = ((pred == 0) * (y_test == 0)).astype(int).sum()
    # statistical measure
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F_measures = 2 * Precision * Recall / (Precision + Recall)
    print('Precision: %f' % Precision, '\nRecall: %f' % Recall, '\nF_measures: %f' % F_measures)


def pred_LSM(trained_model, xy, samples, name):
    """LSM prediction"""
    pred = trained_model.predict_proba(samples)
    data = np.hstack((xy, pred))
    data_df = pd.DataFrame(data)
    writer = pd.ExcelWriter('./tmp/' + name + '_prediction_HK.xlsx')
    data_df.to_excel(writer, 'page_1', float_format='%.5f')
    writer.close()
