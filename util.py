#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2023/3/27 17:38
# @File    : util.py
# @annotation

import numpy as np
import pandas as pd

import tensorflow as tf
import tf_slim as slim


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
    writer = pd.ExcelWriter('./tmp/' + name + '_prediction.xlsx')
    data_df.to_excel(writer, 'page_1', float_format='%.5f')
    writer.close()


def normalize(inp, activation, reuse, scope, norm):
    if norm == 'batch_norm':
        return slim.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif norm == 'layer_norm':
        return slim.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp


def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(input_tensor=tf.square(pred - label))


def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / tf.cast(tf.shape(input=label)[0],
                                                                                        dtype=tf.float32)  # 注意归一
