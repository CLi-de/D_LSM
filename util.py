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


def tasksbatch_generator(data, batch_size, num_samples, dim_input, dim_output):
    """generate batch tasks"""
    init_inputs = np.zeros([batch_size, num_samples, dim_input], dtype=np.float32)
    labels = np.zeros([batch_size, num_samples, dim_output], dtype=np.float32)

    np.random.shuffle(data)
    start_index = np.random.randint(0, len(data) - batch_size)
    batch_tasks = data[start_index:(start_index + batch_size)]

    cnt_sample = []
    for i in range(batch_size):
        cnt_sample.append(len(batch_tasks[i]))

    for i in range(batch_size):
        np.random.shuffle(batch_tasks[i])  # shuffle samples in each task
        start_index1 = np.random.randint(0, len(batch_tasks[i]) - num_samples)
        task_samples = batch_tasks[i][start_index1:(start_index1 + num_samples)]
        for j in range(num_samples):
            init_inputs[i][j] = task_samples[j][0]
            if task_samples[j][1] == 1:
                labels[i][j][0] = 1  # 滑坡
            else:
                labels[i][j][1] = 1  # 非滑坡
    return init_inputs, labels, np.array(cnt_sample).astype(np.float32)


def batch_generator(one_task, dim_input, dim_output, batch_size):
    """generate samples from one tasks"""
    np.random.shuffle(one_task)
    batch_ = one_task[:batch_size]
    init_inputs = np.zeros([batch_size, dim_input], dtype=np.float32)
    labels = np.zeros([batch_size, dim_output], dtype=np.float32)
    for i in range(batch_size):
        init_inputs[i] = batch_[i][0]
        if batch_[i][1] == 1:
            labels[i][0] = 1
        else:
            labels[i][1] = 1
    return init_inputs, labels
