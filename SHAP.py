#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2022/12/2 14:55
# @File    : SHAP.py
# @annotation

import tensorflow as tf
import xgboost
import shap
import warnings
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
from meta_learner import FLAGS
from modeling import Meta_learner

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# construct model
def init_weights(file):
    """读取DAS权参"""
    with tf.compat.v1.variable_scope('model'):  # get variable in 'model' scope, to reuse variables
        npzfile = np.load(file)
        weights = {}
        weights['w1'] = npzfile['arr_0']
        weights['b1'] = npzfile['arr_1']
        weights['w2'] = npzfile['arr_2']
        weights['b2'] = npzfile['arr_3']
        weights['w3'] = npzfile['arr_4']
        weights['b3'] = npzfile['arr_5']
        weights['w4'] = npzfile['arr_6']
        weights['b4'] = npzfile['arr_7']
    return weights


# define model.pred_prob() for shap.KernelExplainer(model, data)
def pred_prob(X_):
    with tf.compat.v1.variable_scope('model', reuse=True):
        return sess.run(tf.nn.softmax(model.forward(X_, model.weights, reuse=True)))


# read subtasks
def read_tasks(file):
    """获取tasks"""
    f = pd.ExcelFile(file)
    tasks = [[] for i in range(len(f.sheet_names))]
    k = 0
    for sheetname in f.sheet_names:
        # attr = pd.read_excel(file, usecols=[i for i in range(FLAGS.dim_input)], sheet_name=sheetname,
        #                      header=None).values.astype(np.float32)
        arr = pd.read_excel(file, sheet_name=sheetname,
                            header=None).values.astype(np.float32)
        tasks[k] = arr
        k = k + 1
    return tasks


print('construct model...')
tf.compat.v1.disable_eager_execution()
model = Meta_learner(FLAGS.dim_input, FLAGS.dim_output, test_num_updates=5)
input_tensors_input = (FLAGS.meta_batch_size, int(FLAGS.num_samples_each_task / 2), FLAGS.dim_input)
input_tensors_label = (FLAGS.meta_batch_size, int(FLAGS.num_samples_each_task / 2), FLAGS.dim_output)
model.construct_model(input_tensors_input=input_tensors_input, input_tensors_label=input_tensors_label,
                      prefix='metatrain_')

tasks = read_tasks('task_sampling/meta_task.xlsx')  # read meta_tasks from excel file

p_data = np.loadtxt('./data_src/p_samples.csv', dtype=str, delimiter=",", encoding='UTF-8-sig')
feature_names = p_data[0, :-4]

sess = tf.compat.v1.InteractiveSession()
init = tf.compat.v1.global_variables()  # optimizer里会有额外variable需要初始化
sess.run(tf.compat.v1.variables_initializer(var_list=init))

# SHAP for ith subtasks
for i in range(150, len(tasks), 10):
    model.weights = init_weights('./adapted_models/' + str(i) + 'th_model.npz')

    tmp_ = tasks[i]
    np.random.shuffle(tmp_)  # shuffle

    X = tmp_[:, :-1]  # 加载i行数据部分
    Y = tmp_[:, -1]  # 加载类别标签部分

    shap.initjs()
    # SHAP demo are using dataframe instead of nparray
    X_ = pd.DataFrame(X)  # convert np.array to pd.dataframe
    # x_test = pd.DataFrame(x_test)
    X_.columns = feature_names  # 添加特征名称
    # x_test.columns = feature_names

    # explainer = shap.KernelExplainer(pred_prob, shap.kmeans(x_train, 80))
    explainer = shap.KernelExplainer(pred_prob, shap.sample(X_, 50))
    shap_values = explainer.shap_values(X_, nsamples=50)  # shap_values
    # (_prob, n_samples, features)

    # shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], x_test.iloc[0, :], show=True, matplotlib=True)  # single feature
    shap.summary_plot(shap_values, X_, plot_type="bar", show=False)
    plt.savefig('tmp/bar_' + str(i) + '.pdf')
    plt.close()
    shap.summary_plot(shap_values[1], X_, plot_type="violin", show=False)
    plt.savefig('tmp/violin_' + str(i) + '.pdf')
    plt.close()
# shap.summary_plot(shap_values[1], x_test, plot_type="compact_dot")

# shap.force_plot(explainer.expected_value[1], shap_values[1], x_test, link="logit")

# shap.dependence_plot('DV', shap_values[1], x_test, interaction_index=None)
# shap.dependence_plot('SPI', shap_values[1], x_test, interaction_index='DV')
# shap.plots.beeswarm(shap_values[0])  # the beeswarm plot requires Explanation object as the `shap_values` argument
