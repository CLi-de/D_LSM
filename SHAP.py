#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2023/3/2 14:55
# @File    : SHAP.py
# @annotation

import tensorflow as tf
import shap
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from meta_learner import FLAGS
from modeling import Meta_learner

from shap.plots import _waterfall, _scatter, _bar

warnings.filterwarnings("ignore")


# construct model
def init_weights(file):
    """读取DAS权参"""
    # with tf.compat.v1.variable_scope('model'):  # get variable in 'model' scope, to reuse variables
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


# set plotting font
def font_setting(plt, xlabel=None):
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


def save_pic(savename, xlabel=None):
    font_setting(plt, xlabel)
    plt.tight_layout()  # keep labels within frame
    plt.savefig(savename)
    plt.close()


print('\n read meta-tasks from file...')
tasks = read_tasks('task_sampling/meta_task.xlsx')  # read meta_tasks from excel file
# tasks = supplement_samples(tasks)
years = [str(1992 + i) for i in range(28)]

p_data = np.loadtxt('./data_src/p_samples.csv', dtype=str, delimiter=",", encoding='UTF-8-sig')
feature_names = p_data[0, :-6]

print('\n construct model...')
tf.compat.v1.disable_eager_execution()
model = Meta_learner(FLAGS.dim_input, FLAGS.dim_output, test_num_updates=5)
input_tensors_input = (FLAGS.meta_batch_size, int(FLAGS.num_samples_each_task / 2), FLAGS.dim_input)
input_tensors_label = (FLAGS.meta_batch_size, int(FLAGS.num_samples_each_task / 2), FLAGS.dim_output)
model.construct_model(input_tensors_input=input_tensors_input, input_tensors_label=input_tensors_label,
                      prefix='metatrain_')

sess = tf.compat.v1.InteractiveSession()
init = tf.compat.v1.global_variables()  # optimizer里会有额外variable需要初始化
sess.run(tf.compat.v1.variables_initializer(var_list=init))

# SHAP for ith subtasks(TODO: not enough memory)
for i in range(2, len(tasks), 3):
    if tasks[i].shape[0] > 0:
        model.weights = init_weights('./adapted_models/' + str(i) + 'th_model.npz')

        print('\n shap_round_' + str(i))
        shap.initjs()
        # SHAP demo are using dataframe instead of nparray
        X_ = pd.DataFrame(tasks[i][:, :-1])  # convert np.array to pd.dataframe
        X_.columns = feature_names  # 添加特征名称
        X_ = X_.iloc[:100, :]

        # explainer = shap.KernelExplainer(pred_prob, shap.kmeans(x_train, 80))
        explainer = shap.KernelExplainer(pred_prob, X_)
        shap_values = explainer.shap_values(X_, nsamples=100)  # shap_values

        '''local (for each sample)'''
        # waterfall
        num_samples = 4
        if tasks[i].shape[0] < 4:
            num_samples = tasks[i].shape[0]
        else:
            for j in range(num_samples):
                _waterfall.waterfall_legacy(explainer.expected_value[1], shap_values[1][j], feature_names=feature_names,
                                            max_display=7, show=False)  # label = 1 (landslide)
                save_pic('tmp/waterfall' + str(i) + '_' + str(j) + '.pdf',
                         'Contribution of different features to output in a single sample')

        # # force plot
        # shap.force_plot(base_value=explainer.expected_value[1], shap_values=shap_values[1][0], features=X_.iloc[0],
        #                 matplotlib=True, show=False)
        # save_pic('tmp/force_plot' + str(i) + '.pdf')

        '''global (for mulyiple samples)'''
        # bar
        # shap.summary_plot(shap_values[1], X_, plot_type="bar", color='r', show=False)
        shap.summary_plot(shap_values, X_, plot_type="bar", show=False, class_names=['landslide', 'non-landslide'],
                          title=years[i])

        save_pic('tmp/bar' + str(i) + '.pdf', 'LIF importance (' + years[i] + ')')

        # violin
        shap.summary_plot(shap_values[1], features=X_, plot_type="dot", show=False, max_display=15,
                          title=years[i])  # summary points
        # shap.summary_plot(shap_values[1], X_, plot_type="violin", show=False, max_display=15)
        save_pic('tmp/violin' + str(i) + '.pdf', 'LIF impact on model output (' + years[i] + ')')

        # scatter (interdependence of two features, in supplemental materials)
        # for name in feature_names.tolist():
        #     _scatter.dependence_legacy(name, shap_values[1], features=X_, show=False, interaction_index=None)
        #     save_pic('tmp/scatter' + str(i) + '_' + name + '.pdf')

print('\n finish SHAP!')
