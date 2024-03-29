#!/usr/bin/env pytho
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2023/4/11 16:02
# @File    : adaptation.py
# @annotation

import numpy as np
import tensorflow as tf
import pandas as pd

from meta_learner import FLAGS
from modeling import Meta_learner
import os

from utils import read_tasks, batch_generator

'''calculate mean and std for feature_normalization'''
p_data = np.loadtxt('./data_src/p_samples.csv', dtype=str, delimiter=",", encoding='UTF-8-sig')
p_samples = p_data[1:, :-5].astype(np.float32)
n_data = np.loadtxt('./data_src/n_samples.csv', dtype=str, delimiter=",", encoding='UTF-8-sig')
n_samples = n_data[1:, :-3].astype(np.float32)
mean = np.mean(np.vstack((p_samples, n_samples))[:, :-1], axis=0)
std = np.std(np.vstack((p_samples, n_samples))[:, :-1], axis=0)

'''construct model'''
tf.compat.v1.disable_eager_execution()
model = Meta_learner(FLAGS.dim_input, FLAGS.dim_output, test_num_updates=5)
input_tensors_input = (FLAGS.meta_batch_size, int(FLAGS.num_samples_each_task / 2), FLAGS.dim_input)
input_tensors_label = (FLAGS.meta_batch_size, int(FLAGS.num_samples_each_task / 2), FLAGS.dim_output)
model.construct_model(input_tensors_input=input_tensors_input, input_tensors_label=input_tensors_label,
                      prefix='metatrain_')

'''path of meta-learned model'''
exp_string = '.mbs' + str(FLAGS.meta_batch_size) + '.nset' + str(FLAGS.num_samples_each_task) \
             + '.nu' + str(FLAGS.test_update_batch_size) + '.in_lr' + str(FLAGS.update_lr) \
             + '.meta_lr' + str(FLAGS.meta_lr) + '.iter' + str(FLAGS.metatrain_iterations)

'''restoring from meta-trained model'''
saver = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES))
sess = tf.compat.v1.InteractiveSession()
init = tf.compat.v1.global_variables()  # optimizer里会有额外variable需要初始化
sess.run(tf.compat.v1.variables_initializer(var_list=init))
model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
if model_file:
    print("Restoring model weights from " + model_file)
    saver.restore(sess, model_file)  # 以model_file初始化sess中图
else:
    print("\nno meta-learned model found!")

'''Adaptation and predict'''
if not os.path.exists('./task_sampling/meta_task.xlsx'):
    pass
else:
    print('\nmodel adaptation and LSM prediction...')
    meta_tasks = read_tasks(FLAGS.dim_input, 'task_sampling/meta_task.xlsx')  # TODO: too slow reading

'''for meta-tasks with too few samples'''
for i in range(len(meta_tasks)):
    len_ = len(meta_tasks[i])
    if len_ < 8:
        # 从相邻年份补充样本
        if i - 1 >= 0:
            meta_tasks[i].extend(meta_tasks[i - 1][:8 - len_])
        else:
            meta_tasks[i].extend(meta_tasks[i + 1][:8 - len_])

for i in range(len(meta_tasks)):
    # np.random.shuffle(meta_tasks[i])
    with tf.compat.v1.variable_scope('model', reuse=True):  # Variable reuse in np.normalize()
        # train_ = meta_tasks[i][:int(len(meta_tasks[i]) / 2)]
        train_ = meta_tasks[i]  # all samples in a certain meta_task can be used for adaptation
        batch_size = FLAGS.test_update_batch_size
        fast_weights = model.weights
        for j in range(FLAGS.num_updates):
            inputa, labela = batch_generator(train_, FLAGS.dim_input, FLAGS.dim_output, batch_size)
            loss = model.loss_func(model.forward(inputa, fast_weights, reuse=True), labela)
            grads = tf.gradients(ys=loss, xs=list(fast_weights.values()))
            gradients = dict(zip(fast_weights.keys(), grads))
            fast_weights = dict(zip(fast_weights.keys(),
                                    [fast_weights[key] - model.update_lr * gradients[key] for key in
                                     fast_weights.keys()]))

        """save model parameters for each year"""
        adapted_weights = sess.run(fast_weights)
        np.savez('adapted_models/' + str(i) + 'th_model',
                 adapted_weights['w1'], adapted_weights['b1'],
                 adapted_weights['w2'], adapted_weights['b2'],
                 adapted_weights['w3'], adapted_weights['b3'],
                 adapted_weights['w4'], adapted_weights['b4'])

        """predict and save LSM result for 1999, 2008, 2017"""
        if i == 3 or i == 16 or i == 25:
            '''load grid data'''
            samples = np.loadtxt('./data_sup/grid_samples_' + str(1992 + i) + '.csv', dtype=str, delimiter=",",
                                 encoding='UTF-8-sig')
            f = samples[1:, :-2].astype(np.float32)
            xy = samples[1:, -2:].astype(np.float32)

            f = (f - mean) / std  # normalization

            pred = model.forward(f, fast_weights, reuse=True)
            pred = sess.run(tf.nn.softmax(pred))
            arr = np.hstack((xy, pred))

            writer = pd.ExcelWriter('tmp/' + str(i) + 'th_LSM.xlsx')
            data_df = pd.DataFrame(arr)
            data_df.to_excel(writer)
            writer.close()

print("\n finished!")
