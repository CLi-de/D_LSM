#!/usr/bin/env pytho
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2023/4/11 14:32
# @File    : meta_learner.py
# @annotation

import numpy as np
import tensorflow as tf
import pandas as pd
from modeling import Meta_learner
from tensorflow.python.platform import flags
from utils import cal_measure, tasksbatch_generator, batch_generator, feature_normalization, save_tasks, \
    read_tasks
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

import warnings
import os

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = flags.FLAGS

"""for task sampling"""
flags.DEFINE_float('M', 500, 'determine how distance influence the segmentation')
flags.DEFINE_integer('K', 512, 'number of superpixels')
flags.DEFINE_integer('loop', 5, 'number of SLIC iterations')
flags.DEFINE_string('str_region', 'HK', 'the study area')
flags.DEFINE_string('sample_pts', './src_data/samples_HK.csv', 'path to (non)/landslide samples')
flags.DEFINE_string('Ts_pts', './src_data/Ts_HK.csv', 'path to Ts samples')

"""for meta-train"""
flags.DEFINE_string('basemodel', 'MLP', 'MLP: no unsupervised pretraining; DAS: pretraining with DAS')
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_string('log', './tmp/data', 'batch_norm, layer_norm, or None')
flags.DEFINE_string('logdir', './checkpoint_dir', 'directory for summaries and checkpoints.')

flags.DEFINE_integer('dim_input', 13, 'dim of input data')
flags.DEFINE_integer('dim_output', 2, 'dim of output data')
flags.DEFINE_integer('meta_batch_size', 16, 'number of tasks sampled per meta-update, not nums tasks')
flags.DEFINE_integer('num_samples_each_task', 16,
                     'number of samples sampling from each task when training, inner_batch_size')
flags.DEFINE_integer('test_update_batch_size', 8,
                     'number of examples used for gradient update during adapting (K=1,3,5 in experiment, K-shot); -1: M.')
flags.DEFINE_integer('metatrain_iterations', 5001, 'number of meta-training iterations.')
flags.DEFINE_integer('num_updates', 5, 'number of inner gradient updates during training.')
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
# flags.DEFINE_integer('num_samples', 18469, 'total number of samples in HK, see samples_HK.')
flags.DEFINE_float('update_lr', 1e-2, 'learning rate of single task objective (inner)')  # le-2 is the best
flags.DEFINE_float('meta_lr', 1e-3, 'the base learning rate of meta objective (outer)')  # le-2 or le-3
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')


def train(model, saver, sess, exp_string, tasks, resume_itr):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    PRINT_INTERVAL = 1000

    print('Done model initializing, starting training...')
    prelosses, postlosses = [], []
    if resume_itr != FLAGS.pretrain_iterations + FLAGS.metatrain_iterations - 1:
        if FLAGS.log:
            train_writer = tf.compat.v1.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
        for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
            batch_x, batch_y, cnt_sample = tasksbatch_generator(tasks, FLAGS.meta_batch_size
                                                                , FLAGS.num_samples_each_task,
                                                                FLAGS.dim_input,
                                                                FLAGS.dim_output)  # task_batch[i]: (x, y, features)
            # batch_y = _transform_labels_to_network_format(batch_y, FLAGS.num_classes)
            inputa = batch_x[:, :int(FLAGS.num_samples_each_task / 2), :]  # a used for training
            labela = batch_y[:, :int(FLAGS.num_samples_each_task / 2), :]
            inputb = batch_x[:, int(FLAGS.num_samples_each_task / 2):, :]  # b used for testing
            labelb = batch_y[:, int(FLAGS.num_samples_each_task / 2):, :]
            # # when deal with few-shot problem
            # inputa = batch_x[:, :int(len(batch_x[0]) / 2), :]  # a used for training
            # labela = batch_y[:, :int(len(batch_y[0]) / 2), :]
            # inputb = batch_x[:, int(len(batch_x[0]) / 2):, :]  # b used for testing
            # labelb = batch_y[:, int(len(batch_y[0]) / 2):, :]

            feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela,
                         model.labelb: labelb, model.cnt_sample: cnt_sample}

            if itr < FLAGS.pretrain_iterations:
                input_tensors = [model.pretrain_op]  # for comparison
            else:
                input_tensors = [model.metatrain_op]  # meta_train

            if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
                input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates - 1]])

            result = sess.run(input_tensors, feed_dict)

            if itr % SUMMARY_INTERVAL == 0:
                prelosses.append(result[-2])
                if FLAGS.log:
                    train_writer.add_summary(result[1], itr)  # add sum_op
                postlosses.append(result[-1])

            if (itr != 0) and itr % PRINT_INTERVAL == 0:
                if itr < FLAGS.pretrain_iterations:
                    print_str = 'Pretrain Iteration ' + str(itr)
                else:
                    print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
                print_str += ': ' + 'mean inner loss:' + str(np.mean(prelosses)) + '; ' \
                                                                                   'outer loss:' + str(
                    np.mean(postlosses))
                print(print_str)
                print('inner lr:', sess.run(model.update_lr))
                prelosses, postlosses = [], []
            #  save model
            if (itr != 0) and itr % SAVE_INTERVAL == 0:
                saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))
        saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))


def test(model, saver, sess, exp_string, tasks, num_updates=5):
    print('start evaluation...')
    print(exp_string)
    total_Ytest, total_Ypred, total_Ytest1, total_Ypred1, sum_accuracies, sum_accuracies1 = [], [], [], [], [], []

    for i in range(len(tasks)):
        np.random.shuffle(tasks[i])
        train_ = tasks[i][:int(len(tasks[i]) / 2)]
        test_ = tasks[i][int(len(tasks[i]) / 2):]  # test_ samples account 25%
        """few-steps tuning （不用op跑是因为采用的batch_size（input shape）不一致，且不想更新model.weight）"""
        with tf.compat.v1.variable_scope('model', reuse=True):  # np.normalize()里Variable重用
            fast_weights = model.weights
            for j in range(num_updates):
                inputa, labela = batch_generator(train_, FLAGS.dim_input, FLAGS.dim_output,
                                                 FLAGS.test_update_batch_size)
                loss = model.loss_func(model.forward(inputa, fast_weights, reuse=True),
                                       labela)  # fast_weight和grads（stopped）有关系，但不影响这里的梯度计算
                grads = tf.gradients(ys=loss, xs=list(fast_weights.values()))
                gradients = dict(zip(fast_weights.keys(), grads))
                fast_weights = dict(zip(fast_weights.keys(),
                                        [fast_weights[key] - model.update_lr * gradients[key] for key in
                                         fast_weights.keys()]))
            """Single task test accuracy"""
            inputb, labelb = batch_generator(test_, FLAGS.dim_input, FLAGS.dim_output, len(test_))
            Y_array = sess.run(tf.nn.softmax(model.forward(inputb, fast_weights, reuse=True)))  # pred_prob
            total_Ypred1.extend(Y_array)  # pred_prob_test
            total_Ytest1.extend(labelb)  # label

            Y_test = []  # for single task test
            for j in range(len(labelb)):
                Y_test.append(labelb[j][0])
                total_Ytest.append(labelb[j][0])
            Y_pred = []  # for single task test
            for j in range(len(labelb)):
                if Y_array[j][0] > Y_array[j][1]:
                    Y_pred.append(1)
                    total_Ypred.append(1)  # total_Ypred: 1d-array label
                else:
                    Y_pred.append(0)
                    total_Ypred.append(0)
            accuracy = accuracy_score(Y_test, Y_pred)
            sum_accuracies.append(accuracy)

    """Overall evaluation (test data)"""
    total_Ypred = np.array(total_Ypred).reshape(len(total_Ypred), )
    total_Ytest = np.array(total_Ytest)
    total_acc = accuracy_score(total_Ytest, total_Ypred)
    print('Test_Accuracy: %f' % total_acc)
    cal_measure(total_Ypred, total_Ytest)
    kappa_value = cohen_kappa_score(total_Ypred, total_Ytest)
    print('Cohen_Kappa: %f' % kappa_value)

    # save prediction for test samples, which can be used in calculating statistical measure such as AUROC
    pred_prob = np.array(total_Ypred1)
    label_bi = np.array(total_Ytest1)
    savearr = np.hstack((pred_prob, label_bi))
    writer = pd.ExcelWriter('proposed_test.xlsx')
    data_df = pd.DataFrame(savearr)
    data_df.to_excel(writer)
    writer.close()

    sess.close()


def metatask_sampling(p_samples, years):
    metatask_p = []
    # for year in years:
    #     if year
    return metatask_p


def main():
    """input data"""
    if not os.path.exists('./task_sampling/meta_task.xlsx'):
        print('meta_task generation...')
        # positive samples
        p_data = np.loadtxt('./data_src/p_samples.csv', dtype=str, delimiter=",", encoding='UTF-8-sig')
        p_samples = p_data[1:, :-3].astype(np.float32)
        f_names = p_data[0, :-4].astype(str)
        years = p_data[1:, -3].astype(int)
        # negative samples
        n_data = np.loadtxt('./data_src/n_samples.csv', dtype=str, delimiter=",", encoding='UTF-8-sig')
        n_samples = n_data[1:, :-2].astype(np.float32)

        # feature normalization
        sample_f, mean, std = feature_normalization(np.vstack((p_samples, n_samples))[:, :-1])
        p_samples_norm = np.hstack((sample_f[:len(p_samples), :], p_samples[:, -1].reshape(-1, 1)))
        n_samples_norm = np.hstack((sample_f[len(p_samples):, :], n_samples[:, -1].reshape(-1, 1)))

        # TODO: divide positive samples yearly (option 1)
        # tasks_train, tasks_test = metatask_sampling(p_samples, years)

        # divide equally (option 2)
        K = 200  # number of meta_tasks
        meta_tasks = []
        for k in range(K):
            p_samples_ = p_samples_norm[k * 30: (k + 1) * 30, :]
            np.random.shuffle(n_samples_norm)
            n_samples_ = n_samples_norm[:60, :]
            meta_tasks.append(np.vstack((p_samples_, n_samples_)))

        def transform_data(meta_tasks):
            tasks = [[] for i in range(len(meta_tasks))]
            for k in range(K):
                tasks[k] = [[] for i in range(len(meta_tasks[k]))]
            for i in range(len(meta_tasks)):
                for j in range(len(meta_tasks[i])):
                    tasks[i][j].append(meta_tasks[i][j][:-1])  # features
                    tasks[i][j].append(meta_tasks[i][j][-1])  # label
            return tasks

        # meta-datasets for meta-training and meta-testing
        meta_tasks_ = transform_data(meta_tasks)
        save_tasks(meta_tasks_, 'task_sampling/meta_task.xlsx')  # save meta_tasks to excel
    else:
        print('read meta_tasks from excel...')
        meta_tasks_ = read_tasks('task_sampling/meta_task.xlsx')

    tasks_train = meta_tasks_[:int(3 / 4 * 200)]
    tasks_test = meta_tasks_[int(3 / 4 * 200):]

    """meta-training and -testing"""
    print('model construction...')
    model = Meta_learner(FLAGS.dim_input, FLAGS.dim_output, test_num_updates=5)

    input_tensors_input = (FLAGS.meta_batch_size, int(FLAGS.num_samples_each_task / 2), FLAGS.dim_input)
    input_tensors_label = (FLAGS.meta_batch_size, int(FLAGS.num_samples_each_task / 2), FLAGS.dim_output)
    model.construct_model(input_tensors_input=input_tensors_input, input_tensors_label=input_tensors_label,
                          prefix='metatrain_')
    model.summ_op = tf.compat.v1.summary.merge_all()

    saver = tf.compat.v1.train.Saver(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES),
                                     max_to_keep=10)

    sess = tf.compat.v1.InteractiveSession()
    init = tf.compat.v1.global_variables()  # optimizer里会有额外variable需要初始化
    sess.run(tf.compat.v1.variables_initializer(var_list=init))

    exp_string = '.mbs' + str(FLAGS.meta_batch_size) + '.ubs_' + \
                 str(FLAGS.num_samples_each_task) + '.numstep' + str(FLAGS.num_updates) + \
                 '.updatelr' + str(FLAGS.update_lr) + '.meta_lr' + str(FLAGS.meta_lr)

    resume_itr = 0

    # 续点训练
    if FLAGS.resume:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1 + 5:])
            # print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)  # 以model_file初始化sess中图

    train(model, saver, sess, exp_string, tasks_train, resume_itr)

    test(model, saver, sess, exp_string, tasks_test, num_updates=FLAGS.num_updates)


if __name__ == "__main__":
    # device=tf.config.list_physical_devices('GPU')
    tf.compat.v1.disable_eager_execution()
    main()
    print('finished!')
