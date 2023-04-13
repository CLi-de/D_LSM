
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.platform import flags
from utils import mse, xent, normalize

FLAGS = flags.FLAGS


class Meta_learner:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.meta_lr = tf.compat.v1.placeholder_with_default(FLAGS.meta_lr, ())
        self.test_num_updates = test_num_updates
        self.dim_hidden = [32, 32, 16]
        self.loss_func = xent
        self.forward = self.forward_fc
        if FLAGS.basemodel == 'MLP':
            self.construct_weights = self.construct_fc_weights  # 参数初始化(random)
        elif FLAGS.basemodel == 'DAS':
            if os.path.exists('unsupervised_pretraining/model_init/savedmodel.npz'):
                self.construct_weights = self.construct_DAS_weights  # DAS初始化
            else:
                raise ValueError('No pretrained model found!')
        else:
            raise ValueError('Unrecognized base model, please specify a base model such as "MLP", "DAS"...')

    def construct_model(self, input_tensors_input=None, input_tensors_label=None, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        self.inputa = tf.compat.v1.placeholder(tf.float32, shape=input_tensors_input)   # for train in a task
        self.inputb = tf.compat.v1.placeholder(tf.float32, shape=input_tensors_input)   # for test in a task
        self.labela = tf.compat.v1.placeholder(tf.float32, shape=input_tensors_label)
        self.labelb = tf.compat.v1.placeholder(tf.float32, shape=input_tensors_label)
        self.cnt_sample = tf.compat.v1.placeholder(tf.float32)  # count number of samples for each task in the batch

        with tf.compat.v1.variable_scope('model', reuse=None) as training_scope:
            # Attention module
            self.A = tf.Variable(tf.zeros([self.dim_input, self.dim_input]))
            # initialize the inner learning rate as tf.Variable within 'model' scope
            self.update_lr = tf.Variable(FLAGS.update_lr)
            # self.update_lr = FLAGS.update_lr
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()  # 初始化FC权重参数

            num_updates = max(self.test_num_updates, FLAGS.num_updates)  # training iteration in a task

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp  # inputa: Task(i)训练输入，batch_size = m(m个samples)
                task_outputbs, task_lossesb = [], []

                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela)

                grads = tf.gradients(ys=task_lossa, xs=list(weights.values()))  # 计算梯度
                if FLAGS.stop_grad:  # maml中的二次求导（）
                    grads = [tf.stop_gradient(grad) for grad in grads]  # 使梯度无法进行二次求偏导（BP）
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr * gradients[key] for key in
                                                         weights.keys()]))  # 更新weight
                output = self.forward(inputb, fast_weights, reuse=True)  # Task(i) test output
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                for j in range(num_updates - 1):  # num_updates:Task(i)中用batch_size个训练样本更新权值的迭代次数
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True),
                                          labela)  # fast_weight和grads（stopped）有关系，但不影响这里的梯度计算
                    grads = tf.gradients(ys=loss, xs=list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(),
                                            [fast_weights[key] - self.update_lr * gradients[key] for key in
                                             fast_weights.keys()]))
                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                return [task_outputa, task_outputbs, task_lossa, task_lossesb]  # task_outpouta, task_lossa是仅

            if FLAGS.norm != 'None':  # 此处不能删，考虑到reuse
                '''to initialize the batch norm vars, might want to combine this, and not run idx 0 twice (use reuse=tf.AUTO_REUSE instead).'''
                task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates]

            """输入各维度（for batch）进行task_metalearn的并行操作， 相较out_dtype多了batch_size的维度"""

            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb),
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)  # parallel calculation

            """ outputas:(num_tasks, num_samples, value)
                outputbs[i]:i是迭代次数，不同迭代次数的预测值(num_tasks, num_samples, value)
                lossesa:(num_tasks, value)
                lossesb[i]:i是迭代次数，不同迭代次数的预测值(num_tasks, value)"""
            outputas, outputbs, lossesa, lossesb = result  # outputas:(num_tasks, num_samples, value)

        # Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(input_tensor=lossesa) / tf.cast(FLAGS.meta_batch_size,
                                                                                           dtype=tf.float32)  # total loss的均值,finn论文中的pretrain（对比用）

            self.total_losses2 = total_losses2 = [
                tf.reduce_sum(lossesb[j]) / tf.cast(FLAGS.meta_batch_size, dtype=tf.float32) \
                for j in range(num_updates)]

            # w = self.cnt_sample / tf.cast(FLAGS.num_samples, dtype=tf.float32)
            # self.total_losses2 = total_losses2 = [tf.reduce_sum(
            #     input_tensor=tf.multiply(tf.nn.softmax(w), tf.reduce_sum(input_tensor=lossesb[j], axis=1)))
            #     for j in range(num_updates)]

            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs  # outputbs：25个task, 每个task迭代五次，value（25,5,1）
            self.pretrain_op = tf.compat.v1.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)  # inner for test

            optimizer = tf.compat.v1.train.AdamOptimizer(self.meta_lr)

            self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[
                                                             FLAGS.num_updates - 1])  # 取最后一次迭代的Lossb，gvs：gradients and variables，对所有trainable variables求梯度
            self.metatrain_op = optimizer.apply_gradients(gvs)  # outer
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(input_tensor=lossesa) / tf.cast(
                FLAGS.meta_batch_size, dtype=tf.float32)  # inner loss
            self.metaval_total_losses2 = total_losses2 = [
                tf.reduce_sum(input_tensor=lossesb[j]) / tf.cast(FLAGS.meta_batch_size, dtype=tf.float32) for j in
                range(num_updates)]  # outer losses(每次迭代的)

        # Summaries
        tf.compat.v1.summary.scalar(prefix + 'Pre-update loss', total_loss1)  # for test accuracy
        for j in range(num_updates):
            tf.compat.v1.summary.scalar(prefix + 'Post-update loss, step ' + str(j + 1), total_losses2[j])

    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.random.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1, len(self.dim_hidden)):
            weights['w' + str(i + 1)] = tf.Variable(
                tf.random.truncated_normal([self.dim_hidden[i - 1], self.dim_hidden[i]], stddev=0.01))
            weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w' + str(len(self.dim_hidden) + 1)] = tf.Variable(
            tf.random.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b' + str(len(self.dim_hidden) + 1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward_fc(self, inp, weights, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1, len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)],
                               activation=tf.nn.relu, reuse=reuse, scope=str(i + 1))
        return tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)]) + weights[
            'b' + str(len(self.dim_hidden) + 1)]

    def construct_DAS_weights(self):
        """读取DAS权参"""
        npzfile = np.load('unsupervised_pretraining/model_init/savedmodel.npz')
        weights = {}
        weights['w1'] = tf.Variable(tf.transpose(a=npzfile['arr_0']))
        weights['b1'] = tf.Variable(npzfile['arr_1'])
        weights['w2'] = tf.Variable(tf.transpose(a=npzfile['arr_2']))
        weights['b2'] = tf.Variable(npzfile['arr_3'])
        weights['w3'] = tf.Variable(tf.transpose(a=npzfile['arr_4']))
        weights['b3'] = tf.Variable(npzfile['arr_5'])
        weights['w4'] = tf.Variable(tf.random.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b4'] = tf.Variable(tf.zeros([self.dim_output]))
        return weights
