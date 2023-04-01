import numpy as np

import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
import umap

from mpl_toolkits.mplot3d import Axes3D  # 3D plot
import pandas as pd
import tensorflow as tf

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from unsupervised_pretraining.dbn_.models import SupervisedDBNClassification
from scipy.interpolate import make_interp_spline
from sklearn.metrics._classification import accuracy_score

"""for visiualization"""


def read_tasks(file, dim_input=16):
    """read csv and obtain tasks"""
    f = pd.ExcelFile(file)
    tasks = []
    for sheetname in f.sheet_names:
        attr = pd.read_excel(file, usecols=dim_input - 1, sheet_name=sheetname).values.astype(np.float32)
        label = pd.read_excel(file, usecols=[dim_input], sheet_name=sheetname).values.reshape((-1, 1)).astype(
            np.float32)
        tasks.append([attr, label])
    return tasks


def read_csv(path):
    tmp = np.loadtxt(path, dtype=np.str, delimiter=",", encoding='UTF-8')
    tmp_feature = tmp[1:, :]
    np.random.shuffle(tmp_feature)  # shuffle
    label_attr = tmp_feature[:, -1].astype(np.float32)  #
    data_atrr = tmp_feature[:, :-1].astype(np.float32)  #
    return data_atrr, label_attr


def load_weights(npzfile):
    npzfile = np.load(npzfile)
    weights = {}
    weights['w0'] = npzfile['arr_0']
    weights['b0'] = npzfile['arr_1']
    weights['w1'] = npzfile['arr_2']
    weights['b1'] = npzfile['arr_3']
    weights['w2'] = npzfile['arr_4']
    weights['b2'] = npzfile['arr_5']
    return weights


def transform_relu(inputX, weights, bias, activations=tf.nn.relu):
    return activations(tf.transpose(a=tf.matmul(weights, tf.transpose(a=inputX))) + bias)


def forward(inp, weights, sess):
    for i in range(int(len(weights) / 2)):  # 3 layers
        inp = transform_relu(inp, tf.transpose(a=weights['w' + str(i)]), weights['b' + str(i)])
    return sess.run(inp)


def _PCA(X, y, figsavename):
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    x_min, x_max = X_pca.min(0), X_pca.max(0)
    X_norm = (X_pca - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(x1,x2,x3,c=pre

    landslide_pts_x = []
    landslide_pts_y = []
    landslide_pts_z = []
    nonlandslide_pts_x = []
    nonlandslide_pts_y = []
    nonlandslide_pts_z = []

    for i in range(len(y)):
        if y[i] == 0:
            nonlandslide_pts_x.append(X_norm[i][0])
            nonlandslide_pts_y.append(X_norm[i][1])
            nonlandslide_pts_z.append(X_norm[i][2])
        if y[i] == 1:
            landslide_pts_x.append(X_norm[i][0])
            landslide_pts_y.append(X_norm[i][1])
            landslide_pts_z.append(X_norm[i][2])

    type_landslide = ax.scatter(landslide_pts_x, landslide_pts_y, landslide_pts_z, c='red')
    type_nonlandslide = ax.scatter(nonlandslide_pts_x, nonlandslide_pts_y, nonlandslide_pts_z, c='blue')

    ax.legend((type_landslide, type_nonlandslide), ('landslide points', 'nonlandslide points'), loc=2)
    # plt.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # 设置坐标标签
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')

    # 设置标题
    plt.title("Visualization with PCA")
    plt.savefig(figsavename)
    # 显示图形
    plt.show()


def ISOMAP(X, y, figsavename):
    isomap = manifold.Isomap(n_components=3)
    X_isomap = isomap.fit_transform(X)

    x_min, x_max = X_isomap.min(0), X_isomap.max(0)
    X_norm = (X_isomap - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(x1,x2,x3,c=pre

    landslide_pts_x = []
    landslide_pts_y = []
    landslide_pts_z = []
    nonlandslide_pts_x = []
    nonlandslide_pts_y = []
    nonlandslide_pts_z = []

    for i in range(len(y)):
        if y[i] == 0:
            nonlandslide_pts_x.append(X_norm[i][0])
            nonlandslide_pts_y.append(X_norm[i][1])
            nonlandslide_pts_z.append(X_norm[i][2])
        if y[i] == 1:
            landslide_pts_x.append(X_norm[i][0])
            landslide_pts_y.append(X_norm[i][1])
            landslide_pts_z.append(X_norm[i][2])

    type_landslide = ax.scatter(landslide_pts_x, landslide_pts_y, landslide_pts_z, c='red')
    type_nonlandslide = ax.scatter(nonlandslide_pts_x, nonlandslide_pts_y, nonlandslide_pts_z, c='blue')

    ax.legend((type_landslide, type_nonlandslide), ('landslide points', 'nonlandslide points'), loc=2)
    # plt.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # 设置坐标标签
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    # 设置标题
    plt.title("Visualization with Isomap")
    plt.savefig(figsavename)
    # 显示图形
    plt.show()


def t_SNE(X, y, figsavename):
    tsne = manifold.TSNE(n_components=3, init='random', random_state=501)
    X_tsne = tsne.fit_transform(X)

    """嵌入空间可视化"""
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(x1,x2,x3,c=pre

    landslide_pts_x = []
    landslide_pts_y = []
    landslide_pts_z = []
    nonlandslide_pts_x = []
    nonlandslide_pts_y = []
    nonlandslide_pts_z = []

    for i in range(len(y)):
        if y[i] == 0:
            nonlandslide_pts_x.append(X_norm[i][0])
            nonlandslide_pts_y.append(X_norm[i][1])
            nonlandslide_pts_z.append(X_norm[i][2])
        if y[i] == 1:
            landslide_pts_x.append(X_norm[i][0])
            landslide_pts_y.append(X_norm[i][1])
            landslide_pts_z.append(X_norm[i][2])

    type_landslide = ax.scatter(landslide_pts_x, landslide_pts_y, landslide_pts_z, c='red')
    type_nonlandslide = ax.scatter(nonlandslide_pts_x, nonlandslide_pts_y, nonlandslide_pts_z, c='blue')

    ax.legend((type_landslide, type_nonlandslide), ('landslide points', 'nonlandslide points'), loc=2)
    # plt.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # 设置坐标标签
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    # 设置标题
    plt.title("Visualization with t-SNE")
    plt.savefig(figsavename)
    # 显示图形
    plt.show()


def UMAP(X, y, figsavename):
    reducer = umap.UMAP(n_components=3)
    X_umap = reducer.fit_transform(X)

    x_min, x_max = X_umap.min(0), X_umap.max(0)
    X_norm = (X_umap - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.scatter(x1,x2,x3,c=pre

    landslide_pts_x = []
    landslide_pts_y = []
    landslide_pts_z = []
    nonlandslide_pts_x = []
    nonlandslide_pts_y = []
    nonlandslide_pts_z = []

    for i in range(len(y)):
        if y[i] == 0:
            nonlandslide_pts_x.append(X_norm[i][0])
            nonlandslide_pts_y.append(X_norm[i][1])
            nonlandslide_pts_z.append(X_norm[i][2])
        if y[i] == 1:
            landslide_pts_x.append(X_norm[i][0])
            landslide_pts_y.append(X_norm[i][1])
            landslide_pts_z.append(X_norm[i][2])

    type_landslide = ax.scatter(landslide_pts_x, landslide_pts_y, landslide_pts_z, c='red')
    type_nonlandslide = ax.scatter(nonlandslide_pts_x, nonlandslide_pts_y, nonlandslide_pts_z, c='blue')

    ax.legend((type_landslide, type_nonlandslide), ('landslide points', 'nonlandslide points'), loc=2)
    # plt.legend( bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # 设置坐标标签
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    # 设置标题
    plt.title("Visualization with UMAP")
    plt.savefig(figsavename)
    # 显示图形
    plt.show()


def visualization():
    fj_tasks = read_tasks('./metatask_sampling/FJ_tasks.xlsx')  # task里的samplles
    # fl_tasks = read_tasks('./metatask_sampling/FL_tasks.xlsx')  # num_samples of FL is too scarce to visualize

    """select part of FJ and FL data for visualization"""
    # def get_oriandinf_Xs(tasks, regionname):
    #     ori_Xs, inf_Xs, Ys = [], [], []
    #     for i in range(len(tasks)):
    #         if tasks[i][0].shape[0] > 30:
    #             """download model parameters"""
    #             w = load_weights('models_of_blocks/'+ regionname +'/model'+str(i) + '.npz')
    #             ori_Xs.append(tasks[i][0])
    #             inf_Xs.append(forward(tasks[i][0], w))
    #             Ys.append(tasks[i][1])
    #     return ori_Xs, inf_Xs, Ys
    """overall FJ and FL data for visualization"""

    def get_oriandinf_Xs(tasks, regionname):
        with tf.compat.v1.Session() as sess:  # for tf calculation
            w = load_weights('models_of_blocks/' + 'overall_' + regionname + '/model_MAML' + '.npz')
            ori_Xs = tasks[0][0]
            inf_Xs = forward(tasks[0][0], w, sess)
            Ys = tasks[0][1]
            for i in range(len(tasks) - 1):
                if len(tasks[i + 1][0]) > 0:
                    ori_Xs = np.vstack((ori_Xs, tasks[i + 1][0]))
                    inf_Xs = np.vstack((inf_Xs, forward(tasks[i + 1][0], w, sess)))
                    Ys = np.vstack((Ys, tasks[i + 1][1]))
            return ori_Xs, inf_Xs, Ys

    ori_FJ_Xs, inf_FJ_Xs, FJ_Ys = get_oriandinf_Xs(fj_tasks, 'FJ')

    # ori_FL_Xs, inf_FL_Xs, FL_Ys = get_oriandinf_Xs(fl_tasks, 'FL')

    # ori_X, y = read_csv('src_data/FJ_FL.csv')
    #
    # tmp = np.loadtxt('src_data/FJ_FL.csv', dtype=np.str, delimiter=",",encoding='UTF-8')
    # w = load_weights('unsupervised_pretraining/model_init/savedmodel.npz')
    # unsupervised_X = forward(ori_X, w)
    def plot_points(ori_X, inf_X, Y, regionname):
        _PCA(ori_X, Y, './figs/' + regionname + '_ori_PCA.pdf')
        _PCA(inf_X, Y, './figs/' + regionname + '_inf_PCA.pdf')

        t_SNE(ori_X, Y, './figs/' + regionname + '_ori_t_SNE.pdf')
        t_SNE(inf_X, Y, './figs/' + regionname + '_inf_SNE.pdf')

        ISOMAP(ori_X, Y, './figs/' + regionname + '_ori_Isomap.pdf')
        ISOMAP(inf_X, Y, './figs/' + regionname + '_inf_Isomap.pdf')

        UMAP(ori_X, Y, './figs/' + regionname + '_ori_UMAP.pdf')
        UMAP(inf_X, Y, './figs/' + regionname + '_inf_UMAP.pdf')

    plot_points(ori_FJ_Xs, inf_FJ_Xs, FJ_Ys, 'FJ')
    # plot_points(ori_FL_Xs, inf_FL_Xs, FL_Ys, 'FL')


"""for figure plotting"""


def read_statistic(file):
    """读取csv获取statistic"""
    f = pd.ExcelFile(file)
    K, meanOA, maxOA, minOA, std = [], [], [], [], []
    for sheetname in f.sheet_names:
        tmp_K, tmp_meanOA, tmp_maxOA, tmp_minOA, tmp_std = np.transpose(
            pd.read_excel(file, sheet_name=sheetname).values)
        K.append(tmp_K)
        meanOA.append(tmp_meanOA)
        maxOA.append(tmp_maxOA)
        minOA.append(tmp_minOA)
        std.append(tmp_std)
    return K, meanOA, maxOA, minOA, std


def read_statistic1(file):
    """读取csv获取statistic"""
    f = pd.ExcelFile(file)
    K, meanOA = [], []
    for sheetname in f.sheet_names:
        tmp_K, tmp_meanOA = np.transpose(pd.read_excel(file, sheet_name=sheetname).values)
        K.append(tmp_K)
        meanOA.append(tmp_meanOA)
    return K, meanOA


def read_statistic2(file):
    """读取csv获取statistic"""
    f = pd.ExcelFile(file)
    measures = []
    for sheetname in f.sheet_names:
        temp = pd.read_excel(file, sheet_name=sheetname).values
        measures.append(temp[:, 1:].tolist())
    return measures


def plot_candle(scenes, K, meanOA, maxOA, minOA, std):
    # 设置框图
    plt.figure("", facecolor="lightgray")
    # plt.style.use('ggplot')
    # 设置图例并且设置图例的字体及大小
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }

    # legend = plt.legend(handles=[A,B],prop=font1)
    # plt.title(scenes, fontdict=font2)
    # plt.xlabel("Various methods", fontdict=font1)
    plt.ylabel("OA(%)", fontdict=font2)

    my_x_ticks = [1, 2, 3, 4, 5]
    # my_x_ticklabels = ['SVM', 'MLP', 'DBN', 'RF', 'Proposed']
    plt.xticks(ticks=my_x_ticks, labels='', fontsize=16)

    plt.ylim((60, 100))
    my_y_ticks = np.arange(60, 100, 5)
    plt.yticks(ticks=my_y_ticks, fontsize=16)

    colors = ['dodgerblue', 'lawngreen', 'gold', 'magenta', 'red']
    edge_colors = np.zeros(5, dtype="U1")
    edge_colors[:] = 'black'

    '''格网设置'''
    plt.grid(linestyle="--", zorder=-1)

    # draw line
    # plt.plot(K[0:-1], meanOA[0:-1], color="b", linestyle='solid',
    #         linewidth=1, label="open", zorder=1)
    # plt.plot(K[-2:], meanOA[-2:], color="b", linestyle="--",
    #          linewidth=1, label="open", zorder=1)

    # draw bar
    barwidth = 0.4
    plt.bar(K, 2 * std, barwidth, bottom=meanOA - std, color=colors,
            edgecolor=edge_colors, linewidth=1, zorder=20, label=['SVM', 'MLP', 'DBN', 'RF', 'Proposed'])

    # draw vertical line
    plt.vlines(K, minOA, maxOA, color='black', linestyle='solid', zorder=10)
    plt.hlines(meanOA, K - barwidth / 2, K + barwidth / 2, color='black', linestyle='solid', zorder=30)
    plt.hlines(minOA, K - barwidth / 4, K + barwidth / 4, color='black', linestyle='solid', zorder=10)
    plt.hlines(maxOA, K - barwidth / 4, K + barwidth / 4, color='black', linestyle='solid', zorder=10)

    # 设置图例
    legend = plt.legend(loc="lower right", prop=font1, ncol=3, fontsize=24)


def plot_scatter(arr):
    '''设置框图'''
    # plt.figure("", facecolor="lightgray")  # 设置框图大小
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             }
    plt.xlabel("Subtasks", fontdict=font1)
    plt.ylabel("Mean accuracy(%)", fontdict=font1)

    '''设置刻度'''
    plt.ylim((50, 100))
    my_y_ticks = np.arange(50, 100, 5)
    plt.yticks(my_y_ticks)
    my_x_ticks = [i for i in range(1, 204, 40)]
    my_x_ticklabel = [str(i) + 'th' for i in range(1, 204, 40)]
    plt.xticks(ticks=my_x_ticks, labels=my_x_ticklabel)
    '''格网设置'''
    plt.grid(linestyle="--")

    x_ = [i for i in range(arr.shape[0])]
    '''draw scatter'''
    L1 = plt.scatter(x_, arr[:, 0], label="L=1", c="none", s=20, edgecolors='magenta')
    L2 = plt.scatter(x_, arr[:, 1], label="L=2", c="none", s=20, edgecolors='cyan')
    L3 = plt.scatter(x_, arr[:, 2], label="L=3", c="none", s=20, edgecolors='b')
    L4 = plt.scatter(x_, arr[:, 3], label="L=4", c="none", s=20, edgecolors='g')
    L5 = plt.scatter(x_, arr[:, 4], label="L=5", c="none", s=20, edgecolors='r')

    '''设置图例'''
    legend = plt.legend(loc="lower left", prop=font2, ncol=3)
    # plt.savefig("C:\\Users\\hj\\Desktop\\brokenline_A")
    # plt.show()


def plot_lines(arr):
    '''设置框图'''
    # plt.figure("", facecolor="lightgray")  # 设置框图大小
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             }
    plt.xlabel("Subtasks", fontdict=font1)
    plt.ylabel("Mean accuracy(%)", fontdict=font1)

    '''设置刻度'''
    plt.ylim((50, 100))
    my_y_ticks = np.arange(50, 100, 5)
    plt.yticks(my_y_ticks)
    my_x_ticks = [i for i in range(6)]
    my_x_ticklabel = [str(i + 1) + '/12 M' for i in range(6)]
    plt.xticks(ticks=my_x_ticks, labels=my_x_ticklabel)
    '''格网设置'''
    plt.grid(linestyle="--")

    x_ = np.array([i for i in range(6)])
    # smooth
    # x_ = np.linspace(x_.min(), x_.max(), 400)
    # arr = make_interp_spline(x_, arr)(x_)
    '''draw line'''
    L1 = plt.plot(x_, arr[:, 0], color="r", linestyle="solid",
                  linewidth=1, label="L=1", markerfacecolor='white', ms=10)
    L2 = plt.plot(x_, arr[:, 1], color="orange", linestyle="solid",
                  linewidth=1, label="L=2", markerfacecolor='white', ms=10)
    L3 = plt.plot(x_, arr[:, 2], color="gold", linestyle="solid",
                  linewidth=1, label="L=3", markerfacecolor='white', ms=10)
    L4 = plt.plot(x_, arr[:, 3], color="g", linestyle="solid",
                  linewidth=1, label="L=4", markerfacecolor='white', ms=10)
    L5 = plt.plot(x_, arr[:, 4], color="b", linestyle="solid",
                  linewidth=1, label="L=5", markerfacecolor='white', ms=10)


def plot_histogram(region, measures):
    '''设置框图'''
    plt.figure("", facecolor="lightgray")  # 设置框图大小
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 14,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }
    # plt.xlabel("Statistical measures", fontdict=font1)
    plt.ylabel("Performance(%)", fontdict=font1)
    plt.title(region, fontdict=font2)

    '''设置刻度'''
    plt.ylim((60, 90))
    my_y_ticks = np.arange(60, 90, 3)
    plt.yticks(my_y_ticks)

    my_x_ticklabels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    bar_width = 0.3
    interval = 0.2
    my_x_ticks = np.arange(bar_width / 2 + 2.5 * bar_width, 4 * 5 * bar_width + 1, bar_width * 6)
    plt.xticks(ticks=my_x_ticks, labels=my_x_ticklabels, fontproperties='Times New Roman', size=14)

    '''格网设置'''
    plt.grid(linestyle="--")

    '''draw bar'''
    rects1 = plt.bar([x - 2 * bar_width for x in my_x_ticks], height=measures[0], width=bar_width, alpha=0.8,
                     color='dodgerblue', label="MLP")
    rects2 = plt.bar([x - 1 * bar_width for x in my_x_ticks], height=measures[1], width=bar_width, alpha=0.8,
                     color='yellowgreen', label="RF")
    rects3 = plt.bar([x for x in my_x_ticks], height=measures[2], width=bar_width, alpha=0.8, color='gold', label="RL")
    rects4 = plt.bar([x + 1 * bar_width for x in my_x_ticks], height=measures[3], width=bar_width, alpha=0.8,
                     color='peru', label="MAML")
    rects5 = plt.bar([x + 2 * bar_width for x in my_x_ticks], height=measures[4], width=bar_width, alpha=0.8,
                     color='crimson', label="proposed")

    '''设置图例'''
    legend = plt.legend(loc="upper left", prop=font1, ncol=3)

    '''add text'''
    # for rect in rects1:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height)+'%', ha="center", va="bottom")
    # for rect in rects2:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height)+'%', ha="center", va="bottom")
    # for rect in rects3:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height)+'%', ha="center", va="bottom")
    # for rect in rects4:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height)+'%', ha="center", va="bottom")
    # for rect in rects5:
    #     height = rect.get_height()
    #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height)+'%', ha="center", va="bottom")

    plt.savefig("C:\\Users\\hj\\Desktop\\histogram" + region + '.pdf')
    plt.show()


"""for AUROC plotting"""


def load_data(filepath, dim_input):
    np.loadtxt(filepath, )
    data = pd.read_excel(filepath).values.astype(np.float32)
    attr = data[:, :dim_input]
    attr = attr / attr.max(axis=0)
    label = data[:, -1].astype(np.int32)
    return attr, label


def SVM_fit_pred(x_train, x_test, y_train, y_test):
    classifier = svm.SVC(C=1, kernel='rbf', gamma=1 / (2 * x_train.var()), decision_function_shape='ovr',
                         probability=True)
    classifier.fit(x_train, y_train)
    return classifier.predict_proba(x_test)


def MLP_fit_pred(x_train, x_test, y_train, y_test):
    classifier = MLPClassifier(hidden_layer_sizes=(32, 32, 16), activation='relu', solver='adam', alpha=0.01,
                               batch_size=32, max_iter=1000)
    classifier.fit(x_train, y_train)
    return classifier.predict_proba(x_test)


def DBN_fit_pred(x_train, x_test, y_train, y_test):
    classifier = SupervisedDBNClassification(hidden_layers_structure=[32, 32],
                                             learning_rate_rbm=0.001,
                                             learning_rate=0.5,
                                             n_epochs_rbm=10,
                                             n_iter_backprop=200,
                                             batch_size=64,
                                             activation_function='relu',
                                             dropout_p=0.1)
    classifier.fit(x_train, y_train)
    pred_prob = classifier.predict_proba(x_test)

    # if pred_prob[0][0] > 0.5:
    #     pred_prob = np.vstack((pred_prob[:, 0], pred_prob[:, -1])).T  # swap 0, 1 prediction

    return pred_prob


def RF_fit_pred(x_train, x_test, y_train, y_test):
    classifier = RandomForestClassifier(n_estimators=200, max_depth=None)
    classifier.fit(x_train, y_train)
    return classifier.predict_proba(x_test)


def plot_auroc(n_times, y_score_SVM, y_score_MLP, y_score_DBN, y_score_RF, y_score_proposed, y_test, y_test_proposed):
    # Compute ROC curve and ROC area for each class
    def cal_(y_score, y_test):
        fpr, tpr = [], []
        for i in range(n_times):
            fpr_, tpr_, thresholds = roc_curve(y_test[i], y_score[i][:, -1], pos_label=1)
            fpr.append(fpr_)
            tpr.append(tpr_)

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_times)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_times):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_times
        mean_auc = auc(all_fpr, mean_tpr)
        return all_fpr, mean_tpr, mean_auc, fpr, tpr

    def plot_(y_score, y_test, color, method):
        all_fpr, mean_tpr, mean_auc, fpr, tpr = cal_(y_score, y_test)
        # draw mean
        plt.plot(all_fpr, mean_tpr,
                 label=method + '_mean_AUC (area = {0:0.3f})'''.format(mean_auc),
                 color=color, linewidth=1.5)
        # draw each
        for i in range(n_times):
            plt.plot(fpr[i], tpr[i],
                     color=color, linewidth=1, alpha=.25)
        # plt.savefig(method + '.pdf')

    # Plot all ROC curves
    # ax = plt.axes()
    # ax.set_facecolor("WhiteSmoke")  # background color
    plot_(y_score_SVM, y_test, color='dodgerblue', method='SVM')
    plot_(y_score_MLP, y_test, color='lawngreen', method='MLP')
    plot_(y_score_DBN, y_test, color='gold', method='DBN')
    plot_(y_score_RF, y_test, color='magenta', method='RF')
    plot_(y_score_proposed, y_test_proposed, color='red', method='Proposed')

    # format
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             }
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontdict=font1)
    plt.ylabel('True Positive Rate', fontdict=font1)
    plt.title('ROC curve by various methods', fontdict=font1)
    plt.legend(loc="lower right", prop=font2)


"""space visualization"""
# visualization()

"""draw histogram"""

# regions = ['FJ', 'FL']
# measures = read_statistic2("C:\\Users\\hj\\Desktop\\performance.xlsx")
# for i in range(len(regions)):
#     plot_histogram(regions[i], measures[i])


"""draw candle"""


# scenes = ['airport', 'urban1', 'urban2', 'plain', 'catchment', 'reservior']
# K, meanOA, maxOA, minOA, std = read_statistic("C:\\Users\\lichen\\OneDrive\\桌面\\statistics_candle.xlsx")
# for i in range(len(scenes)):
#     plot_candle(scenes[i], K[i], meanOA[i], maxOA[i], minOA[i], std[i])
#     plt.savefig("C:\\Users\\lichen\\OneDrive\\桌面\\" + scenes[i] + '_' + 'candle.pdf')
#     plt.show()


def read_f_l_csv(file):
    tmp = np.loadtxt(file, dtype=str, delimiter=",", encoding='UTF-8')
    features = tmp[1:, :-2].astype(np.float32)
    features = features / features.max(axis=0)
    label = tmp[1:, -1].astype(np.float32)
    return features, label


# """draw AUR"""
# print('drawing ROC...')
# x, y = read_f_l_csv('src_data/samples_HK.csv')
# y_score_SVM, y_score_MLP, y_score_DBN, y_score_RF, y_score_proposed, y_test_, y_test_proposed = [], [], [], [], [], [], []
# n_times = 5
# for i in range(n_times):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=.02, shuffle=True)
#     """fit and predict"""
#     # for other methods
#     y_score_SVM.append(SVM_fit_pred(x_train, x_test, y_train, y_test))
#     y_score_MLP.append(MLP_fit_pred(x_train, x_test, y_train, y_test))
#     y_score_DBN.append(MLP_fit_pred(x_train, x_test, y_train, y_test))
#     y_score_RF.append(RF_fit_pred(x_train, x_test, y_train, y_test))
#     y_test_.append(y_test)
#     # for proposed-
#     tmp = pd.read_excel('tmp/' + 'proposed_test' + str(i) + '.xlsx').values.astype(np.float32)
#     y_score_proposed.append(tmp[:, 1:3])
#     y_test_proposed.append(tmp[:, -1])
# # draw roc
# plt.clf()
# plot_auroc(n_times, y_score_SVM, y_score_MLP, y_score_DBN, y_score_RF, y_score_proposed, y_test_, y_test_proposed)
# plt.savefig('ROC.pdf')
# plt.show()
# print('finish')

"""draw scatters for fast adaption performance"""
filename = "C:\\Users\\lichen\\OneDrive\\桌面\\fast_adaption_sheet2.csv"
arr = np.loadtxt(filename, dtype=float, delimiter=",", encoding='utf-8-sig')
plot_scatter(arr)
plt.savefig("C:\\Users\\lichen\\OneDrive\\桌面\\scatters.pdf")
plt.show()

"""draw lines for fast adaption performance"""
# filename = "C:\\Users\\lichen\\OneDrive\\桌面\\fast_adaption1.csv"
# arr = np.loadtxt(filename, dtype=float, delimiter=",", encoding='utf-8-sig')
# plot_lines(arr)
# plt.savefig("C:\\Users\\lichen\\OneDrive\\桌面\\broken.pdf")
# plt.show()

"""
label: for legend
pos_: -2, -1, 0, 1, 2
"""


def plot_candle1(K, meanOA, maxOA, minOA, std, color_, label_, pos_):
    # 设置框图
    # plt.figure("", facecolor="lightgray")
    # plt.style.use('ggplot')
    # 设置图例并且设置图例的字体及大小
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 14,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }

    # legend = plt.legend(handles=[A,B],prop=font1)
    # plt.title(scenes, fontdict=font2)
    plt.xlabel("Number of samples", fontdict=font1)
    plt.ylabel("OA(%)", fontdict=font2)

    my_x_ticks = [1, 2, 3, 4, 5]
    my_x_ticklabels = ['1', '2', '3', '4', '5']
    plt.xticks(ticks=my_x_ticks, labels=my_x_ticklabels, fontsize=14, fontdict=font2)

    plt.ylim((50, 100))
    my_y_ticks = np.arange(50, 100, 5)
    plt.yticks(ticks=my_y_ticks, fontsize=14, font=font2)

    '''格网设置'''
    plt.grid(linestyle="--", zorder=-1)

    colors = ['dodgerblue', 'lawngreen', 'gold', 'magenta', 'red']
    edge_colors = np.zeros(5, dtype="U1")
    edge_colors[:] = 'black'

    # draw bar
    barwidth = 0.15
    K = K + barwidth * pos_
    plt.bar(K, 2 * std, barwidth, bottom=meanOA - std, color=color_,
            edgecolor=edge_colors, linewidth=1, zorder=20, label=label_, alpha=0.5)
    # draw vertical line
    plt.vlines(K, minOA, meanOA - std, color='black', linestyle='solid', zorder=10)
    plt.vlines(K, maxOA, meanOA + std, color='black', linestyle='solid', zorder=10)
    plt.hlines(meanOA, K - barwidth / 2, K + barwidth / 2, color='blue', linestyle='solid', zorder=30)
    plt.hlines(minOA, K - barwidth / 4, K + barwidth / 4, color='black', linestyle='solid', zorder=10)
    plt.hlines(maxOA, K - barwidth / 4, K + barwidth / 4, color='black', linestyle='solid', zorder=10)
    # 设置图例
    legend = plt.legend(loc="lower right", prop=font1, ncol=3, fontsize=24)


"""draw candles for fast adaption performance"""
# K, meanOA, maxOA, minOA, std = read_statistic("C:\\Users\\lichen\\OneDrive\\桌面\\fast_adaption_candle.xlsx")
# colors = ['magenta', 'cyan', 'b', 'g', 'r']
# labels = ['L=1', 'L=2', 'L=3', 'L=4', 'L=5']
# pos = [-2, -1, 0, 1, 2]
# for i in range(5):
#     plot_candle1(K[i], meanOA[i], maxOA[i], minOA[i], std[i], colors[i], labels[i], pos[i])
# # plt.show()
# plt.savefig("C:\\Users\\lichen\\OneDrive\\桌面\\candle.pdf")
