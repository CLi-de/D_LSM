#!/usr/bin/env pytho
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2023/4/19 14:32
# @File    : plot.py
# @annotation

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
# from unsupervised_pretraining.dbn_.models import SupervisedDBNClassification
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
    '''设置长宽'''
    plt.figure(figsize=(10, 4.5))

    # plt.xlabel("Subtasks", fontdict=font1)
    plt.ylabel("Mean accuracy(%)", fontdict=font1)

    '''设置刻度'''
    plt.ylim((60, 100))
    my_y_ticks = np.arange(60, 100, 5)
    plt.yticks(my_y_ticks)
    my_x_ticks = [i for i in range(1, 30, 3)]
    my_x_ticklabel = [str(1991 + i) for i in range(1, 30, 3)]
    plt.xticks(ticks=my_x_ticks, labels=my_x_ticklabel)
    '''格网设置'''
    plt.grid(linestyle="--")

    # x_ = [i + 1 for i in range(arr.shape[0])]
    x_ = np.arange(1, 29, 1)  # para: start, stop, step

    '''draw scatters'''
    # S1 = plt.scatter(x_, arr[:, 0], label="L=1", c="none", s=20, edgecolors='magenta')
    # S2 = plt.scatter(x_, arr[:, 1], label="L=2", c="none", s=20, edgecolors='cyan')
    # S3 = plt.scatter(x_, arr[:, 2], label="L=3", c="none", s=20, edgecolors='b')
    # S4 = plt.scatter(x_, arr[:, 3], label="L=4", c="none", s=20, edgecolors='g')
    # S5 = plt.scatter(x_, arr[:, 4], label="L=5", c="none", s=20, edgecolors='r')
    '''draw lines'''
    L1 = plt.plot(x_, arr[:, 0], color="gold", linestyle=":", marker='o',
                  linewidth=1.5, label="L=1", markerfacecolor='white', ms=7)
    L2 = plt.plot(x_, arr[:, 1], color="c", linestyle=":", marker='^',
                  linewidth=1.5, label="L=2", markerfacecolor='white', ms=8)
    L3 = plt.plot(x_, arr[:, 2], color="b", linestyle=":", marker='s',
                  linewidth=1.5, label="L=3", markerfacecolor='white', ms=7)
    L4 = plt.plot(x_, arr[:, 3], color="g", linestyle=":", marker='p',
                  linewidth=1.5, label="L=4", markerfacecolor='white', ms=9)
    L5 = plt.plot(x_, arr[:, 4], color="r", linestyle=":", marker='*',
                  linewidth=1.5, label="L=5", markerfacecolor='white', ms=10)

    # plt.fill_between(x_, L1, L5,  # 上限，下限
    #                  # facecolor='green',  # 填充颜色
    #                  # edgecolor='red',  # 边界颜色
    #                  alpha=0.3)  # 透明度

    '''设置图例'''
    legend = plt.legend(loc="lower left", prop=font2, ncol=3)


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


def RF_fit_pred(x_train, x_test, y_train, y_test):
    classifier = RandomForestClassifier(n_estimators=200, max_depth=None)
    classifier.fit(x_train, y_train)
    return classifier.predict_proba(x_test)


def plot_auroc(n_times, y_score_SVM, y_score_MLP, y_score_RF, y_score_proposed, y_test, y_test_proposed):
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
                 label=method + '(area = {0:0.3f})'''.format(mean_auc),
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
    plt.xlabel("Number of iterations", fontdict=font1)
    plt.ylabel("OA(%)", fontdict=font2)

    my_x_ticks = [1, 2, 3, 4, 5]
    my_x_ticklabels = ['1', '2', '3', '4', '5']
    plt.xticks(ticks=my_x_ticks, labels=my_x_ticklabels, fontsize=14, fontdict=font2)

    plt.ylim((65, 100))
    my_y_ticks = np.arange(65, 100, 5)
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

    # 绘制趋势曲线

    # trend curve
    model = make_interp_spline(K, meanOA)
    xs = np.linspace(1, 5, 500)
    ys = model(xs)
    plt.plot(xs, ys, color=color_, linestyle='--', lw=1.5, label=label_)

    # 设置图例
    legend = plt.legend(loc="lower right", prop=font1, ncol=3, fontsize=24, ncols=2)


"""draw AUR"""
# print('drawing ROC...')
# x, y = read_f_l_csv('data_src/samples.csv')
# y_score_SVM, y_score_MLP, y_score_RF, y_score_proposed, y_test_, y_test_proposed = [], [], [], [], [], []
# n_times = 5
# for i in range(n_times):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.75, test_size=.03, shuffle=True)
#     """fit and predict"""
#     # for other methods
#     y_score_SVM.append(SVM_fit_pred(x_train, x_test, y_train, y_test))
#     y_score_MLP.append(MLP_fit_pred(x_train, x_test, y_train, y_test))
#     y_score_RF.append(RF_fit_pred(x_train, x_test, y_train, y_test))
#     y_test_.append(y_test)
#     # for proposed-
#     tmp = pd.read_excel('proposed_test' + str(i) + '.xlsx').values.astype(np.float32)
#     y_score_proposed.append(tmp[:, 1:3])
#     y_test_proposed.append(tmp[:, -1])
# # draw roc
# plt.clf()
# plot_auroc(n_times, y_score_SVM, y_score_MLP, y_score_RF, y_score_proposed, y_test_, y_test_proposed)
# plt.savefig('ROC.pdf')
# plt.show()
# print('finish drawing ROC')

"""draw scatters for fast adaption performance"""
filename = "C:\\Users\\lichen\\OneDrive\\桌面\\scatters.csv"
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

"""draw candles for fast adaption performance"""
# K, meanOA, maxOA, minOA, std = read_statistic("C:\\Users\\lichen\\OneDrive\\桌面\\candles.xlsx")
# colors = ['b', 'g', 'r']
# labels = ['1992', '2008', '2017']
# pos = [-1, 0, 1]
# for i in range(3):
#     plot_candle1(K[i], meanOA[i], maxOA[i], minOA[i], std[i], colors[i], labels[i], pos[i])
# plt.savefig("C:\\Users\\lichen\\OneDrive\\桌面\\candle.pdf")
# plt.show()
