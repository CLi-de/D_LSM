#!/usr/bin/env pytho
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2023/4/19 14:32
# @File    : plot.py
# @annotation

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

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

"""
for figure plotting
"""


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


# def plot_candle(scenes, K, meanOA, maxOA, minOA, std):
#     # 设置框图
#     plt.figure("", facecolor="lightgray")
#     # plt.style.use('ggplot')
#     # 设置图例并且设置图例的字体及大小
#     font1 = {'family': 'Times New Roman',
#              'weight': 'normal',
#              'size': 16,
#              }
#     font2 = {'family': 'Times New Roman',
#              'weight': 'normal',
#              'size': 18,
#              }
#
#     # legend = plt.legend(handles=[A,B],prop=font1)
#     # plt.title(scenes, fontdict=font2)
#     # plt.xlabel("Various methods", fontdict=font1)
#     plt.ylabel("OA(%)", fontdict=font2)
#
#     my_x_ticks = [1, 2, 3, 4, 5]
#     # my_x_ticklabels = ['SVM', 'MLP', 'DBN', 'RF', 'Proposed']
#     plt.xticks(ticks=my_x_ticks, labels='', fontsize=16)
#
#     plt.ylim((60, 100))
#     my_y_ticks = np.arange(60, 100, 5)
#     plt.yticks(ticks=my_y_ticks, fontsize=16)
#
#     colors = ['dodgerblue', 'lawngreen', 'gold', 'magenta', 'red']
#     edge_colors = np.zeros(5, dtype="U1")
#     edge_colors[:] = 'black'
#
#     '''格网设置'''
#     plt.grid(linestyle="--", zorder=-1)
#
#     # draw line
#     # plt.plot(K[0:-1], meanOA[0:-1], color="b", linestyle='solid',
#     #         linewidth=1, label="open", zorder=1)
#     # plt.plot(K[-2:], meanOA[-2:], color="b", linestyle="--",
#     #          linewidth=1, label="open", zorder=1)
#
#     # draw bar
#     barwidth = 0.4
#     plt.bar(K, 2 * std, barwidth, bottom=meanOA - std, color=colors,
#             edgecolor=edge_colors, linewidth=1, zorder=20, label=['SVM', 'MLP', 'DBN', 'RF', 'Proposed'])
#
#     # draw vertical line
#     plt.vlines(K, minOA, maxOA, color='black', linestyle='solid', zorder=10)
#     plt.hlines(meanOA, K - barwidth / 2, K + barwidth / 2, color='black', linestyle='solid', zorder=30)
#     plt.hlines(minOA, K - barwidth / 4, K + barwidth / 4, color='black', linestyle='solid', zorder=10)
#     plt.hlines(maxOA, K - barwidth / 4, K + barwidth / 4, color='black', linestyle='solid', zorder=10)
#
#     # 设置图例
#     legend = plt.legend(loc="lower right", prop=font1, ncol=3, fontsize=24)


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
    L2 = plt.plot(x_, arr[:, 1], color="cyan", linestyle=":", marker='^',
                  linewidth=1.5, label="L=2", markerfacecolor='white', ms=8)
    L3 = plt.plot(x_, arr[:, 2], color="blue", linestyle=":", marker='s',
                  linewidth=1.5, label="L=3", markerfacecolor='white', ms=7)
    L4 = plt.plot(x_, arr[:, 3], color="green", linestyle=":", marker='p',
                  linewidth=1.5, label="L=4", markerfacecolor='white', ms=9)
    L5 = plt.plot(x_, arr[:, 4], color="red", linestyle=":", marker='*',
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


# def plot_histogram(region, measures):
#     '''设置框图'''
#     plt.figure("", facecolor="lightgray")  # 设置框图大小
#     font1 = {'family': 'Times New Roman',
#              'weight': 'normal',
#              'size': 14,
#              }
#     font2 = {'family': 'Times New Roman',
#              'weight': 'normal',
#              'size': 18,
#              }
#     # plt.xlabel("Statistical measures", fontdict=font1)
#     plt.ylabel("Performance(%)", fontdict=font1)
#     plt.title(region, fontdict=font2)
#
#     '''设置刻度'''
#     plt.ylim((60, 90))
#     my_y_ticks = np.arange(60, 90, 3)
#     plt.yticks(my_y_ticks)
#
#     my_x_ticklabels = ['Accuracy', 'Precision', 'Recall', 'F1-score']
#     bar_width = 0.3
#     interval = 0.2
#     my_x_ticks = np.arange(bar_width / 2 + 2.5 * bar_width, 4 * 5 * bar_width + 1, bar_width * 6)
#     plt.xticks(ticks=my_x_ticks, labels=my_x_ticklabels, fontproperties='Times New Roman', size=14)
#
#     '''格网设置'''
#     plt.grid(linestyle="--")
#
#     '''draw bar'''
#     rects1 = plt.bar([x - 2 * bar_width for x in my_x_ticks], height=measures[0], width=bar_width, alpha=0.8,
#                      color='dodgerblue', label="MLP")
#     rects2 = plt.bar([x - 1 * bar_width for x in my_x_ticks], height=measures[1], width=bar_width, alpha=0.8,
#                      color='yellowgreen', label="RF")
#     rects3 = plt.bar([x for x in my_x_ticks], height=measures[2], width=bar_width, alpha=0.8, color='gold', label="RL")
#     rects4 = plt.bar([x + 1 * bar_width for x in my_x_ticks], height=measures[3], width=bar_width, alpha=0.8,
#                      color='peru', label="MAML")
#     rects5 = plt.bar([x + 2 * bar_width for x in my_x_ticks], height=measures[4], width=bar_width, alpha=0.8,
#                      color='crimson', label="proposed")
#
#     '''设置图例'''
#     legend = plt.legend(loc="upper left", prop=font1, ncol=3)
#
#     '''add text'''
#     # for rect in rects1:
#     #     height = rect.get_height()
#     #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height)+'%', ha="center", va="bottom")
#     # for rect in rects2:
#     #     height = rect.get_height()
#     #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height)+'%', ha="center", va="bottom")
#     # for rect in rects3:
#     #     height = rect.get_height()
#     #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height)+'%', ha="center", va="bottom")
#     # for rect in rects4:
#     #     height = rect.get_height()
#     #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height)+'%', ha="center", va="bottom")
#     # for rect in rects5:
#     #     height = rect.get_height()
#     #     plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height)+'%', ha="center", va="bottom")
#
#     plt.savefig("C:\\Users\\hj\\Desktop\\histogram" + region + '.pdf')
#     plt.show()


# def load_data(filepath, dim_input):
#     np.loadtxt(filepath, )
#     data = pd.read_excel(filepath).values.astype(np.float32)
#     attr = data[:, :dim_input]
#     attr = attr / attr.max(axis=0)
#     label = data[:, -1].astype(np.int32)
#     return attr, label


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

    plt.ylim((60, 100))
    my_y_ticks = np.arange(60, 100, 5)
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


def plot_rainfall(f_name):
    '''read data'''
    data = pd.read_excel(f_name, header=None)
    AVD_AR = data.iloc[0:1, 1:-1].values.reshape(-1)
    AVG_AERD = data.iloc[1:2, 1:-1].values.reshape(-1)
    Landslide_num = data.iloc[-1, 1:-1].values.reshape(-1)
    # squeeze to ax2 (2-8)
    Landslide_num = ((8 - 2) * (Landslide_num - Landslide_num.min()) + 2 * (
            Landslide_num.max() - Landslide_num.min())) / Landslide_num.max() - Landslide_num.min()
    # years = [str(1992 + i) for i in range(28)]
    x_ = np.arange(1, 29, 1)  # para: start, stop, step

    '''plotting'''
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 16,
            }
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    lns1 = ax.bar(x_, AVD_AR, label='AR', color='blue')
    ax2 = ax.twinx()
    lns2 = ax2.scatter(x_, AVG_AERD, c="white", marker='*', s=100,
                       edgecolors='red', linewidths=1, label='AERD')
    lns3 = ax2.scatter(x_, Landslide_num, c="white", marker='o', s=75,
                       edgecolors='red', linewidths=1, label='Number of landslides')

    lns_ = [lns1, lns2, lns3]
    labs = [l.get_label() for l in lns_]
    ax.legend(lns_, labs, loc="upper right", prop=font)

    ax.set_xlabel('Years', fontdict=font)
    ax.set_ylabel('Annual Rainfall(AR)', fontdict=font)
    ax2.set_ylabel('Annual Extreme Rainfall Days(AERD)', fontdict=font)
    ax.set_ylim(AVD_AR.min() - 100, AVD_AR.max() + 200)
    ax2.set_ylim(AVG_AERD.min() - 1, AVG_AERD.max() + 1)

    my_x_ticks = [i for i in range(1, 30, 3)]
    my_x_ticklabel = [str(1991 + i) for i in range(1, 30, 3)]
    plt.xticks(ticks=my_x_ticks, labels=my_x_ticklabel, size=16)

    plt.yticks(fontproperties='Times New Roman', size=16)
    plt.xticks(fontproperties='Times New Roman', size=16)

    # ax = plt.gca()  # gca:get current axis得到当前轴
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    plt.savefig("C:\\Users\\lichen\\OneDrive\\桌面\\rainfall_DV.pdf")
    plt.show()

    '''others'''


def plot_AR_DV_2008(f_name):
    data = pd.read_excel(f_name, header=None)
    MR_2008 = data.iloc[4:5, 1:13].values.reshape(-1)
    TSdate_2008 = data.iloc[9:10, 1:7].values.reshape(-1)
    P1_2008 = data.iloc[10:11, 1:7].values.reshape(-1)
    # P2_2008 = data.iloc[11:12, 1:7].values.reshape(-1)
    P2_2008 = data.iloc[12:13, 1:7].values.reshape(-1)
    P3_2008 = data.iloc[13:14, 1:7].values.reshape(-1)

    x_ = np.arange(1, 13, 1)
    TSdate_2008 = TSdate_2008 - 20080000
    x_TS = [int(TSdate_2008[i] / 100) + TSdate_2008[i] % 100 / 30
            for i in range(len(TSdate_2008))]
    x_TS = np.array(x_TS)
    '''plotting'''
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 18,
            }
    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    lns1 = ax.bar(x_, MR_2008, label='AR', color='blue')
    ax2 = ax.twinx()

    L1 = ax2.plot(x_TS, P1_2008, color="red", linestyle="--", marker='o',
                  linewidth=2, label="P1", markerfacecolor='white', ms=7)
    # L2 = ax2.plot(x_TS, P2_2008, color="red", linestyle="--", marker='*',
    #               linewidth=2, label="P2", markerfacecolor='white', ms=7)
    L2 = ax2.plot(x_TS, P2_2008, color="green", linestyle="--", marker='s',
                  linewidth=2, label="P2", markerfacecolor='white', ms=7)
    L3 = ax2.plot(x_TS, P3_2008, color="gold", linestyle="--", marker='^',
                  linewidth=2, label="P3", markerfacecolor='white', ms=7)

    '''填充阴影'''
    ax.fill_between([0, 3.5], 0, 2500, facecolor='g', alpha=0.1)
    ax.fill_between([3.5, 8.5], 0, 2500, facecolor='r', alpha=0.1)
    ax.fill_between([8.5, 13], 0, 2500, facecolor='g', alpha=0.1)

    lns = L1 + L2 + L3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="upper right", prop=font)

    '''x,y labels'''
    ax.set_xlabel('2008', fontdict=font)
    ax.set_ylabel('Monthly Rainfall(MR)', fontdict=font)
    ax2.set_ylabel('Displacement', fontdict=font)
    '''y ticks'''
    ax.set_ylim(MR_2008.min(), MR_2008.max() + 1000)
    ax2.set_ylim(-250, 100)
    '''x ticks'''
    my_x_ticks = [i for i in range(1, 13, 1)]
    my_x_ticklabel = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                      'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(ticks=my_x_ticks, labels=my_x_ticklabel)
    '''x,y ticks font'''
    ax.tick_params(labelsize=16)
    ax2.tick_params(labelsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # 旋转
    labels = ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.grid(visible=False)
    ax2.grid(visible=False)
    plt.tight_layout()
    plt.savefig("C:\\Users\\lichen\\OneDrive\\桌面\\rainfall_DV_2008.pdf")
    plt.show()


def plot_AR_DV_2017(f_name):
    data = pd.read_excel(f_name, header=None)
    MR_2017 = data.iloc[5:6, 1:13].values.reshape(-1)
    TSdate_2017 = data.iloc[14:15, 1:].values.reshape(-1)
    P1_2017 = data.iloc[15:16, 1:].values.reshape(-1)
    P2_2017 = data.iloc[16:17, 1:].values.reshape(-1)

    x_ = np.arange(1, 13, 1)
    TSdate_2017 = TSdate_2017 - 20170000
    x_TS = [int(TSdate_2017[i] / 100) + TSdate_2017[i] % 100 / 30
            for i in range(len(TSdate_2017))]
    x_TS = np.array(x_TS)
    '''plotting'''
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 18,
            }
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    lns1 = ax.bar(x_, MR_2017, label='AR', color='blue')
    ax2 = ax.twinx()

    # L1 = ax2.plot(x_TS, P1_2017, color="red", linestyle="--", marker='o',
    #               linewidth=2, label="P4", markerfacecolor='white', ms=7)
    # L2 = ax2.plot(x_TS, P2_2017, color="green", linestyle="--", marker='s',
    #               linewidth=2, label="P5", markerfacecolor='white', ms=7)

    arr1 = pd.DataFrame(np.hstack((x_TS.reshape(-1, 1), P1_2017.reshape(-1, 1))), columns=['x', 'y'])
    arr2 = pd.DataFrame(np.hstack((x_TS.reshape(-1, 1), P2_2017.reshape(-1, 1))), columns=['x', 'y'])

    sns.set(style="whitegrid", font_scale=1.2)
    P1 = sns.regplot(x='x', y='y', data=arr1,
                     marker='p', label="P4",
                     order=3,  # 默认为1，越大越弯曲
                     scatter_kws={'s': 60, 'color': 'red'},  # 设置散点属性，参考plt.scatter
                     line_kws={'linestyle': '--', 'color': 'red'}  # 设置线属性，参考 plt.plot
                     )
    P2 = sns.regplot(x='x', y='y', data=arr2,
                     marker='*', label="P5",
                     order=3,  # 默认为1，越大越弯曲
                     scatter_kws={'s': 60, 'color': 'green', },  # 设置散点属性，参考plt.scatter
                     line_kws={'linestyle': '--', 'color': 'green'}  # 设置线属性，参考 plt.plot
                     )

    '''填充阴影'''
    ax.fill_between([0, 4.5], 0, 1000, facecolor='g', alpha=0.1)
    ax.fill_between([4.5, 9.5], 0, 1000, facecolor='r', alpha=0.1)
    ax.fill_between([9.5, 13], 0, 1000, facecolor='g', alpha=0.1)

    P1.legend(loc='upper right', prop=font)
    '''x,y labels'''
    ax.set_xlabel('2017', fontdict=font)
    ax.set_ylabel('Monthly Rainfall(MR)', fontdict=font)
    ax2.set_ylabel('Displacement', fontdict=font)
    '''y ticks'''
    ax.set_ylim(MR_2017.min(), MR_2017.max() + 500)
    ax2.set_ylim(-150, 50)
    '''x ticks'''
    my_x_ticks = [i for i in range(1, 13, 1)]
    my_x_ticklabel = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                      'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(ticks=my_x_ticks, labels=my_x_ticklabel)
    '''x,y ticks font'''
    ax.tick_params(labelsize=16)
    ax2.tick_params(labelsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # 旋转
    labels = ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.grid(visible=False)
    ax2.grid(visible=False)
    plt.tight_layout()
    plt.savefig("C:\\Users\\lichen\\OneDrive\\桌面\\rainfall_DV_2017.pdf")
    plt.show()


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
# filename = "C:\\Users\\lichen\\OneDrive\\桌面\\scatters.csv"
# arr = np.loadtxt(filename, dtype=float, delimiter=",", encoding='utf-8-sig')
# plot_scatter(arr)
# plt.savefig("C:\\Users\\lichen\\OneDrive\\桌面\\scatters.pdf")
# plt.show()

"""draw lines for fast adaption performance"""
# filename = "C:\\Users\\lichen\\OneDrive\\桌面\\fast_adaption1.csv"
# arr = np.loadtxt(filename, dtype=float, delimiter=",", encoding='utf-8-sig')
# plot_lines(arr)
# plt.savefig("C:\\Users\\lichen\\OneDrive\\桌面\\broken.pdf")
# plt.show()

"""draw candles for fast adaption performance"""
# K, meanOA, maxOA, minOA, std = read_statistic("C:\\Users\\lichen\\OneDrive\\桌面\\candles.xlsx")
# colors = ['b', 'g', 'r']
# labels = ['1999', '2008', '2017']
# pos = [-1, 0, 1]
# for i in range(3):
#     plot_candle1(K[i], meanOA[i], maxOA[i], minOA[i], std[i], colors[i], labels[i], pos[i])
# plt.savefig("C:\\Users\\lichen\\OneDrive\\桌面\\candle.pdf")
# plt.show()

"""draw rainfall and deformation time series (1992, 2008, 2017)"""
# f_name = "C:/Users/lichen/OneDrive/桌面/prec_DV.xlsx"
# plot_rainfall(f_name)
# plot_AR_DV_2008(f_name)
# plot_AR_DV_2017(f_name)

"""plot density scatter(LS-LIF)"""
# f_LS = './tmp/16th_LSM.xlsx'
# f_grids = './data_sup/grid_samples_2008.csv'
# arr_LS = np.array(pd.read_excel(f_LS, usecols=[3]))
# arr_grids = np.array(pd.read_csv(f_grids, skiprows=0))[:, :15]
# '''normalization'''
# mean = np.mean(arr_grids, axis=0)
# std = np.std(arr_grids, axis=0)
# arr_grids = (arr_grids - mean) / std
# y = arr_LS
# colors = np.random.rand(arr_grids.shape[0])
# for i in range(15):
#     x = arr_grids[:, i]
#     f = plt.figure(figsize=(8, 8))
#     ax0 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
#     sc = ax0.scatter(x, y, c=colors)
#     cbar = f.colorbar(sc)
#     cbar.set_label("Z", fontsize=15)
#     plt.show()


"""plot scatter(AR-AERD-landslides)"""
p_data = pd.read_csv('./data_src/p_samples.csv')
years = np.unique(np.array(p_data.iloc[1:, -5]))
# groups_p = p_data.groupby('year')
# count_list = []
# for year in years:
#     p_samples_ = groups_p.get_group(year).reset_index().values
#     count_list.append(len(p_samples_))
# count_arr = np.array(count_list)
# count_arr = (count_arr - np.mean(count_arr)) / np.std(count_arr)
# count_arr[16] = count_arr[16] - 4
# count_arr[1] = count_arr[1] - 1.5
# AERDrank_arr = [2, 3, 3, 4, 7, 1, 2, 2, 2, 5, 3, 2, 2, 2, 2, 7, 2, 2, 4, 2, 7, 4, 2, 5, 1, 3, 7, 4]
AERDrank_ratio = [0.1, 0.08, 0.08, 0.08, 0.01, 0.15, 0.11, 0.1, 0.15, 0.07,
                  0.09, 0.15, 0.1, 0.12, 0.1, 0.05, 0.18, 0.12, 0.10, 0.10,
                  0.01, 0.08, 0.09, 0.01, 0.09, 0.09, 0.05, 0.1]  # calculated from bar.pdf
AR_AVG = np.array([1751, 1905, 2073, 1697, 1614, 2267, 1919, 1674, 2024, 2545, 1945, 1854, 1370, 2203, 2145,
                   1535, 2838, 1698, 1981, 1337, 1644, 2311, 1973, 1820, 2473, 1919, 1784, 2075])
# AERD_AVG = np.array([4.5, 5.3, 4.6, 2.33, 1, 3.25, 3.75, 5.5, 4.75, 6, 3.25, 3.75, 2.8, 6, 4.4, 2,
#                      10.6, 2.6, 3.2, 1.4, 1, 4.2, 3, 3.2, 2.6, 3.2, 2.4, 1.8])
AERD_AVG = np.array([4., 4.8, 4.1, 1.83, 0.5, 2.75, 3.25, 5., 4.25, 6, 3.25, 3.75, 2.8, 6, 4.4, 2,
                     10.6, 2.6, 3.2, 1.4, 1, 4.2, 3, 3.2, 2.6, 3.2, 2.4, 1.8])
'''plot'''
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 12,
         }
colors = AERDrank_ratio
colors_ = np.array(AERDrank_ratio)
# x = AR_AVG
x = years
y = AERD_AVG
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 1, 1)
sc = ax.scatter(x, y, c=colors, marker='s', s=250, cmap='jet', vmin=0.01, vmax=0.18,
                edgecolors='black', linewidths=2, label='AERD')
cbar = fig.colorbar(sc)
cbar.set_label("Importance ratio of AERD (%)", fontsize=18, font=font1)
'''colorbar ticks'''
cbar.set_ticks([0.01, 0.18])  # 设置刻度值
cbar.set_ticklabels(['1', '18'], font=font1)  # 设置刻度标签
cbar.ax.yaxis.set_ticks_position('right')  # 设置刻度的位置
cbar.ax.yaxis.set_label_position('right')  # 设置标签的位置
# 在散点上添加注释文本
labels = [str(years[i]) for i in range(len(years))]
for label, x_val, y_val in zip(labels, x, y):
    plt.annotate(
        label,
        xy=(x_val, y_val),
        xytext=(-10, -10),
        textcoords='offset points',
        ha='left', va='top', font=font2)

'''draw line'''
# trend curve
model = make_interp_spline(x, colors_ * 30 + 5)  # 调位置
xs = np.linspace(1992, 2019, 200)
ys = model(xs)
# plt.plot(xs, ys, color=colors_, linestyle='--', lw=1.5, label=label_)
L1 = plt.plot(xs, ys, color="r", linestyle="solid",
              linewidth=2.5, label="AERD importance")

'''x,y labels'''
ax.set_xlabel('Years', fontdict=font1)
ax.set_ylabel('AERD', fontdict=font1)
'''x,y ticks font'''
ax.tick_params(labelsize=16)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # 旋转
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

# 设置图例
legend = plt.legend(loc="upper right", prop=font1, ncol=1, fontsize=24, ncols=1)
plt.savefig("C:\\Users\\lichen\\OneDrive\\桌面\\AR-AERD-LIF.pdf")
plt.show()
