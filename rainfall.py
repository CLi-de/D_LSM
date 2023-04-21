#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2023/4/21 15:48
# @File    : rainfall.py
# @annotation

import numpy as np
import pandas as pd

'''for landslides'''
# p_data = np.loadtxt('./data_src/p_samples.csv', dtype=str, delimiter=",", encoding='UTF-8-sig')
# station_index = p_data[1:4680, -1]
# years = p_data[1:4680, -4]

rainfall = np.loadtxt('./data_src/stations_prcp.csv', dtype=str, delimiter=",", encoding='UTF-8-sig')
years_ = rainfall[0, 1:]
ID = rainfall[1:, 0]
rainfall_data = pd.DataFrame(rainfall[1:, 1:])
rainfall_data.columns = years_
rainfall_data.index = ID

# rainfall_value = [rainfall_data.loc[[station_index[i]], ['F'+years[i]]].values
#                   for i in range(len(station_index))]
# save excel
# arr = np.array(rainfall_value).reshape(-1, 1).astype(np.float32)
# writer = pd.ExcelWriter('tmp/' + 'p_rainfall.xlsx')
# data_df = pd.DataFrame(arr)
# data_df.to_excel(writer)
# writer.close()

'''for non-landslides and grid samples'''
n_data = np.loadtxt('./data_src/grid_samples.csv', dtype=str, delimiter=",", encoding='UTF-8-sig')
station_index_ = n_data[1:, -1]

rainfall_newest = rainfall[1:, -1]
dict_ = dict(zip(ID, rainfall_newest))


rainfall_value = [dict_[idx] for idx in station_index_]

arr = np.array(rainfall_value).reshape(-1, 1).astype(np.float32)
writer = pd.ExcelWriter('tmp/' + 'grid_rainfall.xlsx')
data_df = pd.DataFrame(arr)
data_df.to_excel(writer)
writer.close()




