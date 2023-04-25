#!/usr/bin/env pytho      
# -*- coding: utf-8 -*-
# @Author  : CHEN Li
# @Time    : 2023/4/21 15:48
# @File    : rainfall.py
# @annotation
"""
This file is mainly for rainfall data sampling.
The daily rainfall data are from https://www.hko.gov.hk/tc/cis/climat.htm and
https://i-lens.hk/hkweather/show_extract_trend.php?mode=S&extract=Y.
Unlike other features, rainfall related data are dynamic varying from year to year.
...
"""
# state: deprecated


import numpy as np
import pandas as pd


# for landslides (positive samples)
def value_p(file_p, file_sta, savefile):
    p_data = np.loadtxt(file_p, dtype=str, delimiter=",", encoding='UTF-8-sig')
    station_index = p_data[1:, -1]
    years = p_data[1:, -6]

    rainfall = np.loadtxt(file_sta, dtype=str, delimiter=",", encoding='UTF-8-sig')
    years_ = rainfall[0, 1:]
    ID = rainfall[1:, 0]
    rainfall_data = pd.DataFrame(rainfall[1:, 1:])
    rainfall_data.columns = years_
    rainfall_data.index = ID

    rainfall_value = [rainfall_data.loc[[station_index[i]], ['F' + years[i]]].values
                      for i in range(len(station_index))]
    # save excel
    arr = np.array(rainfall_value).reshape(-1, 1).astype(np.float32)
    writer = pd.ExcelWriter(savefile)
    data_df = pd.DataFrame(arr)
    data_df.to_excel(writer)
    writer.close()


# for non-landslides rainfall feature valuing, now using average value.
def value_n():
    pass

# n_data = np.loadtxt('./data_src/grid_samples.csv', dtype=str, delimiter=",", encoding='UTF-8-sig')
# station_index_ = n_data[1:, -1]
#
# rainfall_newest = rainfall[1:, -1]
# dict_ = dict(zip(ID, rainfall_newest))
#
#
# rainfall_value = [dict_[idx] for idx in station_index_]
#
# arr = np.array(rainfall_value).reshape(-1, 1).astype(np.float32)
# writer = pd.ExcelWriter('tmp/' + 'grid_rainfall.xlsx')
# data_df = pd.DataFrame(arr)
# data_df.to_excel(writer)
# writer.close()

if __name__ == "__main__":
    # value_p('./data_src/p_samples.csv', './data_src/stations2.csv', 'data_src/p_rainfall.xlsx')

    print('finished!')
