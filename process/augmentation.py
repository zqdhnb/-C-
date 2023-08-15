# 增加高糖样本比例
# 方法:数据生成式填充


import random
import IPython.display as display
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn
from IPython.display import display, clear_output
import time

warnings.filterwarnings('ignore')

input_path = '../input/'
file_names = ['001_new.csv', '002_new.csv', '003_new.csv', '004_new.csv',
              '005_new.csv', '006_new.csv', '007_new.csv', '008_new.csv',
              '009_new.csv', '010_new.csv', '011_new.csv', '012_new.csv',
              '013_new.csv', '014_new.csv', '015_new.csv', '016_new.csv']
high_list = [150, 827, 145, 161, 88, 698, 73, 235, 624, 266, 677, 405, 506, 335, 15, 153]
low_list = [2411, 1292, 2156, 2002, 2470, 2149, 2133, 2269, 1680, 1881, 2165, 1763, 1473, 1904, 477, 1981]

Glucose_WARP = 18  # 血糖转换值
Glucose_Value = 7.5  # 血糖临界值


# data = pd.DataFrame()
# for file_name in file_names:
#     file_path = input_path + file_name
#     df = pd.read_csv(file_path, sep=',', na_values='NULL', )
#     data = pd.concat([data, df], axis=0, ignore_index=True)
# display(data)


def distribute_GlucoseValue(All_data):
    """
    输入:全体数据 pd.DataFrame()
    输出:非高糖组成的数据 & 高糖组成的数据
    """
    high_GlucoseValue = pd.DataFrame()  # 高糖数据
    low_GlucoseValue = pd.DataFrame()  # 非高糖数
    for i in range(len(All_data)):
        if All_data.iloc[i, -1] / Glucose_WARP > Glucose_Value:
            df2 = All_data.iloc[i]
            high_GlucoseValue = high_GlucoseValue.append(df2, ignore_index=True)
        else:
            df2 = All_data.iloc[i]
            low_GlucoseValue = low_GlucoseValue.append(df2, ignore_index=True)
    # display(high_GlucoseValue)
    return high_GlucoseValue, low_GlucoseValue
def awgn(x, snr):
    # snr 信噪比
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    return x + noise

# def get_more_data(df, snr=20, n = 0.8):
#     high_GlucoseValue, low_GlucoseValue = distribute_GlucoseValue(df)
#     order = ['month', 'day', 'hour', 'minute', 'second',
#              'acc_x', 'acc_y', 'acc_z', 'bvp', 'eda', 'calorie', 'total_carb', 'sugar', 'protein',
#              'hr', 'ibi', 'temp', 'Glucose Value']
#     high_GlucoseValue = high_GlucoseValue[order]
#     low_GlucoseValue = low_GlucoseValue[order]
#     w = int(len(low_GlucoseValue) / len(high_GlucoseValue))  # 每个数据要补齐的个数
#     w = int(w * n)
#     data1 = pd.DataFrame()
#     for i in range(len(high_GlucoseValue)):
#         # 先把每个原始数据填上
#         data1 = data1.append(high_GlucoseValue.iloc[i], ignore_index=True)
#         for j in range(w):
#             data = awgn(high_GlucoseValue.iloc[i], snr)
#             data1 = data1.append(data, ignore_index=True)
#     file = pd.concat([data1, low_GlucoseValue], axis=0, ignore_index=True)
#     return file





def main():
    for file_name in file_names:
        file_path = input_path + file_name
        df = pd.read_csv(file_path, sep=',', na_values='NULL', )
        high_GlucoseValue, low_GlucoseValue = distribute_GlucoseValue(df)
        order = ['month', 'day', 'hour', 'minute', 'second',
                 'acc_x', 'acc_y', 'acc_z', 'bvp', 'eda', 'calorie', 'total_carb', 'sugar', 'protein',
                 'hr', 'ibi', 'temp', 'Glucose Value']
        high_GlucoseValue = high_GlucoseValue[order]
        low_GlucoseValue = low_GlucoseValue[order]
        w = int(len(low_GlucoseValue)/len(high_GlucoseValue))  # 每个数据要补齐的个数
        data1 = pd.DataFrame()
        for i in range(len(high_GlucoseValue)):
            # 先把每个原始数据填上
            data1 = data1.append(high_GlucoseValue.iloc[i], ignore_index=True)
            for j in range(w-1):
                data = awgn(high_GlucoseValue.iloc[i], 0.1)
                data1 = data1.append(data, ignore_index=True)
        file = pd.concat([data1, low_GlucoseValue], axis=0, ignore_index=True)
        display(file)
        break
        # high_GlucoseValue.to_csv('../input_augmentation/' + 'high_' + file_name, index=False)
        # low_GlucoseValue.to_csv('../input_augmentation/' + 'low_' + file_name, index=False)




if __name__ == '__main__':
    main()
