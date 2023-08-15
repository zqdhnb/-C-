# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(zhangqiude)s
"""
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn
from IPython.display import display, clear_output
import time

warnings.filterwarnings('ignore')
# %matplotlib inline

## 数据处理
from sklearn import preprocessing

## 数据降维处理的
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA

## 模型预测的
import lightgbm as lgb
import xgboost as xgb

## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

root_path = '../../data/'


input_path = '../../output/001/'

def Timestamp_to_date(data):
    # 时间属性划分
    df = data['Timestamp']
    df1 = df.str.split(' ', expand=True)
    df1.columns = ['date1', 'date2']

    df11 = df1['date1'].str.split('-', expand=True)
    df11.columns = ['date', 'month', 'day']

    df12 = df1['date2'].str.split(':', expand=True)
    df12.columns = ['hour', 'minute', 'second']

    # 这里需要考虑是否将月份作为训练属性 暂时先放进去
    df1 = pd.concat([df11['month'], df11['day'], df12], axis=1)

    data.drop(columns='Timestamp', inplace=True)
    data = pd.concat([df1, data], axis=1)
    return data

def std(Train_data):
    # 数据标准化 采用均值标准差
    mean = Train_data.mean()
    std = Train_data.std()
    # print(mean, std)
    for i in range(len(Train_data)):
        for j in range(5, 17):
            Train_data.iloc[i, j] = (float(Train_data.iloc[i, j]) - float(mean[j])) / float(std[j])
        Train_data.iloc[i, -1] = float(Train_data.iloc[i, -1]) / 18  # 标签
    return Train_data


def build_model_lgb(x_train,y_train):
    estimator = lgb.LGBMRegressor(num_leaves=127,n_estimators = 150)
    param_grid = {
        'learning_rate': [0.1],#
    }
    gbm = GridSearchCV(estimator, param_grid)

    gbm.fit(x_train, y_train)
    print(f"Best:  {gbm}" )

    return gbm

 #XGB = [xgr() for i in range(8)]
def main():
    # # 训练集
    # Train_data = pd.read_csv(input_path + '001_new.csv', sep=',', na_values='NULL',)
    # #display(Train_data.head())  # 输出

    # # 时间属性划分  干脆保存为文件
    # data = Timestamp_to_date(Train_data)
    # display(data)
    # data.to_csv(input_path + '001_new_new.csv', index=False)
    # 训练集
    Train_data = pd.read_csv(input_path + '001_new_new.csv', sep=',', na_values='NULL', )

    # 数据标准化
    std(Train_data)
    #display(Train_data)

    train_x = Train_data.iloc[:100, :-2].values
    train_y = Train_data.iloc[:100, -1].values
    #display(train_x)
    #display(train_y)

    # xgboost模型
    xgr = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, gamma=0, subsample=0.8,
                           colsample_bytree=0.9, max_depth=7)  # ,objective ='reg:squarederror'
    xgr.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='mae', verbose=30, early_stopping_rounds=20,)

    print("train finish!")

    test_x = Train_data.iloc[:, :-2].values
    test_y = Train_data.iloc[:, -1].values

    result = xgr.predict(test_x)
    print('test finish!')

    list_1 = [] #统计大于7.8的个数
    list_true = []
    for i in range(len(test_y)):
        if test_y[i] < 7.8:
            list_true.append(1)
        else:
            list_true.append(0)
        if result[i] < 7.8:
            list_1.append(1)
        else:
            list_1.append(0)
    rate = 0
    for i in range(len(test_y)):
        if list_1[1] == list_true[i]:
            rate += 1
    rate = rate / len(test_y)
    print(rate)

    #display(test_y)
    #display(result)

if __name__ == '__main__':
    main()











