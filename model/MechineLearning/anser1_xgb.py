# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(zhangqiude)s
"""

import random

import pandas as pd

pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')
from process.augmentation import get_more_data
from model.Tools.tools import std, change_Glucose, get_day_list

import xgboost as xgb

## 参数搜索和评价的
from sklearn.model_selection import KFold
Glucose_Value = 7.5
TRAIN_NUM = 16
input_path = '../../input/'
file_names = ['001_new.csv', '002_new.csv', '003_new.csv', '004_new.csv',
              '005_new.csv', '006_new.csv', '007_new.csv', '008_new.csv',
              '009_new.csv', '010_new.csv', '011_new.csv', '012_new.csv',
              '013_new.csv', '014_new.csv', '015_new.csv', '016_new.csv']

def main():
    # 随机选择8个文件
    train_files = random.sample(file_names, 8)
    test_files = list(set(file_names) - set(train_files))
    print('训练集', train_files)
    print('测试集', test_files)
    # 创建一个空的DataFrame用于存储训练集数据
    Train_data = pd.DataFrame()
    Test_data = pd.DataFrame()
    # 逐个读取并合并训练集CSV文件
    for file_name in train_files:
        file_path = input_path + file_name
        df = pd.read_csv(file_path, sep=',', na_values='NULL', )
        df = get_more_data(df, 60, 0.4)
        Train_data = pd.concat([Train_data, df], axis=0, ignore_index=True)
    for file_name in test_files:
        file_path = input_path + file_name

        df = pd.read_csv(file_path, sep=',', na_values='NULL', )
        Test_data = pd.concat([Test_data, df], axis=0, ignore_index=True)
    # 数据标准化
    Train_data = change_Glucose(Train_data)
    Test_data = change_Glucose(Test_data)
    Train_data = std(Train_data)
    Test_data = std(Test_data)
    # display(Train_data)
    # display(Test_data)

    # 划分训练数据和验证数据
    #train, val = train_test_split(Train_data, test_size=0.3, random_state=42)
    # 获取训练数据和验证数据的特征和目标变量
    train_x = Train_data.iloc[:, :-2].values
    train_y = Train_data.iloc[:, -1].values
    #val_x = val.iloc[:, :-2].values
    #val_y = val.iloc[:, -1].values
    # train_x = Train_data.iloc[:, :-2].values
    # train_y = Train_data.iloc[:, -1].values
    xgr = xgb.XGBRegressor(n_estimators=120, learning_rate=0.1, gamma=0, subsample=0.8,
                           colsample_bytree=0.9, max_depth=6)  # ,objective ='reg:squarederror'
    # display(train_x)
    # display(train_y)
    # 进行五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 42似乎是一个经验选择
    accuracy_scores = []
    for train_index, val_index in kf.split(train_x):
        # 划分训练数据和交叉验证数据
        fold_train_x, fold_val_x = train_x[train_index], train_x[val_index]
        fold_train_y, fold_val_y = train_y[train_index], train_y[val_index]

        # 训练模型，并在验证集上进行早停
        xgr.fit(fold_train_x, fold_train_y, eval_set=[(fold_val_x, fold_val_y)], eval_metric='mae', verbose=30, early_stopping_rounds=15, )

        # 在验证集上进行预测
        val_pred = xgr.predict(fold_val_x)
        list_1 = []  # 统计大于7.8的个数
        list_true = []
        # 计算准确率
        for i in range(len(fold_val_y)):
            if fold_val_y[i] < 7.5:
                list_true.append(1)
            else:
                list_true.append(0)
            if val_pred[i] < 7.5:
                list_1.append(1)
            else:
                list_1.append(0)
        rate = 0
        for i in range(len(fold_val_y)):
            if list_1[i] == list_true[i]:
                rate += 1
        rate = rate / len(fold_val_y)
        print('val rate:', rate)

    # 输出五折交叉验证的平均准确率
    #mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    #print("Mean Accuracy:", mean_accuracy)
    # xgboost模型

    # xgr.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='mae', verbose=30, early_stopping_rounds=20, )

    print("train finish!")

    test_x = Test_data.iloc[:, :-2].values
    test_y = Test_data.iloc[:, -1].values

    result = xgr.predict(test_x)
    print('test finish!')

    list_1 = []  # 统计大于7.8的个数
    list_true = []
    for i in range(len(test_y)):
        if test_y[i] < 7.5:
            list_true.append(1)
        else:
            list_true.append(0)
        if result[i] < 7.5:
            list_1.append(1)
        else:
            list_1.append(0)
    rate = 0
    for i in range(len(test_y)):
        if list_1[i] == list_true[i]:
            rate += 1
    rate = rate / len(test_y)
    print(rate)

    # 计算天数差异
    Test_day_list = get_day_list(Test_data)  # 获取了测试集的天数列表
    pred_day_list = [0 for i in range(len(Test_day_list))]  # 预测值的每天高糖次数
    y_day_list = [0 for i in range(len(Test_day_list))]  # 真实值的每天高糖次数
    print(len(result), len(test_y))
    for i in range(len(Test_data)):
        ls_ = [Test_data.iloc[i, 0], Test_data.iloc[i, 1]]
        index_u = Test_day_list.index(ls_)  # 获取了该数据对应的天数下标
        if result[i] > Glucose_Value:
            pred_day_list[index_u] += 1
        if test_y[i] > Glucose_Value:
            y_day_list[index_u] += 1
    print('预测高糖发生次数(每天)', pred_day_list)
    print('实际高糖发生次数(每天)', y_day_list)
    sub_day = 0
    all_day_pred = 0
    all_day_y = 0
    for i in range(len(Test_day_list)):
        all_day_pred += pred_day_list[i]
        all_day_y += y_day_list[i]
        sub_day += abs(pred_day_list[i] - y_day_list[i])
    # sub_day = sub_day / len(Test_day_list)
    print('累计金标差异天数：', sub_day)



if __name__ == '__main__':
    main()
