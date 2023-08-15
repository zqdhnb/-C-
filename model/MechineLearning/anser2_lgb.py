import random

import pandas as pd

from model.Tools.tools import change_one_Glucose, get_day_list

pd.set_option('display.max_columns', None)
import warnings

warnings.filterwarnings('ignore')

## 模型预测的
from lightgbm.sklearn import LGBMRegressor
## 参数搜索和评价的
from sklearn.model_selection import KFold

# root_path = './dataset/'




input_path = '../../input/'
file_names = ['001_new.csv', '002_new.csv', '003_new.csv', '004_new.csv',
              '005_new.csv', '006_new.csv', '007_new.csv', '008_new.csv',
              '009_new.csv', '010_new.csv', '011_new.csv', '012_new.csv',
              '013_new.csv', '014_new.csv', '015_new.csv', '016_new.csv']

DATA_NUM = 16   # 数据集总的个数
ATTRIBUTE_NUM = 18  # 属性数量(包含标签)
TRAIN_NUM = 8   # 训练集的个数
DATE_NUM = 5    # 时间属性的个数
Glucose_WARP = 18  # 血糖转换值
Glucose_Value = 7.5  # 血糖临界值
K = 5  # 交叉验证折数


def std_one(Train_data, DATE_NUM, ATTRIBUTE_NUM):
    mean1 = Train_data.mean()
    std1 = Train_data.std()
    for i in range(len(Train_data)):
        for j in range(DATE_NUM, ATTRIBUTE_NUM - 1):
            Train_data.iloc[i, j] = (float(Train_data.iloc[i, j]) - float(mean1[j])) / float(std1[j])
    return Train_data

def main():

    # 先随机获取八个训练集和八个测试集
    train_files = random.sample(file_names, TRAIN_NUM)
    test_files = list(set(file_names) - set(train_files))
    print('训练集', train_files)
    print('测试集', test_files)
    # 训练数据
    Train_data = [pd.DataFrame() for i in range(TRAIN_NUM)]
    Test_data = pd.DataFrame()
    # print(Train_data[0])

    # 训练集逐个填入pandas
    i = 0
    for file_name in train_files:
        file_path = input_path + file_name

        df = pd.read_csv(file_path, sep=',', na_values='NULL', )
        # 标准化
        df = std_one(df, DATE_NUM, ATTRIBUTE_NUM)
        df = change_one_Glucose(df)
        # display(df)
        Train_data[i] = pd.concat([Train_data[i], df], axis=0, ignore_index=True)
        i += 1
    Train_number = 0  # 训练集的样本个数总和
    for data in Train_data:
        Train_number += len(data)
    alpha = [(len(Train_data[i])/Train_number) for i in range(len(Train_data))]  # 权重参数1
    #
    # 测试集一整个就行
    for file_name in test_files:
        file_path = input_path + file_name
        df = pd.read_csv(file_path, sep=',', na_values='NULL', )
        Test_data = pd.concat([Test_data, df], axis=0, ignore_index=True)


    Test_data = std_one(Test_data, DATE_NUM, ATTRIBUTE_NUM)
    Test_data = change_one_Glucose(Test_data)



    # 划分数据和标签

    train_x = [Train_data[i].iloc[:, :-2].values for i in range(TRAIN_NUM)]
    train_y = [Train_data[i].iloc[:, -1].values for i in range(TRAIN_NUM)]
    # for data in train_x:
    #         print("data!")
    #         display(data)
    lgr = LGBMRegressor(learning_rate=0.0596, boosting_type='gbdt', objective='regression_l1', n_estimators=1000,
                        min_child_samples=168, min_child_weight=0.11, num_leaves=60, max_depth=60, n_jobs=-1,
                        reg_alpha=1, reg_lambda=1)
    LGB = [lgr for i in range(TRAIN_NUM)]  # 8个回归器
    # print(XGB)
    accuracy_scores = [0 for i in range(TRAIN_NUM)]  # 8个验证集的平均准确率
    accuracy_end = [0 for i in range(TRAIN_NUM)]  # 8个验证集的最后一次准确率


    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    # 八个数据集，每个都进行五折交叉验证来学习参数
    for i in range(len(train_x)):
        print('第' + str(i) + '个训练集结果：')
        for train_index, val_index in kf.split(train_x[i]):
            # 划分训练数据和交叉验证数据
            fold_train_x, fold_val_x = train_x[i][train_index], train_x[i][val_index]
            fold_train_y, fold_val_y = train_y[i][train_index], train_y[i][val_index]
            # print(fold_train_x)
            # 训练
            LGB[i].fit(fold_train_x, fold_train_y, eval_set=[(fold_val_x, fold_val_y)],
                       eval_metric='mae', verbose=30, early_stopping_rounds=15, )
            val_pred = LGB[i].predict(fold_val_x)
            list_1 = []  # 统计大于7.8的个数
            list_true = []
            # 计算准确率
            for j in range(len(fold_val_y)):
                if fold_val_y[j] < Glucose_Value:
                    list_true.append(1)
                else:
                    list_true.append(0)
                if val_pred[j] < Glucose_Value:
                    list_1.append(1)
                else:
                    list_1.append(0)
            rate = 0
            for j in range(len(fold_val_y)):
                if list_1[j] == list_true[j]:
                    rate += 1
            rate = rate / len(fold_val_y)
            print('val rate:', rate)
            accuracy_scores[i] += rate/K
        accuracy_end[i] += rate
        print()
    print("train finish!")

    print('alpha:', alpha)
    all_scores = sum(accuracy_scores) - 0.8 * len(accuracy_scores)
    beta = [abs(accuracy_scores[i]-0.8)/all_scores for i in range(TRAIN_NUM)]
    print('beta:', beta)

    all_end = sum(accuracy_end) - 0.8 * len(accuracy_end)
    gamma = [abs(accuracy_end[i]-0.8)/all_end for i in range(TRAIN_NUM)]
    print('gamma:', gamma)

    result = []
    test_x = Test_data.iloc[:, :-2].values
    test_y = Test_data.iloc[:, -1].values
    for lgr in LGB:  # 8个分类器获得8个结果
        result.append(lgr.predict(test_x))
    # 8个取均值结果合并
    import numpy as np
    y_pred_list = []
    arr1 = np.array([0 for i in range(len(test_x))])
    arr2 = np.array([0 for i in range(len(test_x))])
    arr3 = np.array([0 for i in range(len(test_x))])
    arr4 = np.array([0 for i in range(len(test_x))])
    for i in range(DATA_NUM - TRAIN_NUM):
        arr1 = arr1 + np.array(result[i]) / (DATA_NUM - TRAIN_NUM)
        arr2 = arr2 + np.array(result[i]) * alpha[i]
        arr3 = arr3 + np.array(result[i]) * beta[i]
        arr4 = arr4 + np.array(result[i]) * gamma[i]
    y_pred_list.append(list(arr1))
    y_pred_list.append(list(arr2))
    y_pred_list.append(list(arr3))
    y_pred_list.append(list(arr4))

    print('test finish!')
    # 计算准确率
    for y_pred in y_pred_list:
        # print(y_pred[0])
        list_1 = []  # 统计大于7.8的个数
        list_true = []
        for i in range(len(test_y)):
            if test_y[i] < Glucose_Value:
                list_true.append(1)
            else:
                list_true.append(0)
            if y_pred[i] < Glucose_Value:
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
        print(len(y_pred), len(test_y))
        for i in range(len(Test_data)):
            ls_ = [Test_data.iloc[i, 0], Test_data.iloc[i, 1]]
            index_u = Test_day_list.index(ls_)  # 获取了该数据对应的天数下标
            if y_pred[i] > Glucose_Value:
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






















