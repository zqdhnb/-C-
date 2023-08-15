import pandas as pd
import IPython.display as display
import numpy as np
# import torch

DATA_NUM = 16  # 数据集总的个数
TRAIN_NUM = 8  # 训练集的个数
Glucose_Value = 7.5  # 血糖临界值
Glucose_WARP = 18
K = 5  # 交叉验证折数

high_list = [150, 827, 145, 161, 88, 698, 73, 235, 624, 266, 677, 405, 506, 335, 15, 153]
low_list = [2411, 1292, 2156, 2002, 2470, 2149, 2133, 2269, 1680, 1881, 2165, 1763, 1473, 1904, 477, 1981]


def get_day_list(data1):
    # 给定数据，获取不同天的列表 例[[2, 13], [2, 14], [2, 15]]
    # data1为pd.DataFrame()
    day_list = []
    for i in range(len(data1)):
        ls_ = [data1.iloc[i, 0], data1.iloc[i, 1]]
        # print(ls_)
        if ls_ not in day_list:
            day_list.append(ls_)
    return day_list


# 标签处于18不能放在标准化里面
def std(Train_data, Test_data, DATE_NUM, ATTRIBUTE_NUM):
    # 数据标准化 采用均值标准差
    mean1 = Train_data.mean()
    std1 = Train_data.std()
    mean2 = Test_data.mean()
    std2 = Test_data.std()
    for i in range(len(Train_data)):
        for j in range(DATE_NUM, ATTRIBUTE_NUM - 1):
            Train_data.iloc[i, j] = (float(Train_data.iloc[i, j]) - float(mean1[j])) / float(std1[j])
    for i in range(len(Test_data)):
        for j in range(DATE_NUM, ATTRIBUTE_NUM - 1):
            Test_data.iloc[i, j] = (float(Test_data.iloc[i, j]) - float(mean2[j])) / float(std2[j])
    return Train_data, Test_data



def change_Glucose(train_data, test_data, Glucose_WARP=18):  # 每次处理的是一个pd.DataFrame()
    for i in range(len(train_data)):
        train_data.iloc[i, -1] = float(train_data.iloc[i, -1]) / Glucose_WARP  # 标签
    for i in range(len(test_data)):
        test_data.iloc[i, -1] = float(test_data.iloc[i, -1]) / Glucose_WARP  # 标签
    return train_data, test_data


def change_one_Glucose(train_data, Glucose_WARP=18):  # 每次处理的是一个pd.DataFrame()
    for i in range(len(train_data)):
        train_data.iloc[i, -1] = float(train_data.iloc[i, -1]) / Glucose_WARP  # 标签
    return train_data

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


def get_date(data):
    # print(data.dtypes)
    # 合成时间戳 用于排序
    date_list = ['2020-' + str(data.iloc[i, 0]) + '-' + str(data.iloc[i, 1]) + ' ' + str(data.iloc[i, 2]) + ':' + str(
        data.iloc[i, 3]) + ':' + str(data.iloc[i, 4]) for i in range(len(data))]

    data.insert(loc=0, column='date', value=date_list)
    # print(data)
    data['date'] = pd.to_datetime(data['date'])
    # print(data)
    return data


def get_float_data(data):
    data['date'] = pd.to_datetime(data['date'])
    order = ['month', 'day', 'hour', 'minute', 'second',
             'acc_x', 'acc_y', 'acc_z', 'bvp', 'eda', 'calorie', 'total_carb', 'sugar', 'protein',
             'hr', 'ibi', 'temp', 'Glucose Value']
    data[order] = data[order].astype('float32')
    return data


def get_gamma(targets, alpha=2, beta=1):
    """
    输入预测值和真实值，根据高糖数据建立权重矩阵
    """
    m, n, l = targets.shape
    gamma = torch.ones(m, n, l)
    # print(m, n, l)
    for i in range(m):
        for j in range(n):
            for k in range(l):
                if targets[i][j][k] >= Glucose_Value:
                    gamma[i][j][k] = alpha
                else:
                    gamma[i][j][k] = beta
    return gamma


def predict_day_classfication(Test_data, pred_y, true_y):
    # 计算天数差异
    Test_day_list = get_day_list(Test_data)  # 获取了测试集的天数列表
    pred_day_list = [0 for i in range(len(Test_day_list))]  # 预测值的每天高糖次数
    y_day_list = [0 for i in range(len(Test_day_list))]  # 真实值的每天高糖次数
    pred_low = [0 for i in range(len(Test_day_list))]  # 预测值的每天低糖次数
    y_low = [0 for i in range(len(Test_day_list))]  # 真实值的每天低糖次数
    print(len(pred_y), len(true_y))
    for i in range(len(Test_data)):
        ls_ = [Test_data.iloc[i, 0], Test_data.iloc[i, 1]]
        index_u = Test_day_list.index(ls_)  # 获取了该数据对应的天数下标
        if pred_y[i] > 0.5:
            pred_day_list[index_u] += 1
        else:
            pred_low[index_u] += 1
        if true_y[i] > 0.5:
            y_day_list[index_u] += 1
        else:
            y_low[index_u] += 1
    print('高糖数据总和', np.array(y_day_list).sum())
    print('低糖数据总和', np.array(y_low).sum())
    print('预测高糖发生次数(每天)', pred_day_list)
    print('实际高糖发生次数(每天)', y_day_list)
    print('预测低糖发生次数(每天)', pred_low)
    print('实际低糖发生次数(每天)', y_low)
    sub_day = 0
    all_day_pred = 0
    all_day_y = 0
    for i in range(len(Test_day_list)):
        all_day_pred += pred_day_list[i]
        all_day_y += y_day_list[i]
        sub_day += abs(pred_day_list[i] - y_day_list[i])
    # sub_day = sub_day / len(Test_day_list)
    print('累计金标差异天数：', sub_day)


def predict_day(Test_data, pred_y, true_y):
    # 计算天数差异
    Test_day_list = get_day_list(Test_data)  # 获取了测试集的天数列表
    pred_day_list = [0 for i in range(len(Test_day_list))]  # 预测值的每天高糖次数
    y_day_list = [0 for i in range(len(Test_day_list))]  # 真实值的每天高糖次数
    pred_low = [0 for i in range(len(Test_day_list))]  # 预测值的每天低糖次数
    y_low = [0 for i in range(len(Test_day_list))]  # 真实值的每天低糖次数
    print(len(pred_y), len(true_y))
    for i in range(len(Test_data)):
        ls_ = [Test_data.iloc[i, 0], Test_data.iloc[i, 1]]
        index_u = Test_day_list.index(ls_)  # 获取了该数据对应的天数下标
        if pred_y[i] > Glucose_Value:
            pred_day_list[index_u] += 1
        else:
            pred_low[index_u] += 1
        if true_y[i] > Glucose_Value:
            y_day_list[index_u] += 1
        else:
            y_low[index_u] += 1
    print('高糖数据总和', np.array(y_day_list).sum())
    print('低糖数据总和', np.array(y_low).sum())
    print('预测高糖发生次数(每天)', pred_day_list)
    print('实际高糖发生次数(每天)', y_day_list)
    print('预测低糖发生次数(每天)', pred_low)
    print('实际低糖发生次数(每天)', y_low)
    sub_day = 0
    all_day_pred = 0
    all_day_y = 0
    for i in range(len(Test_day_list)):
        all_day_pred += pred_day_list[i]
        all_day_y += y_day_list[i]
        sub_day += abs(pred_day_list[i] - y_day_list[i])
    # sub_day = sub_day / len(Test_day_list)
    print('累计金标差异天数：', sub_day)


def change_to_classfication(data, Glucose_Value=7.5):
    """
    输入处理好的原始数据集，将最后一项根据Glucose_Value改为1或者0
    """
    for i in range(len(data)):
        if data.iloc[i, -1] <= Glucose_Value:
            data.iloc[i, -1] = 0
        else:
            data.iloc[i, -1] = 1
    return data


def main():
    file_path = '../../input/001_new.csv'
    df = pd.read_csv(file_path, sep=',', na_values='NULL', dtype=str)
    data = get_date(data=df)
    data = get_float_data(data)
    data = data.iloc[:, 1:]
    data = change_one_Glucose(data, Glucose_WARP)
    data = change_to_classfication(data)
    print(data)
    # data = torch.randn(2, 1, 128) + 7.5
    # print(get_gamma(data))


if __name__ == '__main__':
    main()
