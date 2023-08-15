# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(zhangqiude)s
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from model.DeepLearning.Time_transformer import TransAm
import torch.nn.functional as F

pd.set_option('display.max_columns', None)
import warnings

warnings.filterwarnings('ignore')


## 参数搜索和评价的
from sklearn.model_selection import KFold

TRAIN_NUM = 16

lr = 0.01

# epochs = 20
# gamma = 0.6
# time_step = 12


epochs = 20
gamma = 0.6
time_step = 12

input_path = r'../../input/'
file_names = ['001_new.csv', '002_new.csv', '003_new.csv', '004_new.csv',
              '005_new.csv', '006_new.csv', '007_new.csv', '008_new.csv',
              '009_new.csv', '010_new.csv', '011_new.csv', '012_new.csv',
              '013_new.csv', '014_new.csv', '015_new.csv', '016_new.csv']

def std(Train_data):
    # 数据标准化 采用均值标准差
    mean = Train_data.mean()
    std = Train_data.std()
    # print(mean, std)
    for i in range(len(Train_data)):
        for j in range(5, 17):
            Train_data.iloc[i, j] = (float(Train_data.iloc[i, j]) - float(mean[j])) / float(std[j])
        # Train_data.iloc[i, -1] = float(Train_data.iloc[i, -1]) / 18  # 标签
    return Train_data


def change_Glucose(data):  # 每次处理的是一个pd.DataFrame()
    for i in range(len(data)):
        data.iloc[i, -1] = float(data.iloc[i, -1]) / 18  # 标签
    return data


def split_sequences(df, steps=10):
    sequences = pd.DataFrame()
    # print((df.iloc[0:10, -1].values)/18.0)
    for i in range(0, df.shape[0], 1):
        if i + steps > df.shape[0]:
            now_seq = df[-steps:]
            sequences = pd.concat([sequences, now_seq], axis=0, ignore_index=True)
        else:
            now_seq = df[i:i + steps]
            sequences = pd.concat([sequences, now_seq], axis=0, ignore_index=True)
    return sequences


def get_day_list(data):
    # 给定数据，获取不同天的列表 例[[2, 13], [2, 14], [2, 15]]
    # data1为pd.DataFrame()
    day_list = []
    for i in range(data.shape[0]):
        ls_ = [int(data[i, 0].item()), int(data[i, 1].item())]
        # print(ls_)
        if ls_ not in day_list:
            day_list.append(ls_)
    return day_list


def predict_day_classfication(Test_data, pred_y, true_y):
    # 计算天数差异
    Test_day_list = get_day_list(Test_data)  # 获取了测试集的天数列表
    # print(Test_day_list )
    pred_day_list = [0 for i in range(len(Test_day_list))]  # 预测值的每天高糖次数
    y_day_list = [0 for i in range(len(Test_day_list))]  # 真实值的每天高糖次数
    pred_low = [0 for i in range(len(Test_day_list))]  # 预测值的每天低糖次数
    y_low = [0 for i in range(len(Test_day_list))]  # 真实值的每天低糖次数
    # print(len(pred_y), len(true_y))
    for i in range(len(Test_data)):
        ls_ = [int(Test_data[i, 0].item()), int(Test_data[i, 1].item())]
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
    print('累计金标差异次数：', sub_day)
    print('平均每天金标差异次数：', sub_day/len(Test_day_list))


# 定义transformer模型
class Transformer(nn.Module):
    def __init__(self, C_size=12, d_model=time_step):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, batch_first=True)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(C_size * d_model, 1)

    def forward(self, x):
        out = self.transformer(x, x)  # 输出为[N, 12, 16]
        out = self.flatten(out)  # 扁平化 两维
        out = self.fc(out)
        # out = torch.unsqueeze(out, dim=1)
        return out


def InfoNCELoss(anchor, positive, negative, temperature=1.0):
    # Calculate similarity between anchor and positive samples
    sim_pos = []
    for i in range(positive.shape[0]):
        # print(anchor.shape, positive[i].unsqueeze(0).shape)
        sim_pos.append(torch.cosine_similarity(anchor, positive[i].unsqueeze(0), dim=1) / temperature)
    sim_pos = torch.cat(sim_pos)
    sim_neg = []
    for i in range(negative.shape[0]):
        sim_neg.append(torch.cosine_similarity(anchor, negative[i].unsqueeze(0), dim=1) / temperature)
    sim_neg = torch.cat(sim_neg)
    numerator = torch.exp(sim_pos)
    denominator = torch.sum(torch.exp(sim_neg))
    loss = -torch.log(numerator / (numerator + denominator)).mean()
    return loss


class WeightBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=0.6):
        super(WeightBCEWithLogitsLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, gt):
        eposion = 1e-10
        count_pos = torch.sum(gt) * 1.0 + eposion
        count_neg = torch.sum(1 - gt) * 1.0
        beta = count_neg / count_pos * self.gamma
        beta_back = count_pos / (count_pos + count_neg)
        #### with weight
        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back * bce1(pred, gt)
        return loss


# 定义训练函数
def train_model(model, train_loader, epochs, lr):
    criterion = WeightBCEWithLogitsLoss(gamma=gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        loss_all = 0
        len_input_all = 0
        for inputs, targets in train_loader:
            inputs = inputs.permute(0, 2, 1)
            # print('输入数据形状:', inputs.shape)
            # print('输出数据形状:', targets.shape)
            optimizer.zero_grad()
            # outputs, output1 = model(inputs)
            outputs = model(inputs)
            # print(outputs.shape)
            b_list_true = [0 if target < 7.5 else 1 for target in targets[:, -1]]  # 1为高血糖
            loss = criterion(outputs, torch.tensor(b_list_true, dtype=torch.float32).unsqueeze(-1))
            loss.backward()
            optimizer.step()
            loss_all += loss
            len_input_all += len(inputs)
        loss_all = loss_all / len_input_all
        print('Epoch [{}], Loss: {:.6f}'.format(epoch + 1,  loss_all))


def main():

    # model = Transformer(C_size=12, d_model=16)
    model = TransAm(feature_size=time_step, num_layers=1, dropout=0.1, l_num=12)
    # 随机选择8个文件
    train_files = random.sample(file_names, 8)
    test_files = list(set(file_names) - set(train_files))
    print('训练集', train_files)
    print('测试集', test_files)
    # 创建一个空的DataFrame用于存储训练集数据
    train_data = []
    test_data = []
    # 逐个读取并合并训练集CSV文件
    for file_name in train_files:
        file_path = input_path + file_name

        df = pd.read_csv(file_path, sep=',', na_values='NULL', )
        data = np.array(df)
        data = data[:, 5:]
        data[:, -1] = data[:, -1] / 18.0
        sequences = [data[i:i + time_step] for i in range(0, len(data) - time_step + 1)]
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
        train_data.append(sequences_tensor)
        # print(Train_data.shape)
    for file_name in test_files:
        file_path = input_path + file_name

        df = pd.read_csv(file_path, sep=',', na_values='NULL', )
        data = np.array(df)
        data = data[:, :]
        data[:, -1] = data[:, -1] / 18.0
        sequences = [data[i:i + time_step] for i in range(0, len(data) - time_step + 1)]
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
        test_data.append(sequences_tensor)
        # print(Test_data.shape)
    train_data_tensor = torch.cat(train_data)
    test_data_tensor = torch.cat(test_data)
    print(train_data_tensor.shape, test_data_tensor.shape)

    train_input = train_data_tensor[:, :, :-1]
    train_target = train_data_tensor[:, :, -1]
    test_input = test_data_tensor[:, :, :-1]
    test_target = test_data_tensor[:, :, -1]


    train_data_mean = torch.mean(train_input)
    train_data_std = torch.std(train_input)
    test_data_mean = torch.mean(test_input[:, :, 5:])
    test_data_std = torch.std(test_input[:, :, 5:])

    # 对数据进行均值和标准差标准化
    train_input = (train_input - train_data_mean) / train_data_std
    test_input[:, :, 5:] = (test_input[:, :, 5:] - test_data_mean) / test_data_std
    # print(test_input[0, :20, :5])

    Train_dataset = TensorDataset(train_input, train_target)
    Test_dataset = TensorDataset(test_input, test_target)

    test_loader = DataLoader(Test_dataset, batch_size=1, shuffle=False)  # 测试集不打乱

    """
    # 原始输入数据形状
    # torch.Size([32, 16, 12])
    # 32是batch_size 16是时间步长 12是属性数量
    """

    # 进行交叉验证
    kf = KFold(n_splits=2, shuffle=True, random_state=42)  # 42似乎是一个经验选择   # 训练集打乱
    accuracy_scores = []
    for train_index, val_index in kf.split(Train_dataset):
        # 划分训练数据和交叉验证数据
        # fold_train_x, fold_val_x = train_x[train_index], train_x[val_index]
        # fold_train_y, fold_val_y = train_y[train_index], train_y[val_index]
        train_dataset = torch.utils.data.Subset(Train_dataset, train_index)
        val_dataset = torch.utils.data.Subset(Train_dataset, val_index)

        # 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        # 初始化模型并进行训练
        # model = LSTMModel(input_size, hidden_size, output_size)
        train_model(model, train_loader, epochs, lr)


        # 在验证集上进行评估
        criterion = WeightBCEWithLogitsLoss()
        list_1 = []  # 统计大于7.8的个数
        list_true = []
        with torch.no_grad():
            total_loss = 0.0
            for inputs, targets in val_loader:
                inputs = inputs.permute(0, 2, 1)
                # outputs, output1 = model(inputs)
                outputs = model(inputs)
                b_list_true = [0 if target < 7.5 else 1 for target in targets[:, -1]]
                loss = criterion(outputs, torch.tensor(b_list_true, dtype=torch.float32).unsqueeze(-1))
                total_loss += loss.item()

                b_list_1 = [0 if torch.sigmoid(output) < 0.5 else 1 for output in outputs]
                list_true = list_true + b_list_true
                list_1 = list_1 + b_list_1
            avg_loss = total_loss / len(val_dataset)
            print(f"Average Loss = {avg_loss}")
            rate = 0
            h_rate, l_rate = 0, 0
            list_true = np.array(list_true)
            list_1 = np.array(list_1)
            c = [int(aa == bb) for aa, bb in zip(list_true, list_1)]
            c = np.array(c)
            rate = sum(c) / len(list_true)
            h_rate = sum(c[list_true == 1]) / len(c[list_true == 1])
            l_rate = sum(c[list_true == 0]) / len(c[list_true == 0])
            print('val rate:', rate)
            print('h_rate:', h_rate, )
            print('l_rate:', l_rate)
            print('实际高糖发生次数:', sum(list_true), '预测高糖发生次数:', sum(list_1))

    criterion = WeightBCEWithLogitsLoss()
    list_1 = []  # 统计大于7.8的个数
    list_true = []
    with torch.no_grad():
        total_loss = 0.0
        a = []
        for inputs, targets in test_loader:
            a.append(inputs[:, :, :5])
            # print(inputs[0, :20, :5])  # 顺序是一样的
            x = inputs[:, :, 5:].permute(0, 2, 1)
            # outputs, output1 = model(inputs[:, :, 5:])
            outputs = model(x)
            b_list_true = [0 if target < 7.5 else 1 for target in targets[:, -1]]
            loss = criterion(outputs, torch.tensor(b_list_true, dtype=torch.float32).unsqueeze(-1))

            total_loss += loss.item()

            b_list_1 = [0 if torch.sigmoid(output) < 0.5 else 1 for output in outputs]

            list_true = list_true + b_list_true
            list_1 = list_1 + b_list_1
        print(len(a), test_input.shape)
        avg_loss = total_loss / test_data_tensor.shape[0]
        print(f"test Average Loss = {avg_loss}")
        rate = 0
        h_rate, l_rate = 0, 0
        list_true = np.array(list_true)
        list_1 = np.array(list_1)
        c = [int(aa == bb) for aa, bb in zip(list_true, list_1)]
        c = np.array(c)
        rate = sum(c) / len(list_true)
        h_rate = sum(c[list_true == 1]) / len(c[list_true == 1])
        l_rate = sum(c[list_true == 0]) / len(c[list_true == 0])
        print('test rate:', rate)
        print('h_rate:', h_rate, )
        print('l_rate:', l_rate)
        print('实际高糖发生次数:', sum(list_true), '预测高糖发生次数:', sum(list_1))

        predict_day_classfication(Test_data=test_input[:, -1, :], pred_y=list_1, true_y=list_true)

if __name__ == '__main__':
    main()
