import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import numpy as np

from model.DeepLearning.augmentation import get_more_data
from model.DeepLearning.dataloader import get_train_loader, get_test_loader
from model.Tools.tools import std, change_Glucose, get_date, get_float_data, get_day_list, get_gamma, predict_day, \
    change_to_classfication, predict_day_classfication
from model.DeepLearning.Loss import FocalLoss, WeightBCEWithLogitsLoss
input_path = '../../input/'
file_names = ['001_new.csv', '002_new.csv', '003_new.csv', '004_new.csv',
              '005_new.csv', '006_new.csv', '007_new.csv', '008_new.csv',
              '009_new.csv', '010_new.csv', '011_new.csv', '012_new.csv',
              '013_new.csv', '014_new.csv', '015_new.csv', '016_new.csv']

DATA_NUM = 16  # 数据集总的个数
ATTRIBUTE_NUM = 18  # 属性数量(包含标签)
TRAIN_NUM = 8  # 训练集的个数
DATE_NUM = 5  # 时间属性的个数
Glucose_WARP = 18  # 血糖转换值
Glucose_Value = 7.5  # 血糖临界值
K = 5  # 交叉验证折数

batch_size = 64
num_epochs = 3
learning_rate = 0.001

alpha = 2.0  # 高糖数据权重
beta = 1.0  # 低糖数据权重


class Transformer(nn.Module):
    def __init__(self, C_size=17, d_model=128):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, batch_first=True)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(C_size * d_model, d_model)

    def forward(self, x):
        out = self.transformer(x, x)  # 输出为[N, 17, 128]
        out = self.flatten(out)  # 扁平化 两维
        out = self.fc(out)
        out = torch.unsqueeze(out, dim=1)
        return out

def main():
    # 随机选择8个文件
    train_files = random.sample(file_names, TRAIN_NUM)
    test_files = list(set(file_names) - set(train_files))
    print('训练集', train_files)
    print('测试集', test_files)
    # 创建一个空的DataFrame用于存储训练集数据
    Train_data = pd.DataFrame()
    Test_data = pd.DataFrame()


    # 逐个读取并合并训练集CSV文件
    for file_name in train_files:
        file_path = input_path + file_name

        df = pd.read_csv(file_path, sep=',', na_values='NULL', dtype=str)
        df = get_more_data(df, 60, 1.0)  # 输入数据必须是string类型的
        Train_data = pd.concat([Train_data, df], axis=0, ignore_index=True)
    for file_name in test_files:
        file_path = input_path + file_name
        df = pd.read_csv(file_path, sep=',', na_values='NULL', dtype=str)
        df = get_date(df)
        df = get_float_data(df)
        Test_data = pd.concat([Test_data, df], axis=0, ignore_index=True)

    # 去除date
    Train_data = Train_data.iloc[:, 1:]
    Test_data = Test_data.iloc[:, 1:]
    # 血糖值转化
    Train_data, Test_data = change_Glucose(Train_data, Test_data, Glucose_WARP)
    # 数据标准化
    Train_data, Test_data = std(Train_data, Test_data, DATE_NUM, ATTRIBUTE_NUM)

    # model = nn.Transformer(d_model=128, batch_first=True)
    model = Transformer(C_size=17, d_model=16)
    # 定义损失函数和优化器
    # criterion = FocalLoss(gamma=1)
    criterion = WeightBCEWithLogitsLoss(eposion=1e-10, alpha=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 模型参数初始化
    model.train()
    for params in model.parameters():
        init.normal_(params, mean=0, std=0.01)

    # 由回归任务改为分类任务
    Train_data = change_to_classfication(Train_data, Glucose_Value)
    Test_data = change_to_classfication(Test_data, Glucose_Value)
    # 数据
    train_loader = get_train_loader(Train_data, batch_size=batch_size, time_step=16)
    test_loader = get_test_loader(Test_data, batch_size=1, time_step=16)
    # 训练模型
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        pred_y = []
        true_y = []
        for i, (inputs, targets) in enumerate(train_loader):
            # print(inputs.shape, targets.shape)
            # 前向传播
            # outputs = model(inputs, targets)
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            # gamma = get_gamma(outputs, alpha=-0.1, beta=0.001)  # 根据实际高糖数据建立的权重矩阵
            # loss += gamma.sum()
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 计算损失和准确率
            loss = loss / len(inputs)
            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                         loss.item()))
        # 在测试集上进行预测
        test_loss = 0
        total_count = 0
        model.eval()
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_loader):
                # outputs = model(inputs, targets)
                outputs = model(inputs)
                # 计算损失和准确率
                for j in range(len(inputs)):
                    true_y.append(targets[j][0][0].item())
                    pred_y.append(outputs[j][0][0].item())
                if i == len(test_loader) - 1:
                    for j in range(1, len(targets[-1][0])):
                        true_y.append(targets[-1][0][j].item())
                        pred_y.append(outputs[-1][0][j].item())

                loss = criterion(outputs, targets)
                test_loss += loss.item()
                total_count += len(inputs)
        # 输出准确率
        rate = 0

        # if (pred_y[i] <= Glucose_Value and true_y[i] <= Glucose_Value) or (
        #         pred_y[i] > Glucose_Value and true_y[i] > Glucose_Value):
        for i in range(len(true_y)):
            if (pred_y[i] >= 0.5 and true_y[i] >= 0.5) or (pred_y[i] < 0.5 and true_y[i] < 0.5):
                rate += 1
        rate = rate / len(true_y)
        average_loss = test_loss / total_count
        print('Test Loss: {:.6f},Test ACC: {:.6f}'.format(average_loss, rate))
        # 计算天数差异
        predict_day_classfication(Test_data=Test_data, pred_y=pred_y, true_y=true_y)




if __name__ == '__main__':
    main()
