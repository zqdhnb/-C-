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
import torch.nn.functional as F
pd.set_option('display.max_columns', None)
import warnings
import matplotlib.pyplot as plt
from scipy.special import jn
from IPython.display import display, clear_output
import time
warnings.filterwarnings('ignore')

## 数据处理
from sklearn import preprocessing

## 数据降维处理的
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, SparsePCA

## 模型预测的
import lightgbm as lgb
import xgboost as xgb

## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
# root_path = './dataset/'


# input_path = './new_dataset/014/'
TRAIN_NUM = 16
# input_path = r"D:/p_file/contest/华为杯/input/"

input_path = r'../../input/'
file_names = ['001_new.csv', '002_new.csv', '003_new.csv', '004_new.csv',
              '005_new.csv', '006_new.csv', '007_new.csv', '008_new.csv',
              '009_new.csv', '010_new.csv', '011_new.csv', '012_new.csv',
              '013_new.csv', '014_new.csv', '015_new.csv', '016_new.csv']

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
        # Train_data.iloc[i, -1] = float(Train_data.iloc[i, -1]) / 18  # 标签
    return Train_data
def change_Glucose(data):  # 每次处理的是一个pd.DataFrame()
    for i in range(len(data)):
        data.iloc[i, -1] = float(data.iloc[i, -1]) / 18  # 标签
    return data

def build_model_lgb(x_train, y_train):
    estimator = lgb.LGBMRegressor(num_leaves=127, n_estimators=150)
    param_grid = {
        'learning_rate': [0.1],  #
    }
    gbm = GridSearchCV(estimator, param_grid)

    gbm.fit(x_train, y_train)
    print(f"Best:  {gbm}")

    return gbm

def split_sequences(df, steps = 10):
    sequences = pd.DataFrame()
    #print((df.iloc[0:10, -1].values)/18.0)
    for i in range(0,df.shape[0],1):
        if i+steps > df.shape[0]:
            now_seq = df[-steps:]
            sequences = pd.concat([sequences, now_seq], axis=0, ignore_index=True)
        else:
            now_seq = df[i:i+steps]
            sequences = pd.concat([sequences, now_seq], axis=0, ignore_index=True)
    return sequences
def get_day_list(data):
    # 给定数据，获取不同天的列表 例[[2, 13], [2, 14], [2, 15]]
    # data1为pd.DataFrame()
    day_list = []
    for i in range(data.shape[0]):
        ls_ = [int(data[i, 0].item()), int(data[i, 1].item())]
        #print(ls_)
        if ls_ not in day_list:
            day_list.append(ls_)
    return day_list 
def predict_day_classfication(Test_data, pred_y, true_y):
    # 计算天数差异
    Test_day_list = get_day_list(Test_data)  # 获取了测试集的天数列表
    #print(Test_day_list )
    pred_day_list = [0 for i in range(len(Test_day_list))]  # 预测值的每天高糖次数
    y_day_list = [0 for i in range(len(Test_day_list))]  # 真实值的每天高糖次数
    pred_low = [0 for i in range(len(Test_day_list))]  # 预测值的每天低糖次数
    y_low = [0 for i in range(len(Test_day_list))]  # 真实值的每天低糖次数
    #print(len(pred_y), len(true_y))
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
    print('累计金标差异天数：', sub_day)
       
# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output1, _ = self.lstm(x)
        #output = torch.sigmoid(self.fc(output1[:, -1, :]))  # 取最后一个时间步的输出
        output = self.fc(output1[:, -1, :])
        #print(self.fc(output1[:, -1, :]))
        return output, output1        
    
def InfoNCELoss(anchor, positive, negative, temperature=1.0):
    # Calculate similarity between anchor and positive samples
    sim_pos = []
    for i in range(positive.shape[0]):
        #print(anchor.shape, positive[i].unsqueeze(0).shape)
        sim_pos.append(torch.cosine_similarity(anchor, positive[i].unsqueeze(0), dim=1) / temperature)
    sim_pos = torch.cat(sim_pos)
    #sim_pos = sim_pos / positive.shape[0]
    #print(sim_pos.shape)
    # Calculate similarity between anchor and negative samples
    sim_neg = []
    for i in range(negative.shape[0]):
        sim_neg.append(torch.cosine_similarity(anchor, negative[i].unsqueeze(0), dim=1) / temperature)
    sim_neg = torch.cat(sim_neg)
    #sim_neg = sim_neg / negative.shape[0]
    # Calculate the numerator and denominator of the InfoNCE loss
    numerator = torch.exp(sim_pos) 
    denominator = torch.sum(torch.exp(sim_neg)) 
    #print(numerator.shape)
    
    loss = -torch.log(numerator / (numerator + denominator)).mean()
    #denominator = torch.exp(sim_pos).unsqueeze(1) + torch.exp(sim_neg), dim=1, keepdim=True)

    # Calculate the loss
    #loss = -torch.log(numerator / numerator + denominator).mean()

    return loss

class WeightBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(WeightBCEWithLogitsLoss, self).__init__()
    def forward(self, pred, gt):
        eposion = 1e-10
        count_pos = torch.sum(gt) * 1.0 + eposion
        count_neg = torch.sum(1-gt) * 1.0
        beta = count_neg/count_pos *0.5
        beta_back = count_pos / (count_pos + count_neg)
        #### with weight
        bce1 = nn.BCEWithLogitsLoss(pos_weight = beta)
        loss = beta_back * bce1(pred, gt)
        
        #### without weight
        #bce1 = nn.BCEWithLogitsLoss()
        #loss = bce1(pred, gt)
        return loss
# 定义训练函数
def train_model(model, train_loader, epochs, lr):
    criterion = WeightBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # print(epoch)
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs, output1 = model(inputs)
            #print(outputs)
            b_list_true = [0 if target < 7.5 else 1 for target in targets[:,-1]] #1为高血糖
            loss = criterion(outputs, torch.tensor(b_list_true, dtype=torch.float32).unsqueeze(-1))
            #print(outputs)
            #print(torch.tensor(b_list_true, dtype=torch.float32))
            #exit(1)
            #b_list_true = [0] # no contrastive learning
            #print(output1[:,-1,:].shape)
            """
            if sum(b_list_true) > 2: 
                positive_samples = output1[:,-1,:][np.array(b_list_true) == 1] #(n,32)
                negative_samples = output1[:,-1,:][np.array(b_list_true) == 0]
                #print(positive_samples.shape, negative_samples.shape)
                anchor = torch.mean(positive_samples, dim=0)
                c_loss = InfoNCELoss(anchor.unsqueeze(0), positive_samples, negative_samples , temperature=1.0)
                #print(c_loss)
                loss += c_loss 
            """
            #print(loss)
            loss.backward()
            optimizer.step()  
            

input_size = 12  # 输入特征的维度
hidden_size = 32  # 隐藏状态的维度
output_size = 1  # 输出维度
epochs = 10
lr = 0.01

def main():
    # 随机选择8个文件
    train_files = random.sample(file_names, 8)
    test_files = list(set(file_names) - set(train_files))
    print('训练集', train_files)
    print('测试集', test_files)
    # 创建一个空的DataFrame用于存储训练集数据
    train_data = []
    test_data = []
    # 逐个读取并合并训练集CSV文件
    time_step = 10
    for file_name in train_files:
        file_path = input_path + file_name

        df = pd.read_csv(file_path, sep=',', na_values='NULL', )
        #df = split_sequences(df, steps = 10)
        #Train_data = pd.concat([Train_data, df], axis=0, ignore_index=True)
        data = np.array(df)
        data = data[:,5:]
        data[:,-1] = data[:,-1] / 18.0
        sequences = [data[i:i+time_step] for i in range(0, len(data)-time_step+1)]
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
        train_data.append(sequences_tensor)
        #print(Train_data.shape)
    for file_name in test_files:
        file_path = input_path + file_name

        df = pd.read_csv(file_path, sep=',', na_values='NULL', )
        #df = split_sequences(df, steps = 10)
        #Test_data = pd.concat([Test_data, df], axis=0, ignore_index=True)
        data = np.array(df)
        data = data[:,:]
        data[:,-1] = data[:,-1] / 18.0
        sequences = [data[i:i+time_step] for i in range(0, len(data)-time_step+1)]
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
        test_data.append(sequences_tensor)
        #print(Test_data.shape)
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
    
    
    Train_dataset = TensorDataset(train_input, train_target)
    Test_dataset = TensorDataset(test_input, test_target)
    
    test_loader = DataLoader(Test_dataset, batch_size=1, shuffle=True)  # 测试集打乱
    
  
    # 进行五折交叉验证
    kf = KFold(n_splits=2, shuffle=True, random_state=42)  # 42似乎是一个经验选择   # 训练集打乱
    accuracy_scores = []
    for train_index, val_index in kf.split(Train_dataset):
        # 划分训练数据和交叉验证数据
        #fold_train_x, fold_val_x = train_x[train_index], train_x[val_index]
        #fold_train_y, fold_val_y = train_y[train_index], train_y[val_index]
        train_dataset = torch.utils.data.Subset(Train_dataset, train_index)
        val_dataset = torch.utils.data.Subset(Train_dataset, val_index)
        
        # 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        # 初始化模型并进行训练
        model = LSTMModel(input_size, hidden_size, output_size)
        train_model(model, train_loader, epochs, lr)
        
        # 在验证集上进行评估
        criterion = WeightBCEWithLogitsLoss()
        list_1 = []  # 统计大于7.8的个数
        list_true = []
        with torch.no_grad():
            total_loss = 0.0
            for inputs, targets in val_loader:
                outputs, output1 = model(inputs)
                b_list_true = [0 if target < 7.5 else 1 for target in targets[:,-1]]
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
            c = [int(aa==bb) for aa,bb in zip(list_true,list_1)]
            c = np.array(c)
            rate = sum(c) / len(list_true)
            h_rate = sum(c[list_true==1]) / len(c[list_true==1])
            l_rate = sum(c[list_true==0]) / len(c[list_true==0])
            print('val rate:', rate)
            print('h_rate:', h_rate,)
            print('l_rate:', l_rate)
            print('实际高糖发生次数:', sum(list_true), '预测高糖发生次数:', sum(list_1))
        
    criterion = WeightBCEWithLogitsLoss()
    list_1 = []  # 统计大于7.8的个数
    list_true = []
    with torch.no_grad():
        total_loss = 0.0
        a = []
        for inputs, targets in test_loader:
            #print(get_day_list(inputs[:,:,:2]))
            a.append(inputs[:,:,:5])
            outputs, output1 = model(inputs[:,:,5:])
            b_list_true = [0 if target < 7.5 else 1 for target in targets[:,-1]]
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
        c = [int(aa==bb) for aa,bb in zip(list_true,list_1)]
        c = np.array(c)
        rate = sum(c) / len(list_true)
        h_rate = sum(c[list_true==1]) / len(c[list_true==1])
        l_rate = sum(c[list_true==0]) / len(c[list_true==0])
        print('test rate:', rate)
        print('h_rate:', h_rate,)
        print('l_rate:', l_rate)
        print('实际高糖发生次数:', sum(list_true), '预测高糖发生次数:', sum(list_1))
        
        predict_day_classfication(Test_data=test_input[:,-1,:], pred_y=list_1, true_y=list_true)
        """
            list_1 = []  # 统计大于7.8的个数
            list_true = []
            # 计算准确率
            for i in range(len(fold_val)):
                if fold_val[i] < 7.5:
                    list_true.append(1)
                else:
                    list_true.append(0)
                if val_pred[i] < 7.5:
                    list_1.append(1)
                else:
                    list_1.append(0)
            rate = 0
            for i in range(len(fold_val_y)):
                if list_1[1] == list_true[i]:
                    rate += 1
            rate = rate / len(fold_val_y)
            print('val rate:', rate)
        """
    """
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
    """
    """
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
            if list_1[1] == list_true[i]:
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
        if list_1[1] == list_true[i]:
            rate += 1
    rate = rate / len(test_y)
    print(rate)

    # display(test_y)
    # display(result)
    """

if __name__ == '__main__':
    main()
    #df = pd.read_csv(r"D:/p_file/contest/华为杯/input/001_new.csv", sep=',', na_values='NULL', )
    #split_sequences(df, steps = 10)
    
    '''
    baseline XGBoost*1 8+8 5K
    
    [0]	validation_0-mae:5.43329
    [30]	validation_0-mae:0.59793
    [60]	validation_0-mae:0.50684
    [90]	validation_0-mae:0.47734
    [119]	validation_0-mae:0.45626
    val rate: 0.8106722942417848
    [0]	validation_0-mae:5.39718
    [30]	validation_0-mae:0.58812
    [60]	validation_0-mae:0.51224
    [90]	validation_0-mae:0.47719
    [119]	validation_0-mae:0.45224
    val rate: 0.827253542357552
    [0]	validation_0-mae:5.39908
    [30]	validation_0-mae:0.60492
    [60]	validation_0-mae:0.52713
    [90]	validation_0-mae:0.48734
    [119]	validation_0-mae:0.46008
    val rate: 0.8202653799758746
    [0]	validation_0-mae:5.41224
    [30]	validation_0-mae:0.59828
    [60]	validation_0-mae:0.50863
    [90]	validation_0-mae:0.47085
    [119]	validation_0-mae:0.43997
    val rate: 0.8265983112183354
    [0]	validation_0-mae:5.41803
    [30]	validation_0-mae:0.61216
    [60]	validation_0-mae:0.53277
    [90]	validation_0-mae:0.49393
    [119]	validation_0-mae:0.47320
    val rate: 0.8202653799758746
    train finish!
    test finish!
    0.8608155094299863
    '''
