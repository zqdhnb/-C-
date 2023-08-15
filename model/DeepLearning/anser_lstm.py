import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.nn import init
from torch.utils.data import Dataset, DataLoader

from model.DeepLearning.augmentation import get_more_data
from model.Tools.tools import get_day_list
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


class MyDataset(Dataset):
    def __init__(self, data):
        self.x = torch.tensor(data[:, :-1], dtype=torch.float32)
        self.y = torch.tensor(data[:, -1], dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# 创建LSTM模型类
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # 修改此处的维度
        out = self.fc(out)
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

        df = pd.read_csv(file_path, sep=',', na_values='NULL')
        df = get_more_data(df, 60, 1.2)
        Train_data = pd.concat([Train_data, df], axis=0, ignore_index=True)
    for file_name in test_files:
        file_path = input_path + file_name

        df = pd.read_csv(file_path, sep=',', na_values='NULL')
        Test_data = pd.concat([Test_data, df], axis=0, ignore_index=True)
    # # 数据标准化
    # Train_data = change_Glucose(Train_data)
    # Test_data = change_Glucose(Test_data)
    #
    # Train_data = std(Train_data)
    # Test_data = std(Test_data)
    # 获取特征和目标变量
    train_data = Train_data.values.reshape(-1, 18)
    test_data = Test_data.values.reshape(-1, 18)

    # 创建K折交叉验证对象
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1

    # 设置超参数
    num_epochs = 5
    batch_size = 32
    learning_rate = 0.001
    # 创建模型实例
    input_size = 17  # 输入数据的特征数
    hidden_size = 256  # 隐藏层的单元数
    num_layers = 2  # LSTM层数
    model = LSTMModel(input_size, hidden_size, num_layers)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for params in model.parameters():
        init.normal_(params, mean=0, std=0.01)


    # 进行交叉验证
    for train_index, val_index in kf.split(train_data):
        # 创建训练集和验证集
        train_dataset = MyDataset(train_data[train_index])
        val_dataset = MyDataset(train_data[val_index])

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 训练模型
        total_step = len(train_loader)
        for epoch in range(num_epochs):
            for i, (inputs, targets) in enumerate(train_loader):
                # 前向传播
                outputs = model(inputs)

                # 计算损失
                loss = criterion(outputs, targets)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss = loss/len(inputs)

                if (i + 1) % 100 == 0:
                    print('Fold [{}/{}], Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(fold, kf.n_splits, epoch+1, num_epochs, i+1, total_step, loss.item()))

        print('Validation on fold [{}/{}]'.format(fold, kf.n_splits))

        # 在验证集上进行预测
        total_loss = 0
        total_count = 0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                total_count += len(inputs)

        average_loss = total_loss / total_count
        print('Validation Loss: {:.4f}'.format(average_loss))

        fold += 1

    print("Cross-validation finish!")

    # 在测试数据集上进行预测
    test_dataset = MyDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_loss = 0
    total_count = 0
    list_1 = []  # 统计大于7.8的个数
    list_true = []

    rate = 0
    targets_len = 0
    y_pred, test_y = [], []
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        total_count += 1
        average_loss = total_loss / total_count
        print('Test Loss: {:.4f}'.format(average_loss))
        for i in range(len(targets)):
            y_pred.append(outputs[i])
            test_y.append(targets[i])
            if targets[i] < 7.8:
                list_true.append(1)
            else:
                list_true.append(0)
            if outputs[i] < 7.8:
                list_1.append(1)
            else:
                list_1.append(0)

    for i in range(len(list_true)):
        if list_1[i] == list_true[i]:
            rate += 1
    rate = rate / len(list_true)
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

main()