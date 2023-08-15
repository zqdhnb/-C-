import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data1, data2):
        self.x = torch.tensor(data1, dtype=torch.float32)
        self.y = torch.tensor(data2, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_train_loader(data, batch_size, time_step=128):
    """
    输入:pd.DataFrame()
    输出:train_x:三维tensor(N,C,T) train_y:三维tensor(N,1,T)
    """
    train_x = data.iloc[:, :-1]
    train_y = data.iloc[:, -1]
    train_X = [train_x[i:i + time_step].values.tolist() for i in range(0, len(train_x) - time_step + 1)]
    train_Y = [train_y[i:i + time_step].values.tolist() for i in range(0, len(train_x) - time_step + 1)]
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_Y = torch.tensor(train_Y, dtype=torch.float32)
    train_X = train_X.permute(0, 2, 1)
    train_Y = train_Y.unsqueeze(1)
    print(train_X.shape, train_Y.shape)
    train_data = MyDataset(train_X, train_Y)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    return train_loader


def get_test_loader(data, batch_size=1, time_step=128):
    """
    一个个测试？好像也没有必要
    """
    test_x = data.iloc[:, :-1]
    test_y = data.iloc[:, -1]
    test_X = [test_x[i:i + time_step].values.tolist() for i in range(0, len(test_x) - time_step + 1)]
    test_Y = [test_y[i:i + time_step].values.tolist() for i in range(0, len(test_x) - time_step + 1)]
    test_X = torch.tensor(test_X, dtype=torch.float32)
    test_Y = torch.tensor(test_Y, dtype=torch.float32)
    test_X = test_X.permute(0, 2, 1)
    test_Y = test_Y.unsqueeze(1)
    print(test_X.shape, test_Y.shape)
    test_data = MyDataset(test_X, test_Y)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return test_loader
