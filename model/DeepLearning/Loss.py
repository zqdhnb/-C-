import torch
from torch import nn


def acc_recall_loss(y_pred, targets):
    list_true, list_1 = [], []
    for j in range(len(targets)):
        for i in range(len(targets[j])):
            if targets[j][i] < 7.8:
                list_true.append(1)
            else:
                list_true.append(0)
            if y_pred[j][i] < 7.8:
                list_1.append(1)
            else:
                list_1.append(0)
                # 计算准确率
    rate, true_positive, false_negative = 0, 0, 0
    for i in range(len(list_true)):
        if list_1[i] == list_true[i]:
            rate += 1
        if (list_1[i] == 1) & (list_true[i] == 1):
            true_positive += 1
        if (list_1[i] == 0) & (list_true[i] == 1):
            false_negative += 1
    rate = rate / len(list_true)
    accuracy = rate / len(list_true)

    # 计算召回率
    # 假设正样本标签为1，负样本标签为0
    # true_positive = ((list_1 == 1) & (list_true == 1)).sum().float()
    # false_negative = ((list_1 == 0) & (list_true == 1)).sum().float()
    if (true_positive + false_negative) == 0:
        recall = 0
    else:
        recall = true_positive / (true_positive + false_negative)

    # 定义损失函数为1 - 准确率 - 召回率
    loss = 1.0 - (accuracy + recall) / 2.0
    loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
    # loss = (torch.tensor(0.0, requires_grad=True) if loss == 0 else loss)
    return loss, accuracy, recall


class FocalLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        # self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, inputs, target):
        log_p = self.mse(inputs, target)
        p = torch.exp(-log_p)
        loss = (1 - p) ** self.gamma * log_p
        return loss.mean()


class WeightBCEWithLogitsLoss(nn.Module):
    def __init__(self, eposion=1e-10, alpha=1.0):
        super(WeightBCEWithLogitsLoss, self).__init__()
        self.eposion = eposion
        self.alpha = alpha

    def forward(self, pred, gt):
        count_pos = torch.sum(gt) * self.alpha + self.eposion
        count_neg = torch.sum(1 - gt) * self.alpha
        beta = count_neg / count_pos
        beta_back = count_pos / (count_pos + count_neg)
        #### with weight
        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back * bce1(pred, gt)
        #### without weight
        # bce1 = nn.BCEWithLogitsLoss()
        # loss = bce1(pred, gt)
        return loss


# loss = WeightBCEWithLogitsLoss()
# data1 = torch.randn(10, 1, 128)
# data2 = torch.randn(10, 1, 128)
# print(loss(data1, data2))
