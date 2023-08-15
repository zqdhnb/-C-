import pandas as pd
from IPython import display
list1 = [i for i in range(10000)]
import csv


def function():
    '''
    with open('./output/015/ACC_list.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(map(lambda x: x, list1))
    '''
    f = open('../output/015/IBI_number_list.csv', 'r')
    reader = csv.reader(f)
    final_list = list(reader)
    da = final_list[0]
    print(len(da), da)
    number = 0
    for i in range(len(da)):
        number += int(da[i])
    print(number)
    f.close()
# 102704
def func():
    f = open('../output/015/IBI_list.csv', 'r')
    reader = csv.reader(f)
    final_list = list(reader)
    da = final_list[0]
    print(len(da), da)

#function()
#func()

def process(data1, data2, data3, data4, data5, data6, data7, data8):

    time = data8['Timestamp']
    d1 = data1.iloc[:, 1:]
    d2 = data2.iloc[:, 1:]
    d3 = data3.iloc[:, 1:]
    d4 = data4.iloc[:, 1:]
    d5 = data5.iloc[:, 1:]
    d6 = data6.iloc[:, 1:]
    d7 = data7.iloc[:, 1:]
    d8 = data8.iloc[:, 1:]
    data = pd.concat([time, d1, d2, d3, d4, d5, d6, d7, d8], axis=1)
    #display(data)
    data.to_csv('./output/005/' + '005_new.csv', index=False)
    print("finish!")

# output_path = './output/005/'
# data1 = pd.read_csv(output_path + 'ACC_005_new.csv', sep=',', na_values='NULL', )
# data2 = pd.read_csv(output_path + 'BVP_005_new.csv', sep=',', na_values='NULL', )
# data3 = pd.read_csv(output_path + 'EDA_005_new.csv', sep=',', na_values='NULL', )
# data4 = pd.read_csv(output_path + 'Food_Log_005_new.csv', sep=',', na_values='NULL', )
# data5 = pd.read_csv(output_path + 'HR_005_new.csv', sep=',', na_values='NULL', )
# data6 = pd.read_csv(output_path + 'IBI_005_new.csv', sep=',', na_values='NULL', )
# data7 = pd.read_csv(output_path + 'TEMP_005_new.csv', sep=',', na_values='NULL', )
# data8 = pd.read_csv(output_path + 'Dexcom_005_new.csv', sep=',', na_values='NULL', )
# process(data1, data2, data3, data4, data5, data6, data7, data8)

# data1 = pd.read_csv('./output/005/' + '005_new.csv', sep=',', na_values='NULL', )
# print(data1.isnull().any(axis=0), len(data1))

# list_name = ["ACC", "BVP", "Dexcom", "EDA", "Food_Log", "HR", "IBI", "IBI"]
#
# for name in list_name:
#     path = './output/016/'+str(name)+'_016_new.csv'
#     data1 = pd.read_csv(path, sep=',', na_values='NULL', )
#     print(data1.isnull().any(axis=0), len(data1))
#     print()



for i in range(9):
    path = './input/00'+str(i+1)+'_new.csv'
    print('00'+str(i+1)+'_new.csv')
    data1 = pd.read_csv(path, sep=',', na_values='NULL', )
    print(data1.isnull().any(axis=0), len(data1))
    print()
for i in range(9, 16):
    path = './input/0'+str(i+1)+'_new.csv'
    print('0'+str(i+1)+'_new.csv')
    data1 = pd.read_csv(path, sep=',', na_values='NULL', )
    print(data1.isnull().any(axis=0), len(data1))
    print()


'''
# 构造数据
data = pd.DataFrame({'id': [1, 1, 2, 2], '品牌': ['A', 'B', 'C', 'D']})
# 合并数据
data_new = data.groupby(['id'])['品牌'].apply(list).to_frame()
data_new['品牌'] = data_new['品牌'].apply(lambda x: str(x).replace('[', '').replace(']', ''))
'''

def acc_recall_loss(y_pred, targets):
    list_true, list_1 = [], []
    for i in range(len(targets)):
        if targets[i] < 7.8:
            list_true.append(1)
        else:
            list_true.append(0)
        if y_pred[i] < 7.8:
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
    recall = true_positive / (true_positive + false_negative)

    # 定义损失函数为1 - 准确率 - 召回率
    loss = 1.0 - (accuracy + recall) / 2.0
    torch.tensor(loss, requires_grad=True)
    return loss, accuracy, recall