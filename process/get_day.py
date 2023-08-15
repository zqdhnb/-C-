import pandas as pd

pd.set_option('display.max_columns', None)
import warnings
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

warnings.filterwarnings('ignore')

input_path = '../input/'
file_names = ['001_new.csv', '002_new.csv', '003_new.csv', '004_new.csv',
              '005_new.csv', '006_new.csv', '007_new.csv', '008_new.csv',
              '009_new.csv', '010_new.csv', '011_new.csv', '012_new.csv',
              '013_new.csv', '014_new.csv', '015_new.csv', '016_new.csv']
Glucose_Value = 7.8
Glucose_WARP = 18

def get_day_list(data1):  # 给定数据，获取不同天的列表
    # data1为pd.DataFrame()
    day_list = []
    for i in range(len(data1)):
        ls_ = [data1.iloc[i, 0], data1.iloc[i, 1]]
        # print(ls_)
        if ls_ not in day_list:
            day_list.append(ls_)
    return day_list
def change_Glucose(data):  # 每次处理的是一个pd.DataFrame()
    for i in range(len(data)):
        data.iloc[i, -1] = float(data.iloc[i, -1]) / Glucose_WARP  # 标签
    return data

def main():
    files = list(set(file_names))
    data = pd.DataFrame()
    for file_name in files:
        file_path = input_path + file_name
        df = pd.read_csv(file_path, sep=',', na_values='NULL', )
        data = pd.concat([data, df], axis=0, ignore_index=True)
    # display(data)
    # data = data.iloc[:1000, :]
    # day_list = get_day_list(data)
    # print(day_list)
    Test_data = data
    Test_data = change_Glucose(Test_data)
    test_y = Test_data.iloc[:, -1].values

    # listA = ["o", "u", "i"]
    # # 获取u的索引并打印
    # index_u = listA.index("u")
    # print(index_u)


    Test_day_list = get_day_list(Test_data)  # 获取了测试集的天列表
    pred_day_list = [0 for i in range(len(Test_day_list))]  # 预测值的每天高糖次数
    y_day_list = [0 for i in range(len(Test_day_list))]  # 真实值的每天高糖次数
    for i in range(len(Test_data)):
        ls_ = [Test_data.iloc[i, 0], Test_data.iloc[i, 1]]
        index_u = Test_day_list.index(ls_)  # 获取了该数据对应的天数下标
        # if y_pred[i] > Glucose_Value:
        #     pred_day_list[index_u] += 1
        if test_y[i] > Glucose_Value:
            y_day_list[index_u] += 1
    print(len(y_day_list), y_day_list)
    sub_day = 0
    all_day_pred = 0
    all_day_y = 0
    for i in range(len(Test_day_list)):
        # all_day_pred += pred_day_list[i]
        all_day_y += y_day_list[i]
        # sub_day = abs(pred_day_list[i] - y_day_list[i])
    # sub_day = sub_day / len(Test_day_list)
    print(all_day_y)


if __name__ == '__main__':
    main()
