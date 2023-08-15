from IPython.display import display, clear_output
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import warnings

warnings.filterwarnings('ignore')
'''处理Dexcom数据'''


def process_Dexcom(data1):
    for i in range(0, 12):
        data1 = data1.drop(index=i)

    data1 = data1.drop(columns='Index')
    # Event Type,Event Subtype,Patient Info,Device Info,Source Device ID
    data1 = data1.drop(columns='Event Type')
    data1 = data1.drop(columns='Event Subtype')
    data1 = data1.drop(columns='Patient Info')
    data1 = data1.drop(columns='Device Info')
    data1 = data1.drop(columns='Source Device ID')

    # Insulin Value (u),Carb Value (grams),Duration (hh:mm:ss),Glucose Rate of Change (mg/dL/min),Transmitter Time (Long Integer)

    data1 = data1.drop(columns='Insulin Value (u)')
    data1 = data1.drop(columns='Carb Value (grams)')
    data1 = data1.drop(columns='Duration (hh:mm:ss)')
    data1 = data1.drop(columns='Glucose Rate of Change (mg/dL/min)')

    # 时间的重复项也删掉

    data1 = data1.drop(columns='Transmitter Time (Long Integer)')
    data1 = data1.rename(
        columns={'Timestamp (YYYY-MM-DDThh:mm:ss)': 'Timestamp', 'Glucose Value (mg/dL)': 'Glucose Value'})
    # 列分割

    df1 = data1['Timestamp'].str.split(' ', expand=True)
    # df1 = df1.rename(columns={'day', 'time'})
    df1.columns = ['day', 'time']
    # print(df1.head(5))

    # 日期划分为年月日  这一步不划分，等到最后模型计算的时候划分
    # ***由于年和月是相同的，干脆合并一下***
    '''
    df11 = df1['day'].str.split('-', expand=True)
    df11.columns = ['year', 'month', 'day']
    df11.astype('int')
    for i, row in df11.iterrows():
        row['day'] = (int(row['year'])-2020) + (int(row['month'])-2) + int(row['day'])
    df11 = df11['day']
    #print(df11.head(5))
    #时间划分为时分秒
    #*秒钟*
    df12 = df1['time'].str.split(':', expand=True)
    df12.columns = ['hour', 'minute', 'second']

    df1 = pd.concat([df11, df12], axis=1)
    #print(df1.head(5))
    df2 = data1['Glucose Value']

    data1 = pd.concat([df1, df2], axis=1)
    data1 = data1.reset_index()
    data1 = data1.drop(columns='index')
    '''
    print(data1.columns)
    print(data1.shape)
    # print(data1.isnull().any(axis=0))
    print(data1.head(20))
    data1.to_csv('./output/015/' + 'Dexcom_015_new.csv', index=False)


# data = pd.read_csv('./data/015/' + 'Dexcom_015.csv', sep=',', na_values='NULL', )
# process_Dexcom(data)


def plot_15(data15):
    # print(data15.shape)
    # display(data15)
    # EDA曲线
    '''
    list15 = []
    m = 0
    n = m
    for i in range(int(len(data15)/100)):
        n += 100
        temp = data15.iloc[m:n, 1].mean()
        list15.append(temp)
    plt.plot([i for i in range(len(list15))], list15, color="blue", label="EDA15")
    plt.show()
    '''
    # Dexcom曲线
    list15 = []
    for i in range(len(data15)):
        list15.append(data15.iloc[i, 1])
    plt.plot([i for i in range(len(list15))], list15, color="blue", label="data")
    plt.show()


def process_Dexcom_time(data1):
    display(data1)
    data = pd.DataFrame(columns=['start_time', 'end_time'], index=[0, 1, 2])
    '''
    2020-07-19 17:33:52--2020-07-20 00:03:53
    2020-07-22 02:53:34--2020-07-22 09:36:45
    2020-07-24 07:17:21--2020-07-25 11:05:05
    '''
    data.iloc[0, 0] = '2020-07-19 17:33:52'
    data.iloc[0, 1] = '2020-07-20 00:03:53'
    data.iloc[1, 0] = '2020-07-22 02:53:34'
    data.iloc[1, 1] = '2020-07-22 09:36:45'
    data.iloc[2, 0] = '2020-07-24 07:17:21'
    data.iloc[2, 1] = '2020-07-25 11:05:05'
    data['start_time'] = pd.to_datetime(data['start_time'])
    data['end_time'] = pd.to_datetime(data['end_time'])
    # print(type(data))  # 转化为了时间数据
    display(data)

    # 取出规定时间段里面的数据
    save_list = []
    data1['Timestamp'] = pd.to_datetime(data1['Timestamp'])
    for i in range(len(data1)):
        if ((data.iloc[0, 0] <= data1.iloc[i, 0] <= data.iloc[0, 1]) or
                (data.iloc[1, 0] <= data1.iloc[i, 0] <= data.iloc[1, 1]) or
                (data.iloc[2, 0] <= data1.iloc[i, 0] <= data.iloc[2, 1])):
            # data1.drop(index=i, inplace=True)
            save_list.append(i)
    display(data1)
    # print(len(save_list), save_list)
    Dexcom_015_new_new = pd.DataFrame(columns=['Timestamp', 'Glucose Value'])
    for i in range(len(save_list)):
        Timestamp = data1.iloc[save_list[i], 0]
        Glucose_Value = data1.iloc[save_list[i], 1]
        Dexcom_015_new_new = Dexcom_015_new_new.append(pd.DataFrame({'Timestamp': [Timestamp],
                                                                     'Glucose Value': [Glucose_Value]}),
                                                       ignore_index=True)
    display(Dexcom_015_new_new)
    plot_15(Dexcom_015_new_new)
    Dexcom_015_new_new.to_csv('./output/015/' + 'Dexcom_015_new_new.csv', index=False)

# data1 = pd.read_csv('./output/015/' + 'Dexcom_015_new.csv', sep=',', na_values='NULL', )
# process_Dexcom_time(data1)


'''
由于数据过大，可以尝试先把数据缩减一些
利如：抽样
'''

# 获取处于规定范围的数据下标
def process_BVP_list(data1):
    display(data1)
    data = pd.DataFrame(columns=['start_time', 'end_time'], index=[0, 1, 2])
    '''
    2020-07-19 17:33:52--2020-07-20 00:03:53
    2020-07-22 02:53:34--2020-07-22 09:36:45
    2020-07-24 07:17:21--2020-07-25 11:05:05
    '''
    data.iloc[0, 0] = '2020-07-19 17:33:52'
    data.iloc[0, 1] = '2020-07-20 00:03:53'
    data.iloc[1, 0] = '2020-07-22 02:53:34'
    data.iloc[1, 1] = '2020-07-22 09:36:45'
    data.iloc[2, 0] = '2020-07-24 07:17:21'
    data.iloc[2, 1] = '2020-07-25 11:05:05'
    data['start_time'] = pd.to_datetime(data['start_time'])
    data['end_time'] = pd.to_datetime(data['end_time'])
    # print(type(data))  # 转化为了时间数据
    display(data)

    # 取出规定时间段里面的数据
    save_list = []
    data1['datetime'] = pd.to_datetime(data1['datetime'])
    data1 = data1.iloc[7000000:, :]  # 把有问题的数据排除
    for i in range(len(data1)):
        if ((data.iloc[0, 0] <= data1.iloc[i, 0] <= data.iloc[0, 1]) or
                (data.iloc[1, 0] <= data1.iloc[i, 0] <= data.iloc[1, 1]) or
                (data.iloc[2, 0] <= data1.iloc[i, 0] <= data.iloc[2, 1])):
            # data1.drop(index=i, inplace=True)
            save_list.append(i)
    display(data1)


    # 干脆将下标写入文件保存 替代原数据的存储
    import csv
    with open('../output/015/BVP_list.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(map(lambda x: x, save_list))

    '''
    BVP_015_new = pd.DataFrame(columns=['datetime', 'bvp'])
    for i in range(len(save_list)):
        datetime = data1.iloc[save_list[i], 0]
        bvp = data1.iloc[save_list[i], 1]
        BVP_015_new = BVP_015_new.append(pd.DataFrame({'datetime': [datetime],
                                                       'bvp': [bvp]}), ignore_index=True)
    display(BVP_015_new)
    BVP_015_new.to_csv('./output/015/' + 'BVP_015_new.csv', index=False)
    '''
# 获得493段数据
def process_BVP_number(data1, data2):
    data1['datetime'] = pd.to_datetime(data1['datetime'])
    data2['Timestamp'] = pd.to_datetime(data2['Timestamp'])
    display(data1, data2)

    # 读取下标列表
    import csv
    f = open('../output/015/BVP_list.csv', 'r')
    csvreader = csv.reader(f)
    final_list = list(csvreader)
    bvp_list = final_list[0]  # 下标列表
    f.close()

    number_list = []  # 统计每一个时间段的个数
    number = 0  # 计数器，初始化为0
    j = 0  # 定位器
    begin_i = 7000000  # 开始位置
    for bvp_i in bvp_list:
        # 对于下标列表里面的每一个样本 我们要将其分块
        # 9449987个样本 分成493块   9445059 + 0
        if data1.iloc[int(bvp_i)+begin_i, 0] <= data2.iloc[j, 0]:
            number += 1
        else:
            print(j)
            number_list.append(number)
            number = 1
            if j < len(data2) - 1:
                j += 1
            else:
                number_list.append(0)
                break
    with open('../output/015/BVP_number_list.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(map(lambda x: x, number_list))


# 读取下标文件，跟Dexcom进行对齐
def process_BVP_time(data1, data2):
    data1['datetime'] = pd.to_datetime(data1['datetime'])
    data2['Timestamp'] = pd.to_datetime(data2['Timestamp'])
    display(data1, data2)

    # 读取下标列表
    import csv
    f = open('../output/015/BVP_list.csv', 'r')
    reader = csv.reader(f)
    final_list = list(reader)
    bvp_list = final_list[0]  # 下标列表
    f.close()

    f = open('../output/015/BVP_number_list.csv', 'r')
    reader2 = csv.reader(f)
    final_list2 = list(reader2)
    number_i = final_list2[0] #段数列表
    number1 = 0
    for i in range(len(number_i)):
        number1 += int(number_i[i])
    number_i[-1] = len(bvp_list) - number1
    print(number_i, len(number_i))
    f.close()

    # 合并
    #  493段合并成492条数据
    begin_i = 7000000  # 起始的位置仍要加入
    data3 = pd.DataFrame(columns=['date', 'bvp'])
    # bvp_list 全部文件的下标列表 共计9449987个样本
    # number_i 段数列表，每段多少个数据，共计493段
    num = 0
    m_i = 0  # bvp_list开始的下标
    m = int(bvp_list[0]) + begin_i  # data1开始的下标
    n_i = m_i + int(number_i[0])  # bvp_list第二个下标
    # n_i = m_i
    # n = int(bvp_list[n_i]) + begin_i   # 下一位的下标

    for i in range(len(number_i) - 1):  # 492条数据
        n_i = n_i + int(number_i[i+1])-1  # 为了利用到493段数据，这里每两段取一下均值
        n = int(bvp_list[n_i]) + begin_i
        num += 1
        print('m_i:', m_i, 'n_i:', n_i, 'm:', m, 'n:', n, num, number_i[i+1])

        bvp = data1.iloc[m:n, 1].mean()
        data3 = data3.append(pd.DataFrame({'date': [data2.iloc[i, 0]],
                                           'bvp': [bvp]}), ignore_index=True)
        m_i = m_i + int(number_i[i])
        m = int(bvp_list[m_i]) + begin_i

    print(data3.shape)
    display(data3)
    data3.to_csv('./output/015/' + 'BVP_015_new.csv', index=False)

# data1 = pd.read_csv('./data/015/' + 'BVP_015.csv', sep=',', na_values='NULL', )
# data2 = pd.read_csv('./output/015/' + 'Dexcom_015_new_new.csv', sep=',', na_values='NULL', )
# process_BVP_list(data1)
# process_BVP_number(data1, data2)
# process_BVP_time(data1, data2)


def process_ACC_list(data1):
    display(data1)
    data = pd.DataFrame(columns=['start_time', 'end_time'], index=[0, 1, 2])
    '''
    2020-07-19 17:33:52--2020-07-20 00:03:53
    2020-07-22 02:53:34--2020-07-22 09:36:45
    2020-07-24 07:17:21--2020-07-25 11:05:05
    '''
    data.iloc[0, 0] = '2020-07-19 17:33:52'
    data.iloc[0, 1] = '2020-07-20 00:03:53'
    data.iloc[1, 0] = '2020-07-22 02:53:34'
    data.iloc[1, 1] = '2020-07-22 09:36:45'
    data.iloc[2, 0] = '2020-07-24 07:17:21'
    data.iloc[2, 1] = '2020-07-25 11:05:05'
    data['start_time'] = pd.to_datetime(data['start_time'])
    data['end_time'] = pd.to_datetime(data['end_time'])
    # print(type(data))  # 转化为了时间数据
    display(data)

    # 取出规定时间段里面的数据
    save_list = []
    data1['datetime'] = pd.to_datetime(data1['datetime'])
    data1 = data1.iloc[3600000:, :]  # 把有问题的数据排除

    # 注意：下标是从3600000开始的
    for i in range(len(data1)):
        if ((data.iloc[0, 0] <= data1.iloc[i, 0] <= data.iloc[0, 1]) or
                (data.iloc[1, 0] <= data1.iloc[i, 0] <= data.iloc[1, 1]) or
                (data.iloc[2, 0] <= data1.iloc[i, 0] <= data.iloc[2, 1])):
            # data1.drop(index=i, inplace=True)
            save_list.append(i)
    display(data1)
    # print(len(save_list), save_list)

    # 干脆将下标写入文件保存 替代原数据的存储
    import csv
    with open('../output/015/ACC_list.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(map(lambda x: x, save_list))
        #writer = csv.writer(csv_file)
        #for row in save_list:
         #   writer.writerow(row)

# data1 = pd.read_csv('./data/015/' + 'ACC_015.csv', sep=',', na_values='NULL', )
# process_ACC_list(data1)

def process_ACC_number(data1, data2):
    data1['datetime'] = pd.to_datetime(data1['datetime'])
    data2['Timestamp'] = pd.to_datetime(data2['Timestamp'])
    display(data1, data2)

    # 读取下标列表
    import csv
    f = open('../output/015/ACC_list.csv', 'r')
    reader = csv.reader(f)
    final_list = list(reader)
    acc_list = final_list[0]  # 下标列表
    f.close()

    number_list = []  # 统计每一个时间段的个数
    number = 0  # 计数器，初始化为0
    j = 0  # 定位器
    begin_i = 3600000  # 开始位置


    for acc_i in acc_list:
        # 对于下标列表里面的每一个样本 我们要将其分块
        # 4724995个数据 分成493块   9445059 + 0
        if data1.iloc[int(acc_i) + begin_i, 0] <= data2.iloc[j, 0]:
            number += 1
        else:
            print(j)
            number_list.append(number)
            number = 1
            if j < len(data2) - 1:
                j += 1
            else:
                number_list.append(0)
                break
    with open('../output/015/ACC_number_list.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(map(lambda x: x, number_list))

# data1 = pd.read_csv('./data/015/' + 'ACC_015.csv', sep=',', na_values='NULL', )
# data2 = pd.read_csv('./output/015/' + 'Dexcom_015_new_new.csv', sep=',', na_values='NULL', )
# process_ACC_number(data1, data2)


# 读取下标文件，跟Dexcom进行对齐
def process_ACC_time(data1, data2):
    data1['datetime'] = pd.to_datetime(data1['datetime'])
    data2['Timestamp'] = pd.to_datetime(data2['Timestamp'])
    display(data1, data2)

    # 读取下标列表
    import csv
    f = open('../output/015/ACC_list.csv', 'r')
    reader = csv.reader(f)
    final_list = list(reader)
    acc_list = final_list[0]  # 下标列表
    f.close()

    f = open('../output/015/ACC_number_list.csv', 'r')
    reader2 = csv.reader(f)
    final_list2 = list(reader2)
    number_i = final_list2[0]  # 段数列表
    number1 = 0
    for i in range(len(number_i)):
        number1 += int(number_i[i])
    number_i[-1] = len(acc_list) - number1
    print(number_i, len(number_i))
    f.close()

    # 合并
    #  493段合并成492条数据
    begin_i = 3600000  # 起始的位置仍要加入
    data3 = pd.DataFrame(columns=['date', 'acc_x', 'acc_y', 'acc_z'])
    # bvp_list 全部文件的下标列表 共计9449987个样本
    # number_i 段数列表，每段多少个数据，共计493段
    num = 0
    m_i = 0  # bvp_list开始的下标
    m = int(acc_list[0]) + begin_i  # data1开始的下标
    n_i = m_i + int(number_i[0])  # bvp_list第二个下标
    # n_i = m_i
    # n = int(bvp_list[n_i]) + begin_i   # 下一位的下标

    for i in range(len(number_i) - 1):  # 492条数据
        n_i = n_i + int(number_i[i + 1]) - 1  # 为了利用到493段数据，这里每两段取一下均值
        n = int(acc_list[n_i]) + begin_i
        num += 1
        print('m_i:', m_i, 'n_i:', n_i, 'm:', m, 'n:', n, num, number_i[i + 1])

        acc_x = data1.iloc[m:n, 1].mean()
        acc_y = data1.iloc[m:n, 2].mean()
        acc_z = data1.iloc[m:n, 3].mean()
        data3 = data3.append(pd.DataFrame({'date': [data2.iloc[i, 0]],
                                           'acc_x': [acc_x],
                                           'acc_y': [acc_y],
                                           'acc_z': [acc_z]}), ignore_index=True)
        m_i = m_i + int(number_i[i])
        m = int(acc_list[m_i]) + begin_i

    print(data3.shape)
    display(data3)
    data3.to_csv('./output/015/' + 'ACC_015_new.csv', index=False)


# data1 = pd.read_csv('./data/015/' + 'ACC_015.csv', sep=',', na_values='NULL', )
# data2 = pd.read_csv('./output/015/' + 'Dexcom_015_new_new.csv', sep=',', na_values='NULL', )
# process_ACC_time(data1, data2)


def porcess_EDA(data1, data2):
    import csv
    display(data1)
    data = pd.DataFrame(columns=['start_time', 'end_time'], index=[0, 1, 2])
    '''
    2020-07-19 17:33:52--2020-07-20 00:03:53
    2020-07-22 02:53:34--2020-07-22 09:36:45
    2020-07-24 07:17:21--2020-07-25 11:05:05
    '''
    data.iloc[0, 0] = '2020-07-19 17:33:52'
    data.iloc[0, 1] = '2020-07-20 00:03:53'
    data.iloc[1, 0] = '2020-07-22 02:53:34'
    data.iloc[1, 1] = '2020-07-22 09:36:45'
    data.iloc[2, 0] = '2020-07-24 07:17:21'
    data.iloc[2, 1] = '2020-07-25 11:05:05'
    data['start_time'] = pd.to_datetime(data['start_time'])
    data['end_time'] = pd.to_datetime(data['end_time'])
    # print(type(data))  # 转化为了时间数据
    display(data)

    # 取出规定时间段里面的数据
    save_list = []
    data1['datetime'] = pd.to_datetime(data1['datetime'])
    data2['Timestamp'] = pd.to_datetime(data2['Timestamp'])
    data_1 = data1.iloc[700000:, :]  # 把有问题的数据排除
    for i in range(len(data_1)):  # 注意下标是从700000开始的
        if ((data.iloc[0, 0] <= data_1.iloc[i, 0] <= data.iloc[0, 1]) or
                (data.iloc[1, 0] <= data_1.iloc[i, 0] <= data.iloc[1, 1]) or
                (data.iloc[2, 0] <= data_1.iloc[i, 0] <= data.iloc[2, 1])):
            # data1.drop(index=i, inplace=True)
            save_list.append(i)
    display(data_1)
    import os
    if os.path.exists('../output/015/EDA_list.csv'):
            print('文件已存在！')
            with open('../output/015/EDA_list.csv', 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(map(lambda x: x, save_list))
    else:
        with open('../output/015/EDA_list.csv', 'w', newline='') as csv_file:
            print("文件创建成功！")
            writer = csv.writer(csv_file)
            writer.writerow(map(lambda x: x, save_list))
    eda_list = save_list
    print('acc_list:', eda_list)

    number_list = []  # 统计每一个时间段的个数
    number = 0  # 计数器，初始化为0
    j = 0  # 定位器
    begin_i = 700000  # 开始位置
    for eda_i in eda_list:
        # 对于下标列表里面的每一个样本 我们要将其分块
        if data1.iloc[int(eda_i) + begin_i, 0] <= data2.iloc[j, 0]:
            number += 1
        else:
            print(j)
            number_list.append(number)
            number = 1
            if j < len(data2) - 1:
                j += 1
            else:
                number_list.append(0)
                break
    if os.path.exists('../output/015/EDA_number_list.csv'):
            print('文件已存在！')
            with open('../output/015/EDA_number_list.csv', 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(map(lambda x: x, number_list))
    else:
        import csv
        with open('../output/015/EDA_number_list.csv', 'w', newline='') as csv_file:
            print("文件创建成功！")
            writer = csv.writer(csv_file)
            writer.writerow(map(lambda x: x, number_list))
    number_i = number_list
    print(eda_list)
    print(number_i)

    # 计算492数据并存储

    data3 = pd.DataFrame(columns=['date', 'eda'])
    # bvp_list 全部文件的下标列表 共计9449987个样本
    # number_i 段数列表，每段多少个数据，共计493段
    num = 0
    m_i = 0  # bvp_list开始的下标
    m = int(eda_list[0]) + begin_i  # data1开始的下标
    n_i = m_i + int(number_i[0])  # bvp_list第二个下标
    # n_i = m_i
    # n = int(bvp_list[n_i]) + begin_i   # 下一位的下标

    for i in range(len(number_i) - 1):  # 492条数据
        n_i = n_i + int(number_i[i + 1]) - 1  # 为了利用到493段数据，这里每两段取一下均值
        n = int(eda_list[n_i]) + begin_i
        num += 1
        print('m_i:', m_i, 'n_i:', n_i, 'm:', m, 'n:', n, num, number_i[i + 1])

        eda = data1.iloc[m:n, 1].mean()
        data3 = data3.append(pd.DataFrame({'date': [data2.iloc[i, 0]],
                                           'eda': [eda]}), ignore_index=True)
        m_i = m_i + int(number_i[i])
        m = int(eda_list[m_i]) + begin_i

    print(data3.shape)
    display(data3)
    data3.to_csv('./output/015/' + 'EDA_015_new.csv', index=False)

def porcess_HR(data1, data2):
    import csv
    display(data1)
    data = pd.DataFrame(columns=['start_time', 'end_time'], index=[0, 1, 2])
    '''
    2020-07-19 17:33:52--2020-07-20 00:03:53
    2020-07-22 02:53:34--2020-07-22 09:36:45
    2020-07-24 07:17:21--2020-07-25 11:05:05
    '''
    data.iloc[0, 0] = '2020-07-19 17:33:52'
    data.iloc[0, 1] = '2020-07-20 00:03:53'
    data.iloc[1, 0] = '2020-07-22 02:53:34'
    data.iloc[1, 1] = '2020-07-22 09:36:45'
    data.iloc[2, 0] = '2020-07-24 07:17:21'
    data.iloc[2, 1] = '2020-07-25 11:05:05'
    data['start_time'] = pd.to_datetime(data['start_time'])
    data['end_time'] = pd.to_datetime(data['end_time'])
    # print(type(data))  # 转化为了时间数据
    display(data)

    # 取出规定时间段里面的数据
    save_list = []
    data1['datetime'] = pd.to_datetime(data1['datetime'])
    data2['Timestamp'] = pd.to_datetime(data2['Timestamp'])
    data_1 = data1.iloc[240000:, :]  # 把有问题的数据排除
    for i in range(len(data_1)):  # 注意下标是从700000开始的
        if ((data.iloc[0, 0] <= data_1.iloc[i, 0] <= data.iloc[0, 1]) or
                (data.iloc[1, 0] <= data_1.iloc[i, 0] <= data.iloc[1, 1]) or
                (data.iloc[2, 0] <= data_1.iloc[i, 0] <= data.iloc[2, 1])):
            # data1.drop(index=i, inplace=True)
            save_list.append(i)
    display(data_1)
    import os
    if os.path.exists('../output/015/HR_list.csv'):
            print('文件已存在！')
            with open('../output/015/HR_list.csv', 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(map(lambda x: x, save_list))
    else:
        with open('../output/015/HR_list.csv', 'w', newline='') as csv_file:
            print("文件创建成功！")
            writer = csv.writer(csv_file)
            writer.writerow(map(lambda x: x, save_list))
    hr_list = save_list
    print('acc_list:', hr_list)

    number_list = []  # 统计每一个时间段的个数
    number = 0  # 计数器，初始化为0
    j = 0  # 定位器
    begin_i = 240000  # 开始位置
    for eda_i in hr_list:
        # 对于下标列表里面的每一个样本 我们要将其分块
        if data1.iloc[int(eda_i) + begin_i, 0] <= data2.iloc[j, 0]:
            number += 1
        else:
            print(j)
            number_list.append(number)
            number = 1
            if j < len(data2) - 1:
                j += 1
            else:
                number_list.append(0)
                break
    if os.path.exists('../output/015/HR_number_list.csv'):
            print('文件已存在！')
            with open('../output/015/HR_number_list.csv', 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(map(lambda x: x, number_list))
    else:
        import csv
        with open('../output/015/HR_number_list.csv', 'w', newline='') as csv_file:
            print("文件创建成功！")
            writer = csv.writer(csv_file)
            writer.writerow(map(lambda x: x, number_list))
    number_i = number_list
    print(hr_list)
    print(number_i)

    # 计算492数据并存储

    data3 = pd.DataFrame(columns=['date', 'hr'])
    # bvp_list 全部文件的下标列表 共计9449987个样本
    # number_i 段数列表，每段多少个数据，共计493段
    num = 0
    m_i = 0  # bvp_list开始的下标
    m = int(hr_list[0]) + begin_i  # data1开始的下标
    n_i = m_i + int(number_i[0])  # bvp_list第二个下标
    # n_i = m_i
    # n = int(bvp_list[n_i]) + begin_i   # 下一位的下标

    for i in range(len(number_i) - 1):  # 492条数据
        n_i = n_i + int(number_i[i + 1]) - 1  # 为了利用到493段数据，这里每两段取一下均值
        n = int(hr_list[n_i]) + begin_i
        num += 1
        print('m_i:', m_i, 'n_i:', n_i, 'm:', m, 'n:', n, num, number_i[i + 1])

        hr = data1.iloc[m:n, 1].mean()
        data3 = data3.append(pd.DataFrame({'date': [data2.iloc[i, 0]],
                                           'hr': [hr]}), ignore_index=True)
        m_i = m_i + int(number_i[i])
        m = int(hr_list[m_i]) + begin_i

    print(data3.shape)
    display(data3)
    data3.to_csv('./output/015/' + 'HR_015_new.csv', index=False)

# data1 = pd.read_csv('./data/015/' + 'HR_015.csv', sep=',', na_values='NULL', )
# data2 = pd.read_csv('./output/015/' + 'Dexcom_015_new_new.csv', sep=',', na_values='NULL', )
# porcess_HR(data1, data2)  # 最后一条NAN？？


#IBI分开处理一下，看问题在哪
def porcess_IBI_in(data1, data2):
    import csv
    display(data1)
    data = pd.DataFrame(columns=['start_time', 'end_time'], index=[0, 1, 2])
    '''
    2020-07-19 17:33:52--2020-07-20 00:03:53
    2020-07-22 02:53:34--2020-07-22 09:36:45
    2020-07-24 07:17:21--2020-07-25 11:05:05
    '''
    data.iloc[0, 0] = '2020-07-19 17:33:52'
    data.iloc[0, 1] = '2020-07-20 00:03:53'
    data.iloc[1, 0] = '2020-07-22 02:53:34'
    data.iloc[1, 1] = '2020-07-22 09:36:45'
    data.iloc[2, 0] = '2020-07-24 07:17:21'
    data.iloc[2, 1] = '2020-07-25 11:05:05'
    data['start_time'] = pd.to_datetime(data['start_time'])
    data['end_time'] = pd.to_datetime(data['end_time'])
    # print(type(data))  # 转化为了时间数据
    display(data)

    # 取出规定时间段里面的数据
    save_list = []
    data1['datetime'] = pd.to_datetime(data1['datetime'])
    data2['Timestamp'] = pd.to_datetime(data2['Timestamp'])
    data_1 = data1.iloc[100000:, :]  # 把有问题的数据排除
    for i in range(len(data_1)):  # 注意下标是从700000开始的
        if ((data.iloc[0, 0] <= data_1.iloc[i, 0] <= data.iloc[0, 1]) or
                (data.iloc[1, 0] <= data_1.iloc[i, 0] <= data.iloc[1, 1]) or
                (data.iloc[2, 0] <= data_1.iloc[i, 0] <= data.iloc[2, 1])):
            # data1.drop(index=i, inplace=True)
            save_list.append(i)
    display(data_1)
    import os
    if os.path.exists('../output/015/IBI_list.csv'):
            print('文件已存在！')
            #with open('./output/015/IBI_list.csv', 'w', newline='') as csv_file:
             #   writer = csv.writer(csv_file)
              #  writer.writerow(map(lambda x: x, save_list))
    else:
        with open('../output/015/IBI_list.csv', 'w', newline='') as csv_file:
            print("文件创建成功！")
            writer = csv.writer(csv_file)
            writer.writerow(map(lambda x: x, save_list))
    ibi_list = save_list
    print('ibi_list:', ibi_list)

    # ibi_list没有问题

    number_list = []  # 统计每一个时间段的个数
    number = 0  # 计数器，初始化为0
    j = 0  # 定位器
    begin_i = 100000  # 开始位置
    for ibi_i in ibi_list:
        # 对于下标列表里面的每一个样本 我们要将其分块
        if data1.iloc[int(ibi_i) + begin_i, 0] <= data2.iloc[j, 0]:
            number += 1
        else:
            print(j)
            number_list.append(number)
            number = 1
            if j < len(data2) - 1:
                j += 1
            else:
                number_list.append(0)
                break
    if os.path.exists('../output/015/IBI_number_list.csv'):
            print('文件已存在！')
            #with open('./output/015/HR_number_list.csv', 'w', newline='') as csv_file:
                #writer = csv.writer(csv_file)
                #writer.writerow(map(lambda x: x, number_list))
    else:
        import csv
        with open('../output/015/IBI_number_list.csv', 'w', newline='') as csv_file:
            print("文件创建成功！")
            writer = csv.writer(csv_file)
            writer.writerow(map(lambda x: x, number_list))



def porcess_IBI_on(data1, data2):
    data1['datetime'] = pd.to_datetime(data1['datetime'])
    data2['Timestamp'] = pd.to_datetime(data2['Timestamp'])
    display(data1, data2)
    # 读取下标列表
    import csv
    f = open('../output/015/IBI_list.csv', 'r')
    reader = csv.reader(f)
    final_list = list(reader)
    ibi_list = final_list[0]  # 下标列表
    f.close()

    f = open('../output/015/IBI_number_list.csv', 'r')
    reader2 = csv.reader(f)
    final_list2 = list(reader2)
    number_i = final_list2[0]  # 段数列表
    number1 = 0
    for i in range(len(number_i)):
        number1 += int(number_i[i])
    number_i.append(len(ibi_list) - number1)
    print(len(number_i), number_i)
    f.close()

    #  492段合并成492条数据
    begin_i = 100000  # 起始的位置仍要加入
    data3 = pd.DataFrame(columns=['date', 'ibi'])

    num = 0
    m_i = 0  # bvp_list开始的下标
    m = int(ibi_list[0]) + begin_i  # data1开始的下标
    n_i = m_i  # bvp_list第二个下标
    # n_i = m_i
    # n = int(bvp_list[n_i]) + begin_i   # 下一位的下标

    for i in range(len(number_i)):  # 492条数据
        n_i = n_i + int(number_i[i])-1   # ***这里要-1 下面也要减1
        n = int(ibi_list[n_i]) + begin_i
        print('m_i:', m_i, 'n_i:', n_i, 'm:', m, 'n:', n, i, number_i[i])

        ibi = data1.iloc[m:n, 1].mean()
        data3 = data3.append(pd.DataFrame({'date': [data2.iloc[i, 0]],
                                           'ibi': [ibi]}), ignore_index=True)
        if i != (len(number_i)-1):
            m_i = m_i + int(number_i[i])-1
            m = int(ibi_list[m_i]) + begin_i

    print(data3.shape)
    display(data3)
    data3.to_csv('./output/015/' + 'IBI_015_new.csv', index=False)



#data1 = pd.read_csv('./data/015/' + 'IBI_015.csv', sep=',', na_values='NULL', )
#data2 = pd.read_csv('./output/015/' + 'Dexcom_015_new_new.csv', sep=',', na_values='NULL', )
#porcess_IBI_on(data1, data2)




def porcess_TEMP(data1, data2):
    import csv
    display(data1)
    data = pd.DataFrame(columns=['start_time', 'end_time'], index=[0, 1, 2])
    '''
    2020-07-19 17:33:52--2020-07-20 00:03:53
    2020-07-22 02:53:34--2020-07-22 09:36:45
    2020-07-24 07:17:21--2020-07-25 11:05:05
    '''
    data.iloc[0, 0] = '2020-07-19 17:33:52'
    data.iloc[0, 1] = '2020-07-20 00:03:53'
    data.iloc[1, 0] = '2020-07-22 02:53:34'
    data.iloc[1, 1] = '2020-07-22 09:36:45'
    data.iloc[2, 0] = '2020-07-24 07:17:21'
    data.iloc[2, 1] = '2020-07-25 11:05:05'
    data['start_time'] = pd.to_datetime(data['start_time'])
    data['end_time'] = pd.to_datetime(data['end_time'])
    # print(type(data))  # 转化为了时间数据
    display(data)

    # 取出规定时间段里面的数据
    save_list = []
    data1['datetime'] = pd.to_datetime(data1['datetime'])
    data2['Timestamp'] = pd.to_datetime(data2['Timestamp'])
    data_1 = data1.iloc[700000:, :]  # 把有问题的数据排除
    for i in range(len(data_1)):  # 注意下标是从700000开始的
        if ((data.iloc[0, 0] <= data_1.iloc[i, 0] <= data.iloc[0, 1]) or
                (data.iloc[1, 0] <= data_1.iloc[i, 0] <= data.iloc[1, 1]) or
                (data.iloc[2, 0] <= data_1.iloc[i, 0] <= data.iloc[2, 1])):
            # data1.drop(index=i, inplace=True)
            save_list.append(i)
    display(data_1)
    import os
    if os.path.exists('../output/015/TEMP_list.csv'):
            print('文件已存在！')
            #with open('./output/015/TEMP_list.csv', 'w', newline='') as csv_file:
                #writer = csv.writer(csv_file)
                #writer.writerow(map(lambda x: x, save_list))
    else:
        with open('../output/015/TEMP_list.csv', 'w', newline='') as csv_file:
            print("文件创建成功！")
            writer = csv.writer(csv_file)
            writer.writerow(map(lambda x: x, save_list))
    hr_list = save_list
    print('acc_list:', hr_list)

    number_list = []  # 统计每一个时间段的个数
    number = 0  # 计数器，初始化为0
    j = 0  # 定位器
    begin_i = 700000  # 开始位置
    for eda_i in hr_list:
        # 对于下标列表里面的每一个样本 我们要将其分块
        if data1.iloc[int(eda_i) + begin_i, 0] <= data2.iloc[j, 0]:
            number += 1
        else:
            print(j)
            number_list.append(number)
            number = 1
            if j < len(data2) - 1:
                j += 1
            else:
                number_list.append(0)
                break
    if os.path.exists('../output/015/TEMP_number_list.csv'):
            print('文件已存在！')
            #with open('./output/015/TEMP_number_list.csv', 'w', newline='') as csv_file:
                #writer = csv.writer(csv_file)
                #writer.writerow(map(lambda x: x, number_list))
    else:
        import csv
        with open('../output/015/TEMP_number_list.csv', 'w', newline='') as csv_file:
            print("文件创建成功！")
            writer = csv.writer(csv_file)
            writer.writerow(map(lambda x: x, number_list))
    number_i = number_list
    print(hr_list)
    print(number_i)

    # 计算492数据并存储

    data3 = pd.DataFrame(columns=['date', 'temp'])
    # bvp_list 全部文件的下标列表 共计9449987个样本
    # number_i 段数列表，每段多少个数据，共计493段
    num = 0
    m_i = 0  # bvp_list开始的下标
    m = int(hr_list[0]) + begin_i  # data1开始的下标
    n_i = m_i + int(number_i[0])  # bvp_list第二个下标
    # n_i = m_i
    # n = int(bvp_list[n_i]) + begin_i   # 下一位的下标

    for i in range(len(number_i) - 1):  # 492条数据
        n_i = n_i + int(number_i[i + 1]) - 1  # 为了利用到493段数据，这里每两段取一下均值
        n = int(hr_list[n_i]) + begin_i
        num += 1
        print('m_i:', m_i, 'n_i:', n_i, 'm:', m, 'n:', n, num, number_i[i + 1])

        hr = data1.iloc[m:n, 1].mean()
        data3 = data3.append(pd.DataFrame({'date': [data2.iloc[i, 0]],
                                           'temp': [hr]}), ignore_index=True)
        m_i = m_i + int(number_i[i])
        m = int(hr_list[m_i]) + begin_i

    print(data3.shape)
    display(data3)
    data3.to_csv('./output/015/' + 'TEMP_015_new.csv', index=False)

#data1 = pd.read_csv('./data/015/' + 'TEMP_015.csv', sep=',', na_values='NULL', )
#data2 = pd.read_csv('./output/015/' + 'Dexcom_015_new_new.csv', sep=',', na_values='NULL', )
#porcess_TEMP(data1, data2)


def porcess_FOODLOG_1(data1):
    display(data1)


    data1.drop(columns='time_end', inplace=True)
    data1.drop(columns='unit', inplace=True)
    data1.drop(columns='dietary_fiber', inplace=True)
    data1.drop(columns='total_fat', inplace=True)
    # 两个时间时完全一样的，删掉一个
    data1.drop(columns='time_begin', inplace=True)

    data1.drop(columns='searched_food', inplace=True)

    # log_food不好处理，干脆也删掉
    data1.drop(columns='logged_food', inplace=True)

    # date和time合并
    for i, row in data1.iterrows():
        data1.iloc[i, 0] = str(row['date']) + ' ' + str(row['time'])
        # row['date'] = (row['date']) + ' ' + row['time']
        # print(row['date'])

    data1.drop(columns='time', inplace=True)
    data1.drop(columns='amount', inplace=True)
    display(data1)
    print(data1.isnull().any(axis=0))

    data1.to_csv('./output/015/' + 'Food_Log_015_new.csv', index=False)

def process_FOODLOG_2(data1):
    data1['date'] = pd.to_datetime(data1['date'])
    display(data1)

    data = pd.DataFrame(columns=['start_time', 'end_time'], index=[0, 1, 2])
    '''
    2020-07-19 17:33:52--2020-07-20 00:03:53
    2020-07-22 02:53:34--2020-07-22 09:36:45
    2020-07-24 07:17:21--2020-07-25 11:05:05
    '''
    data.iloc[0, 0] = '2020-07-19 17:33:52'
    data.iloc[0, 1] = '2020-07-20 00:03:53'
    data.iloc[1, 0] = '2020-07-22 02:53:34'
    data.iloc[1, 1] = '2020-07-22 09:36:45'
    data.iloc[2, 0] = '2020-07-24 07:17:21'
    data.iloc[2, 1] = '2020-07-25 11:05:05'
    data['start_time'] = pd.to_datetime(data['start_time'])
    data['end_time'] = pd.to_datetime(data['end_time'])
    # print(type(data))  # 转化为了时间数据
    display(data)

    data3 = pd.DataFrame(columns=['date', 'calorie', 'total_carb', 'sugar', 'protein'])
    for i in range(len(data1)):
        if ((data.iloc[0, 0] <= data1.iloc[i, 0] <= data.iloc[0, 1]) or
                (data.iloc[1, 0] <= data1.iloc[i, 0] <= data.iloc[1, 1]) or
                (data.iloc[2, 0] <= data1.iloc[i, 0] <= data.iloc[2, 1])):
            data3 = data3.append(pd.DataFrame({'date': [data1.iloc[i, 0]],
                                               'calorie': [data1.iloc[i, 1]],
                                               'total_carb': [data1.iloc[i, 2]],
                                               'sugar': [data1.iloc[i, 3]],
                                               'protein': [data1.iloc[i, 4]]}), ignore_index=True)

    display(data3)
    data3.to_csv('./output/015/' + 'Food_Log_015_new_new.csv', index=False)


def porcess_Food_Log(data1, data2):
    data1['date'] = pd.to_datetime(data1['date'])  # 8
    data2['Timestamp'] = pd.to_datetime(data2['Timestamp'])  # 492

    display(data1, data2)
    number_list = []  # 统计每一个时间段的个数
    number = 0  # 计数器，初始化为0
    sum = 0
    j = 0  # j 定位 data2['date']
    '''
    遍历整个大的时间段，计算小于定位值的样本数量，作为需要填充的区间样本数量
    注意：只适用于分割时间段被包含在被分割时间段的数据
    '''
    for i in range(len(data2)):  # i 定位 data1['Timestamp']
        if data2.iloc[i, 0] < data1.iloc[j, 0]:  # 会分成十段
            number += 1
        else:
            sum += number
            number_list.append(number)
            number = 1
            if j <= 7:
                j += 1
            else:
                number_list.append(len(data2)-sum)
                break
    print(number_list)
    i = 0
    for number in number_list:
        i += number
    print(i, len(number_list))

    data3 = pd.DataFrame(columns=['date', 'calorie', 'total_carb', 'sugar', 'protein'])
    for i in range(number_list[0]):
        data3 = data3.append(pd.DataFrame({'date': [data1.iloc[0, 0]],
                                           'calorie': [data1.iloc[0, 1]],
                                           'total_carb': [data1.iloc[0, 2]],
                                           'sugar': [data1.iloc[0, 3]],
                                           'protein': [data1.iloc[0, 4]]}), ignore_index=True)

    for i in range(1, len(number_list) - 1):
        for j in range(number_list[i]):
            data3 = data3.append(pd.DataFrame({'date': [data1.iloc[i, 0]],
                                               'calorie': [data1.iloc[i, 1]],
                                               'total_carb': [data1.iloc[i, 2]],
                                               'sugar': [data1.iloc[i, 3]],
                                               'protein': [data1.iloc[i, 4]]}), ignore_index=True)
        print(data3.shape)

    for i in range(number_list[-1]):
        data3 = data3.append(pd.DataFrame({'date': [data1.iloc[-1, 0]],
                                           'calorie': [data1.iloc[-1, 1]],
                                           'total_carb': [data1.iloc[-1, 2]],
                                           'sugar': [data1.iloc[-1, 3]],
                                           'protein': [data1.iloc[-1, 4]]}), ignore_index=True)

    display(data3)
    data3.to_csv('./output/015/' + 'Food_Log_015_new_new_new.csv', index=False)



# data1 = pd.read_csv('./output/015/' + 'Food_Log_015_new_new.csv', sep=',', na_values='NULL', )
# data2 = pd.read_csv('./output/015/' + 'Dexcom_015_new_new.csv', sep=',', na_values='NULL', )
# process_FOODLOG_2(data1)
# porcess_Food_Log(data1, data2)




# 还是一个个处理
def process_015(data1, data2, data3, data4, data5, data6, data7, data8):

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
    display(data)
    data.to_csv('./output/015/' + '015_new.csv', index=False)
    print("finish!")

# output_path = './output/015/'
# data1 = pd.read_csv(output_path + 'ACC_015_new.csv', sep=',', na_values='NULL', )
# data2 = pd.read_csv(output_path + 'BVP_015_new.csv', sep=',', na_values='NULL', )
# data3 = pd.read_csv(output_path + 'EDA_015_new.csv', sep=',', na_values='NULL', )
# data4 = pd.read_csv(output_path + 'Food_Log_015_new_new_new.csv', sep=',', na_values='NULL', )
# data5 = pd.read_csv(output_path + 'HR_015_new.csv', sep=',', na_values='NULL', )
# data6 = pd.read_csv(output_path + 'IBI_015_new.csv', sep=',', na_values='NULL', )
# data7 = pd.read_csv(output_path + 'TEMP_015_new.csv', sep=',', na_values='NULL', )
# data8 = pd.read_csv(output_path + 'Dexcom_015_new_new.csv', sep=',', na_values='NULL', )
# process_015(data1, data2, data3, data4, data5, data6, data7, data8)

def porcess_Foodlog_003(data1):
    display(data1)
    #data1.drop(columns='time_end', inplace=True)
    data1.drop(columns='unit', inplace=True)
    #data1.drop(columns='dietary_fiber', inplace=True)
    #data1.drop(columns='total_fat', inplace=True)
    # 两个时间时完全一样的，删掉一个
    data1.drop(columns='time_begin', inplace=True)

    data1.drop(columns='searched_food', inplace=True)

    # log_food不好处理，干脆也删掉
    data1.drop(columns='logged_food', inplace=True)

    # date和time合并
    for i, row in data1.iterrows():
        data1.iloc[i, 0] = str(row['date']) + ' ' + str(row['time'])
        # row['date'] = (row['date']) + ' ' + row['time']
        # print(row['date'])

    data1.drop(columns='time', inplace=True)
    data1.drop(columns='amount', inplace=True)
    display(data1)
    print(data1.isnull().any(axis=0))
    data1.to_csv('./output/003/' + 'Food_Log_003_new.csv', index=False)

def porcess_Foodlog_003_time(data1, data2):
    data1['date'] = pd.to_datetime(data1['date'])  # 58
    data2['Timestamp'] = pd.to_datetime(data2['Timestamp'])  # 2301
    display(data1, data2)

    number_list = []  # 统计每一个时间段的个数
    number = 0  # 计数器，初始化为0
    sum = 0
    j = 0  # j 定位 data2['date']
    '''
    遍历整个大的时间段，计算小于定位值的样本数量，作为需要填充的区间样本数量
    注意：只适用于分割时间段被包含在被分割时间段的数据
    '''
    for i in range(len(data2)):  # i 定位 data1['Timestamp']
        if data2.iloc[i, 0] < data1.iloc[j, 0]:  # 会分成59段
            number += 1
        else:
            sum += number
            print(j, number)
            number_list.append(number)
            number = 1
            if j < 57:
                j += 1
            else:
                number_list.append(len(data2) - sum)
                break
    print(number_list)
    i = 0
    for number in number_list:
        i += number
    print(i, len(number_list))

    data3 = pd.DataFrame(columns=['date', 'calorie', 'total_carb', 'sugar', 'protein'])
    for i in range(number_list[0]):
        data3 = data3.append(pd.DataFrame({'date': [data1.iloc[0, 0]],
                                           'calorie': [data1.iloc[0, 1]],
                                           'total_carb': [data1.iloc[0, 2]],
                                           'sugar': [data1.iloc[0, 3]],
                                           'protein': [data1.iloc[0, 4]]}), ignore_index=True)

    for i in range(1, len(number_list) - 1):
        for j in range(number_list[i]):
            data3 = data3.append(pd.DataFrame({'date': [data1.iloc[i, 0]],
                                               'calorie': [data1.iloc[i, 1]],
                                               'total_carb': [data1.iloc[i, 2]],
                                               'sugar': [data1.iloc[i, 3]],
                                               'protein': [data1.iloc[i, 4]]}), ignore_index=True)
        print(data3.shape)

    for i in range(number_list[-1]):
        data3 = data3.append(pd.DataFrame({'date': [data1.iloc[-1, 0]],
                                           'calorie': [data1.iloc[-1, 1]],
                                           'total_carb': [data1.iloc[-1, 2]],
                                           'sugar': [data1.iloc[-1, 3]],
                                           'protein': [data1.iloc[-1, 4]]}), ignore_index=True)

    display(data3)
    data3.to_csv('./output/003/' + 'Food_Log_003_new_new.csv', index=False)

# data1 = pd.read_csv('./output/003/' + 'Food_Log_003_new.csv', sep=',', na_values='NULL', )
# data2 = pd.read_csv('./output/003/' + 'Dexcom_003_new.csv', sep=',', na_values='NULL', )
# porcess_Foodlog_003_time(data1, data2)


def process_003(data1, data2, data3, data4, data5, data6, data7, data8):

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
    display(data)
    data.to_csv('./output/003/' + '003_new.csv', index=False)
    print("finish!")

# output_path = './output/003/'
# data1 = pd.read_csv(output_path + 'ACC_003_new.csv', sep=',', na_values='NULL', )
# data2 = pd.read_csv(output_path + 'BVP_003_new.csv', sep=',', na_values='NULL', )
# data3 = pd.read_csv(output_path + 'EDA_003_new.csv', sep=',', na_values='NULL', )
# data4 = pd.read_csv(output_path + 'Food_Log_003_new_new.csv', sep=',', na_values='NULL', )
# data5 = pd.read_csv(output_path + 'HR_003_new.csv', sep=',', na_values='NULL', )
# data6 = pd.read_csv(output_path + 'IBI_003_new.csv', sep=',', na_values='NULL', )
# data7 = pd.read_csv(output_path + 'TEMP_003_new.csv', sep=',', na_values='NULL', )
# data8 = pd.read_csv(output_path + 'Dexcom_003_new.csv', sep=',', na_values='NULL', )
# process_003(data1, data2, data3, data4, data5, data6, data7, data8)