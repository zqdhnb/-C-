import numpy as np
import pandas as pd
import warnings
import IPython.display as display
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

input_path = '../data/001/'
output_path = '../output/001/'

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
# data1 = pd.read_csv(input_path+'Dexcom_001.csv', sep=',', na_values='NULL', ) #nrows=15
# process_Dexcom(data1)

'''处理Food_Log数据'''


def process_Food_Log(data1):
    # df['列名'].isnull().sum(axis=0)
    # for iter in data1.columns:
    # print(iter, data1[iter].isnull().sum(axis=0))
    # 删除缺失值非常多的列
    data1.drop(columns='time_end', inplace=True)
    data1.drop(columns='unit', inplace=True)
    data1.drop(columns='dietary_fiber', inplace=True)
    data1.drop(columns='total_fat', inplace=True)
    # 两个时间时完全一样的，删掉一个
    data1.drop(columns='time_begin', inplace=True)

    # for iter in data1.columns:
    # print(iter, data1[iter].isnull().sum(axis=0))
    # dd = data1.isnull().sum(axis=1)
    # for i in dd:
    # print(i)
    # 最后一行有两项空缺，由于food一项没有对应值，故将其删掉
    data1.drop(data1.tail(1).index, inplace=True)
    # print(data1.isnull().sum(axis=1))

    # dd = pd.concat([data1['logged_food'], data1['searched_food']], axis=1)
    # for i, row in dd.iterrows():
    # print(row)
    # searched_food似乎不是摄入的，删除以较少工作量
    data1.drop(columns='searched_food', inplace=True)

    # log_food不好处理，干脆也删掉
    data1.drop(columns='logged_food', inplace=True)

    '''
    df1 = data1['date'].str.split('-', expand=True)

    df1.columns = ['year', 'month', 'day']
    df1.astype('int')
    for i, row in df1.iterrows():
        row['day'] = (int(row['year']) - 2020) + (int(row['month']) - 2) + int(row['day'])
    df1 = df1['day']

    df2 = data1['time'].str.split(':', expand=True)
    df2.columns = ['hour', 'minute', 'second']
    df = pd.concat([df1, df2], axis=1)
    #print(df.head(10))
    #删除并合并
    data1.drop(columns='date', inplace=True)
    data1.drop(columns='time', inplace=True)
    data1 = pd.concat([df, data1], axis=1)

    #logged_food不能一个个文件处理，需要集中处理
    '''

    # date和time合并
    for i, row in data1.iterrows():
        data1.iloc[i, 0] = str(row['date']) + ' ' + str(row['time'])
        # row['date'] = (row['date']) + ' ' + row['time']
        # print(row['date'])

    data1.drop(columns='time', inplace=True)
    data1.drop(columns='amount', inplace=True)
    # print(data1.head(10))
    # 合并同一时间的增量
    # print(len(data1))
    drop = []
    for i in range(len(data1) - 1):
        if data1.iloc[i, 0] == data1.iloc[i + 1, 0]:
            drop.append(i)
            data1.iloc[i + 1, 1] = data1.iloc[i + 1, 1].astype(int) + data1.iloc[i, 1].astype(int)
            data1.iloc[i + 1, 2] = data1.iloc[i + 1, 2].astype(int) + data1.iloc[i, 2].astype(int)
            data1.iloc[i + 1, 3] = data1.iloc[i + 1, 3].astype(int) + data1.iloc[i, 3].astype(int)
            data1.iloc[i + 1, 4] = data1.iloc[i + 1, 4].astype(int) + data1.iloc[i, 4].astype(int)
    for i in drop:
        data1.drop(index=i, inplace=True)

    print(data1.columns)
    print(data1.shape)

    print(data1.head(10))
    data1.to_csv(output_path + 'Food_Log_001_new.csv', index=False)


# data2 = pd.read_csv(input_path+'Food_Log_001.csv', sep=',', na_values='NULL', ) #nrows=15
# process_Food_Log(data2)

'''以处理后的Dexcom为准对齐Dexcom和Food_Log的时间 注意：需要对Food_Log数据进行预处理'''


def process_DexcomANDFood_Log(data1, data2):
    data1['Timestamp'] = pd.to_datetime(data1['Timestamp'])
    data2['date'] = pd.to_datetime(data2['date'])

    number_list = []  # 统计每一个时间段的个数
    '''
    if data1.iloc[0, 0] > data2.iloc[0, 0]:
        print(True)
    else:
        print(False)
    '''
    number = 0  # 计数器，初始化为0
    j = 0  # j 定位 data2['date']
    '''
    遍历整个大的时间段，计算小于定位值的样本数量，作为需要填充的区间样本数量
    注意：只适用于分割时间段被包含在被分割时间段的数据
    '''
    for i in range(len(data1)):  # i 定位 data1['Timestamp']
        if data1.iloc[i, 0] < data2.iloc[j, 0]:
            number += 1
        else:
            number_list.append(number)
            number = 1
            if j < 43:
                j += 1
            else:
                number_list.append(len(data1) - i)
                break
    print(number_list)
    i = 0
    for number in number_list:
        i += number
    print(i, len(number_list))

    # 开始构建与Dexcom规格相同的foodlog
    data3 = pd.DataFrame(columns=['date', 'calorie', 'total_carb', 'sugar', 'protein'])

    for i in range(number_list[0]):
        data3 = data3.append(pd.DataFrame({'date': [data2.iloc[0, 0]],
                                           'calorie': [data2.iloc[0, 1]],
                                           'total_carb': [data2.iloc[0, 2]],
                                           'sugar': [data2.iloc[0, 3]],
                                           'protein': [data2.iloc[0, 4]]}), ignore_index=True)
    print(data3.shape)
    for i in range(1, len(number_list) - 1):
        for j in range(number_list[i]):
            data3 = data3.append(pd.DataFrame({'date': [data2.iloc[i, 0]],
                                               'calorie': [data2.iloc[i, 1]],
                                               'total_carb': [data2.iloc[i, 2]],
                                               'sugar': [data2.iloc[i, 3]],
                                               'protein': [data2.iloc[i, 4]]}), ignore_index=True)
        print(data3.shape)
    for i in range(number_list[-1]):
        data3 = data3.append(pd.DataFrame({'date': [data2.iloc[-1, 0]],
                                           'calorie': [data2.iloc[-1, 1]],
                                           'total_carb': [data2.iloc[-1, 2]],
                                           'sugar': [data2.iloc[-1, 3]],
                                           'protein': [data2.iloc[-1, 4]]}), ignore_index=True)

    print(data3.head(20))
    print(data1.shape, data2.shape, data3.shape)
    data3.to_csv(output_path + 'Food_Log_001_new.csv', index=False)
    # print(data1.head(10))
    # print(data1['Timestamp'].head(10))
    # print(data2['date'].head(10))


# data1 = pd.read_csv(output_path + 'Dexcom_001_new.csv', sep=',', na_values='NULL',)# nrows=1800
# data2 = pd.read_csv(output_path + 'Food_Log_001_new.csv', sep=',', na_values='NULL', )
# process_DexcomANDFood_Log(data1, data2)


'''以处理后的Dexcom为准对齐Dexcom和TEMP的时间'''


def process_TempANDDexcom(data1, data2):
    # datetime转化
    data1['Timestamp'] = pd.to_datetime(data1['Timestamp'])  # 小数据
    data2['datetime'] = pd.to_datetime(data2['datetime'])  # 大数据

    number_list = []  # 统计每一个时间段的个数
    number = 0  # 计数器，初始化为0
    j = 0  # 定位器

    for i in range(len(data2)):
        if data2.iloc[i, 0] < data1.iloc[j, 0]:
            number += 1
        else:
            number_list.append(number)
            number = 1
            if j < len(data1) - 1:
                j += 1
            else:
                number_list.append(len(data2) - i)
                break
    '''            
    print(number_list)
    i = 0
    for number in number_list:
        i += number
    print(i, len(number_list))
    '''
    # 2537040条数据 2562段 number_list存储了每段多少数据

    # 开始合并 由于第一段数据过多，因此用后续数据为准
    data3 = pd.DataFrame(columns=['date', 'temp'])
    m = number_list[0] - 1  # 开始的下标
    n = m

    for i in range(0, len(number_list) - 1):  # 2561段
        n = n + number_list[i + 1]
        temp = data2.iloc[m:n, 1].mean()  # 前闭后开
        data3 = data3.append(pd.DataFrame({'date': [data1.iloc[i, 0]],
                                           'temp': [temp]}), ignore_index=True)
        m = n
    print(data1.shape, data2.shape, data3.shape)
    data3.to_csv(output_path + 'TEMP_001_new.csv', index=False)

    # print(data2.iloc[0, :], data2.iloc[-1, :])
    # print(data1.shape, data2.shape)
    # print(type(data2.iloc[0, 0]))
    # print(data1.head(10))
    # print(data2.head(10))


# data1 = pd.read_csv(output_path + 'Dexcom_001_new.csv', sep=',', na_values='NULL',)# nrows=1800
# data2 = pd.read_csv(input_path + 'TEMP_001.csv', sep=',', na_values='NULL',)
# process_TempANDDexcom(data1, data2)

'''以处理后的Dexcom为准对齐Dexcom和IBI的时间'''


def process_IbiANDDexcom(data1, data2):
    # datetime转化
    data1['Timestamp'] = pd.to_datetime(data1['Timestamp'])  # 小数据
    data2['datetime'] = pd.to_datetime(data2['datetime'])  # 大数据

    number_list = []  # 统计每一个时间段的个数
    number = 0  # 计数器，初始化为0
    j = 0  # 定位器

    for i in range(len(data2)):
        if data2.iloc[i, 0] < data1.iloc[j, 0]:
            number += 1
        else:
            number_list.append(number)
            number = 1
            if j < len(data1) - 1:
                j += 1
            else:
                number_list.append(len(data2) - i)
                break
    # 如果大数据没有完全包含小数据，那么对于小数据最后尾端需要额外增加处理
    if data2.iloc[-1, 0] < data1.iloc[-1, 0]:
        number_list.append(number)
    '''
    print(number_list)
    i = 0
    for number in number_list:
        i += number
    print(i, len(number_list))
    '''
    # 只有2561段 因此直接前对齐就行

    data3 = pd.DataFrame(columns=['date', 'ibi'])
    m = 0  # 开始的下标
    n = m

    for i in range(0, len(number_list)):  # 2561段
        n = n + number_list[i]
        ibi = data2.iloc[m:n, 1].mean()  # 前闭后开
        data3 = data3.append(pd.DataFrame({'date': [data1.iloc[i, 0]],
                                           'ibi': [ibi]}), ignore_index=True)
        m = n
    print(data1.shape, data2.shape, data3.shape)
    data3.to_csv(output_path + 'IBI_001_new.csv', index=False)

    # print(data1.shape, data2.shape)
    # print(data1.head(10))
    # print(data2.head(10))


# data1 = pd.read_csv(output_path + 'Dexcom_001_new.csv', sep=',', na_values='NULL', )  # nrows=1800
# data2 = pd.read_csv(input_path + 'IBI_001.csv', sep=',', na_values='NULL', )
# process_IbiANDDexcom(data1, data2)


# 001的 HR数据有明显问题，先做处理
def process_Hr(data2):
    df1 = data2['datetime'].str.split(' ', expand=True)  # 首先按照空格划分一下 2020-02-13 17:23:32
    print(data2.columns)
    df1.columns = ['day', 'time']
    df11 = df1['day'].str.split('-', expand=True)
    df11.columns = ['date', 'm', 'h']
    # print(df1, df11)
    for i in range(len(df11)):
        df11.iloc[i, 1] = str(df11.iloc[i, 2]) + '-02-' + str(int(df11.iloc[i, 2]) + 1) + ' ' + str(df1.iloc[i, 1])
    data1 = pd.concat([df11['m'], data2[' hr']], axis=1)
    data1.columns = ['datetime', 'hr']
    print(data1)
    data1.to_csv(output_path + 'HR_001_process.csv', index=False)


# data2 = pd.read_csv(input_path + 'HR_001.csv', sep=',', na_values='NULL',)
# process_Hr(data2)


def process_HrANDDexcom(data1, data2):
    data1['Timestamp'] = pd.to_datetime(data1['Timestamp'])  # 小数据
    data2['datetime'] = pd.to_datetime(data2['datetime'])  # 大数据

    number_list = []  # 统计每一个时间段的个数
    number = 0  # 计数器，初始化为0
    j = 0  # 定位器

    for i in range(len(data2)):
        if data2.iloc[i, 0] < data1.iloc[j, 0]:
            number += 1
        else:
            number_list.append(number)
            number = 1
            if j < len(data1) - 1:
                j += 1
            else:
                number_list.append(len(data2) - i)
                break
    # 如果大数据没有完全包含小数据，那么对于小数据最后尾端需要额外增加处理
    if data2.iloc[-1, 0] < data1.iloc[-1, 0]:
        number_list.append(number)
    '''
    print(number_list)
    i = 0
    for number in number_list:
        i += number
    print(i, len(number_list))
    '''
    # 634188个数据 2562段
    data3 = pd.DataFrame(columns=['date', 'hr'])
    m = 0  # 开始的下标
    n = m + number_list[0]

    for i in range(0, len(number_list) - 1):  # 2561段 以中间为准
        n = n + number_list[i + 1]
        hr = data2.iloc[m:n, 1].mean()  # 前闭后开
        data3 = data3.append(pd.DataFrame({'date': [data1.iloc[i, 0]],
                                           'hr': [hr]}), ignore_index=True)
        m = m + number_list[i]
    print(data1.shape, data2.shape, data3.shape)
    data3.to_csv(output_path + 'HR_001_new.csv', index=False)

    # print(data1.head(10))
    # print(data2.head(10))


# data1 = pd.read_csv(output_path + 'Dexcom_001_new.csv', sep=',', na_values='NULL', )  # nrows=1800
# data2 = pd.read_csv(output_path + 'HR_001_process.csv', sep=',', na_values='NULL', )
# process_HrANDDexcom(data1, data2)

def process_EdaANDDexcom(data1, data2):
    data1['Timestamp'] = pd.to_datetime(data1['Timestamp'])  # 小数据
    data2['datetime'] = pd.to_datetime(data2['datetime'])  # 大数据

    number_list = []  # 统计每一个时间段的个数
    number = 0  # 计数器，初始化为0
    j = 0  # 定位器

    for i in range(len(data2)):
        if data2.iloc[i, 0] < data1.iloc[j, 0]:
            number += 1
        else:
            number_list.append(number)
            number = 1
            if j < len(data1) - 1:
                j += 1
            else:
                number_list.append(len(data2) - i)
                break
    # 如果大数据没有完全包含小数据，那么对于小数据最后尾端需要额外增加处理
    if data2.iloc[-1, 0] < data1.iloc[-1, 0]:
        number_list.append(number)
    '''
    print(number_list)
    i = 0
    for number in number_list:
        i += number
    print(i, len(number_list))
    print(data1.shape, data2.shape)
    '''
    # 2537046个数据 2562段
    data3 = pd.DataFrame(columns=['date', 'eda'])
    m = 0  # 开始的下标
    n = m + number_list[0]

    for i in range(0, len(number_list) - 1):  # 2561段 以中间为准
        n = n + number_list[i + 1]
        eda = data2.iloc[m:n, 1].mean()  # 前闭后开
        data3 = data3.append(pd.DataFrame({'date': [data1.iloc[i, 0]],
                                           'eda': [eda]}), ignore_index=True)
        m = m + number_list[i]
    print(data1.shape, data2.shape, data3.shape)
    data3.to_csv(output_path + 'EDA_001_new.csv', index=False)


# data1 = pd.read_csv(output_path + 'Dexcom_001_new.csv', sep=',', na_values='NULL', )  # nrows=1800
# data2 = pd.read_csv(input_path + 'EDA_001.csv', sep=',', na_values='NULL', )
# process_EdaANDDexcom(data1, data2)


def process_BvpANDDexcom(data1, data2):
    data1['Timestamp'] = pd.to_datetime(data1['Timestamp'])  # 小数据
    data2['datetime'] = pd.to_datetime(data2['datetime'])  # 大数据

    number_list = []  # 统计每一个时间段的个数
    number = 0  # 计数器，初始化为0
    j = 0  # 定位器

    for i in range(len(data2)):
        if data2.iloc[i, 0] < data1.iloc[j, 0]:
            number += 1
        else:
            number_list.append(number)
            number = 1
            if j < len(data1) - 1:
                j += 1
            else:
                number_list.append(len(data2) - i)
                break
    # 如果大数据没有完全包含小数据，那么对于小数据最后尾端需要额外增加处理
    if data2.iloc[-1, 0] < data1.iloc[-1, 0]:
        number_list.append(number)
    '''
    print(number_list)
    i = 0
    for number in number_list:
        i += number
    print(i, len(number_list))
    print(data1.shape, data2.shape)
    '''
    #  2562段
    data3 = pd.DataFrame(columns=['date', 'bvp'])
    m = 0  # 开始的下标
    n = m + number_list[0]

    for i in range(0, len(number_list) - 1):  # 2561段 以中间为准
        n = n + number_list[i + 1]
        bvp = data2.iloc[m:n, 1].mean()  # 前闭后开
        data3 = data3.append(pd.DataFrame({'date': [data1.iloc[i, 0]],
                                           'bvp': [bvp]}), ignore_index=True)
        m = m + number_list[i]
    print(data1.shape, data2.shape, data3.shape)
    data3.to_csv(output_path + 'BVP_001_new.csv', index=False)


# data1 = pd.read_csv(output_path + 'Dexcom_001_new.csv', sep=',', na_values='NULL', )  # nrows=1800
# data2 = pd.read_csv(input_path + 'BVP_001.csv', sep=',', na_values='NULL', )
# process_BvpANDDexcom(data1, data2)


def process_DexcomANDAcc(data1, data2):
    data1['Timestamp'] = pd.to_datetime(data1['Timestamp'])  # 小数据
    data2['datetime'] = pd.to_datetime(data2['datetime'])  # 大数据

    number_list = []  # 统计每一个时间段的个数
    number = 0  # 计数器，初始化为0
    j = 0  # 定位器

    for i in range(len(data2)):
        if data2.iloc[i, 0] < data1.iloc[j, 0]:
            number += 1
        else:
            number_list.append(number)
            number = 1
            if j < len(data1) - 1:
                j += 1
            else:
                number_list.append(len(data2) - i)
                break
    # 如果大数据没有完全包含小数据，那么对于小数据最后尾端需要额外增加处理
    if data2.iloc[-1, 0] < data1.iloc[-1, 0]:
        number_list.append(number)
    '''
    print(number_list)
    i = 0
    for number in number_list:
        i += number
    print(i, len(number_list))
    print(data1.shape, data2.shape)
    '''
    #  2562段
    data3 = pd.DataFrame(columns=['date', 'acc_x', 'acc_y', 'acc_z'])
    m = 0  # 开始的下标
    n = m + number_list[0]

    for i in range(0, len(number_list) - 1):  # 2561段 以中间为准
        n = n + number_list[i + 1]
        acc_x = data2.iloc[m:n, 1].mean()  # 前闭后开
        acc_y = data2.iloc[m:n, 2].mean()
        acc_z = data2.iloc[m:n, 3].mean()
        data3 = data3.append(pd.DataFrame({'date': [data1.iloc[i, 0]],
                                           'acc_x': [acc_x],
                                           'acc_y': [acc_y],
                                           'acc_z': [acc_z]}), ignore_index=True)
        m = m + number_list[i]
    print(data1.shape, data2.shape, data3.shape)
    data3.to_csv(output_path + 'ACC_001_new.csv', index=False)


# data1 = pd.read_csv(output_path + 'Dexcom_001_new.csv', sep=',', na_values='NULL', )  # nrows=1800
# data2 = pd.read_csv(input_path + 'ACC_001.csv', sep=',', na_values='NULL', )


# process_DexcomANDAcc(data1, data2)
# (2561, 2) (40592838, 2) (2561, 2)
# (2561, 2) (20296428, 4) (2561, 4)

def process_all(data1, data2, data3, data4, data5, data6, data7, data8):
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
    data.to_csv(output_path + '001_new.csv', index=False)
    print("finish!")
    # print(data.shape)
    # print(data.head(10))


# data1 = pd.read_csv(output_path + 'ACC_001_new.csv', sep=',', na_values='NULL', )
# data2 = pd.read_csv(output_path + 'BVP_001_new.csv', sep=',', na_values='NULL', )
# data3 = pd.read_csv(output_path + 'EDA_001_new.csv', sep=',', na_values='NULL', )
# data4 = pd.read_csv(output_path + 'Food_Log_001_new.csv', sep=',', na_values='NULL', )
# data5 = pd.read_csv(output_path + 'HR_001_new.csv', sep=',', na_values='NULL', )
# data6 = pd.read_csv(output_path + 'IBI_001_new.csv', sep=',', na_values='NULL', )
# data7 = pd.read_csv(output_path + 'TEMP_001_new.csv', sep=',', na_values='NULL', )
# data8 = pd.read_csv(output_path + 'Dexcom_001_new.csv', sep=',', na_values='NULL', )
# process_all(data1, data2, data3, data4, data5, data6, data7, data8)


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
def pro():
    out_path = '../input/'
    data1 = pd.read_csv(out_path + '016_new.csv', sep=',', na_values='NULL', )
    data2 = Timestamp_to_date(data1)
    print(data2)
    data2.to_csv(out_path + '016_new.csv', index=False)




