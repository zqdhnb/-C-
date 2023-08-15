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