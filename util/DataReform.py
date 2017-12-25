import random
import json

import Constants

LABEL_SWITCH = {
    # 0-undefined 1-human 2-machine
    0: [0, 0],
    1: [0, 1],
    2: [1, 0]
}


def fetch_data(dataSet, keySet, sampleNum=10, evt_filter=(Constants.MOUSE_MOVE_EVENT, Constants.MOUSE_DOWN_EVENT)):
    """
    从数据库查询所得中提取所需的训练数据
    :param dataSet: 输入的原始数据
    :param keySet: 输入数据对应的标签
    :param sampleNum: 采样数
    :param evt_filter: 获初步取鼠标操作类型，默认只获得鼠标移动事件
    :return:
    """
    opIndex = keySet.index('op')
    labelIndex = keySet.index('label')
    # 输入数据
    input = []
    # 数据标签
    label = []
    humNum = 0
    machNum = 0
    for item in dataSet:
        opline = reform(item[opIndex], sampleNum, evt_filter)
        # 对op_str数据进行reform操作
        if len(opline) > 0:
            input.append(reform(item[opIndex], sampleNum, evt_filter))
            intLabel = int(item[labelIndex])
            label.append(LABEL_SWITCH[intLabel])
            if intLabel == 1:
                humNum += 1
            elif intLabel == 2:
                machNum += 1

    entry = {
        'input': input,
        'label': label,
        'humlabel': humNum,
        'machlabel': machNum
    }
    return entry


def reform(data, sampleNum, evt_filter=(Constants.MOUSE_MOVE_EVENT, Constants.MOUSE_DOWN_EVENT)):
    """
    对操作数据进行采样变换，一次验证生成一个统一长度的数组
    :param data: 鼠标操作数据
    :param sampleNum: 采样数
    :param evt_filter: 获初步取鼠标操作类型，默认只获得鼠标移动事件
    :return:
    """
    # 数据分割
    opList = data.split('-')
    opListInt = []
    for i in range(len(opList)):
        # 0-x 1-y 2-type 3-time
        opList[i] = opList[i].split(',')
        if int(opList[i][2]) in evt_filter:
            # 转换string为int
            opListInt.append([
                int(opList[i][0]),
                int(opList[i][1]),
                int(opList[i][3])
            ])
    for i in range(len(opListInt)):
        # 消除初始位置的影响
        if i > 0:
            opListInt[i][0] = opListInt[i][0] - opListInt[0][0]
            opListInt[i][1] = opListInt[i][0] - opListInt[0][1]
    # 初始位置设为0
    opListInt[0][0] = 0
    opListInt[0][1] = 0
    if len(opListInt) < 2:
        return []
    else:
        return sampling(opListInt, sampleNum)


def json_reform(data, sample_num, evt_filter=(Constants.MOUSE_MOVE_EVENT, Constants.MOUSE_DOWN_EVENT)):
    """
    对操作数据进行采样变换，输入数据为JSON字符串
    :param data: 鼠标操作数据
    :param sample_num: 采样数
    :param evt_filter: 获初步取鼠标操作类型，默认只获得鼠标移动事件
    :return:
    """
    # JSON转换为数组对象
    opList = json.loads(data)
    # 如果参数不符合要求（为一个数组），或操作长度过短则直接返回空数组
    if (not isinstance(opList, list)) or len(opList) < 2:
        return [];
    # 计算该操作数列经历的总时间
    starttime = opList[0][-1]
    endtime = opList[-1][-1]
    totaltime = endtime - starttime
    opListInt = []
    for i in range(len(opList)):
        # 0-x 1-y 2-type 3-time
        if int(opList[i][2]) in evt_filter:
            # 转换string为int
            opListInt.append([
                int(opList[i][0]),
                int(opList[i][1]),
                int(opList[i][3])
            ])
    for i in range(1, len(opListInt)):
        # 消除初始位置的影响
        opListInt[i][0] = opListInt[i][0] - opListInt[0][0]
        opListInt[i][1] = opListInt[i][1] - opListInt[0][1]
    # 初始位置设为0
    opListInt[0][0] = 0
    opListInt[0][1] = 0
    if len(opListInt) < 3:
        return []
    else:
        return sampling(opListInt, sample_num)


def sampling(data, sample_num):
    """
    :param data: 待采样数组，每一个数据行为[x,y,time],数据长度必须大于等于2
    :param sample_num: 采样数
    :param isRound: 是否取整
    :return:
    """
    starttime = data[0][2]
    endtime = data[-1][2]
    totaltime = endtime - starttime
    # 计算采样时间间隔
    dt = totaltime / (sample_num - 1);

    # 将位置信息除以总时间，来消除不同长度的数据对采样带来的影响
    dataCoe = 1 / totaltime * 1000;
    sampleOp = []

    # 单独加入第一个点
    sampleOp.append([data[0][0], data[0][1]])
    # 数据指针指向第二个数据
    dataPoint = 1;
    for i in range(sample_num - 1):
        starttime += dt
        while (dataPoint < len(data)) and (starttime > data[dataPoint][2]):
            dataPoint += 1
        if (dataPoint) == len(data):
            # 先判断是否达到末尾，直接将尾结点插入
            sampleOp.append([data[-1][0], data[-1][1]])
        else:
            sampleOp.append(interpolation(data[dataPoint - 1], data[dataPoint], starttime))

    # 乘上时间反比系数
    for i in range(len(sampleOp)):
        sampleOp[i][0] = round(sampleOp[i][0] * dataCoe)
        sampleOp[i][1] = round(sampleOp[i][1] * dataCoe)
    return sampleOp


def interpolation(pos1, pos2, t):
    """
    插值函数，根据两端时间和位置获得中间某个时间点的位置信息
    :param pos1: 前一个位置 [x,y,time]
    :param pos2: 后一个位置
    :param t: 当前时间
    :return:
    """
    t1 = pos1[2]
    t2 = pos2[2]
    coe = (t - t1) / (t2 - t1)
    newPos = []
    newPos.append((pos2[0] - pos1[0]) * coe + pos1[0])
    newPos.append((pos2[1] - pos1[1]) * coe + pos1[1])

    return newPos


def data_cut(data, rate):
    """
    切割测试集 input和label数组需要同样长
    :param data: 输入数据
    :param rate:  测试数据所占百分比
    :return:
    """
    random.seed(100)
    testData = []
    # 提取测试集
    for i in range(len(data)):
        rn = random.uniform(0, 1)
        if rn < rate:
            testData.append(data[i])
    # 从训练集中删除测试数据
    for i in range(len(testData)):
        data.remove(testData[i])
    # FIXME
    # 对训练集做打乱操作
    # random.seed(1)
    # random.shuffle(data)
    return testData


def toline(data):
    """
    op数组整体线性化
    :param data: 样例[[[0,0],[44,1],[93,2]],
                      [[0,0],[42,2],[60,2]]]
                ->   [[0,0,44,1,93,2],
                      [0,0,42,2,60,2]]
    :return:
    """

    for i in range(len(data)):
        row = data[i]
        data[i] = d2toline(row)
    return data


def d2toline(data):
    """
    op线性化
    :param data:
            样例：[[0,0],[44,1],[93,2]] -> [0,0,44,1,93,2]
    :return:
    """
    rowline = [];
    for item in data:
        rowline.append(item[0])
        rowline.append(item[1])
    return rowline


def main():
    op_str = "[[422,676,1,0],[606,676,6,675],[606,676,5,675]]"
    op_array = json_reform(op_str, 10)
    print(op_array)


if __name__ == "__main__":
    main()
