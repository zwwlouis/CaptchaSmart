import os
import json
import random
from TensorNN4Captcha import TensorNN4C

from util import DataReform

# 正常人为操作数组
hum_op = []
hum_test = []
hum_index = 0
# global hum_index
# 机器操作数组
mach_op = []
mach_test = []
mach_index = 0
# 包含null值得数据数目
null_data_num = 0
# opStr长度过短的情况
op_illegal_num = 0


def getMoveOPData(filePath, num=0):
    """
    从文件中获取滑动验证信息
    :param filePath: 文件路径
    :param num: 读取文件数量
    :return:
    """
    if not os.path.exists(filePath):
        print("not exist")
        return
    pathDir = os.listdir(filePath)
    read_num = 0
    # 设置固定的随机种子
    random.seed(100)
    for i in range(len(pathDir)):
        each = pathDir[i]
        if each.startswith("move"):
            print("读取文件: %s" % each)
            fileName = os.path.join('%s/%s' % (filePath, each));
            read_num += 1
            readFile(fileName)
        # 读取num个文件，num=0时读取全部
        if num != 0 and read_num >= num:
            break
    print("人操作：训练数据：%d" % len(hum_op))
    print("       测试数据：%d" % len(hum_test))
    print("机器操作：训练数据：%d" % len(mach_op))
    print("         测试数据：%d" % len(mach_test))


def readFile(fileName, test_rate=0.2):
    """
    读取文件
    :param fileName:
    :param test_rate: 测试集切割的比例
    :return:
    """
    global op_illegal_num
    global null_data_num
    fopen = open(fileName, 'r')
    for eachline in fopen:
        # 剪切开头的日期得到操作对象的json字符串
        jsonStr = eachline[19:]
        # 生成操作对象
        opObj = json.loads(jsonStr)
        if ('op' in opObj) and (opObj['op'] != 'NULL'):
            if ('label' in opObj) and (opObj['label'] != 'NULL'):
                try:
                    op_array = DataReform.json_reform(opObj['op'], 10)
                except (TypeError, IndexError, ZeroDivisionError) as err:
                    continue
                op_array = DataReform.d2toline(op_array)
                rand = random.random()
                if len(op_array) == 0:
                    op_illegal_num += 1
                elif opObj['label'] == 1:
                    if rand > test_rate:  # 切割一部分数据到测试集
                        hum_op.append(op_array)
                    else:
                        hum_test.append(op_array)
                elif opObj['label'] == 2:
                    if rand > test_rate:
                        mach_op.append(op_array)
                    else:
                        mach_test.append(op_array)
                continue
        null_data_num += 1

def get_hum_data(batch_size):
    """
    只获得用户操作数据，不需要label
    :param batch_size:
    :return:
    """
    global hum_index
    hum_len = batch_size
    batch_data = []
    # 获取普通数据
    if (hum_len + hum_index) > len(hum_op):
        # 当需求长度大于剩余数组元素时，先将尾部的部分全部加入
        batch_data += (hum_op[hum_index:])
        hum_index = hum_len - len(hum_op) + hum_index
        # 再通过开头部分补足
        batch_data += (hum_op[0: hum_index])
    else:
        batch_data += (hum_op[hum_index: hum_index + hum_len])
        hum_index = hum_index + hum_len
    return batch_data

def get_mach_data(batch_size):
    """
    只获得机器操作数据，不需要label
    :param batch_size:
    :return:
    """
    global mach_index
    mach_len = batch_size
    batch_data = []
    # 获取机器数据
    if (mach_len + mach_index) > len(mach_op):
        # 当需求长度大于剩余数组元素时，先将尾部的部分全部加入
        batch_data += (mach_op[mach_index:])
        mach_index = mach_len - len(mach_op) + mach_index
        # 再通过开头部分补足
        batch_data += (mach_op[0: mach_index])
    else:
        batch_data += (mach_op[mach_index: mach_index + mach_len])
        mach_index = mach_index + mach_len
    return batch_data

def get_data(batch_size):
    """
    获取指定数量的训练数据
    :param batch_size:
    :return:
    """
    global hum_index
    global mach_index
    hum_len = int(batch_size / 2)
    mach_len = batch_size - hum_len
    batch_data = []
    batch_label = []

    # 生成label
    for i in range(hum_len):
        batch_label.append([0, 1])
    for i in range(hum_len):
        batch_label.append([1, 0])

    # 获取普通数据
    if (hum_len + hum_index) > len(hum_op):
        # 当需求长度大于剩余数组元素时，先将尾部的部分全部加入
        batch_data += (hum_op[hum_index:])
        hum_index = hum_len - len(hum_op) + hum_index
        # 再通过开头部分补足
        batch_data += (hum_op[0: hum_index])
    else:
        batch_data += (hum_op[hum_index: hum_index + hum_len])
        hum_index = hum_index + hum_len

    # 获取机器数据
    if (mach_len + mach_index) > len(mach_op):
        # 当需求长度大于剩余数组元素时，先将尾部的部分全部加入
        batch_data += (mach_op[mach_index:])
        mach_index = mach_len - len(mach_op) + mach_index
        # 再通过开头部分补足
        batch_data += (mach_op[0: mach_index])
    else:
        batch_data += (mach_op[mach_index: mach_index + mach_len])
        mach_index = mach_index + mach_len

    return {
        "data": batch_data,
        "label": batch_label
    }


def get_hum_test_data(bath_size):
    """
    只获取用户操作的测试数据
    :param bath_size:
    :return:
    """
    # 测试数据总是从头开始取
    hum_len = bath_size
    bath_data = []
    if hum_len < len(hum_test):
        bath_data += hum_test[:hum_len]
    else:
        bath_data += hum_test
    return bath_data


def get_mach_test_data(bath_size):
    """
    只获取机器操作的测试数据
    :param bath_size:
    :return:
    """
    # 测试数据总是从头开始取
    mach_len = bath_size
    bath_data = []
    bath_label = []
    for i in range(mach_len):
        bath_label.append([1, 0])
    if mach_len < len(mach_test):
        bath_data += mach_test[:mach_len]
    else:
        bath_data += mach_test
    return {
        "data": bath_data,
        "label": bath_label
    }

def get_test_data(bath_size):
    """
    获取测试集数据
    :param bath_size:
    :return:
    """
    # 测试数据总是从头开始取
    hum_len = int(bath_size / 2)
    mach_len = bath_size - hum_len
    bath_data = []
    bath_label = []
    for i in range(hum_len):
        bath_label.append([0, 1])
    for i in range(mach_len):
        bath_label.append([1, 0])

    if hum_len < len(hum_test):
        bath_data += hum_test[:hum_len]
    else:
        bath_data += hum_test

    if mach_len < len(mach_test):
        bath_data += mach_test[:mach_len]
    else:
        bath_data += mach_test

    return {
        "data": bath_data,
        "label": bath_label
    }

def shuffle_data():
    """
    打乱数据
    :return:
    """
    random.seed(100)
    random.shuffle(hum_op)
    random.seed(100)
    random.shuffle(hum_test)
    random.seed(100)
    random.shuffle(mach_op)
    random.seed(100)
    random.shuffle(mach_test)


def main():
    print('hello world!')
    # 获得当前文件路径
    curDir = os.path.dirname(__file__)
    parentDir = os.path.dirname(curDir)
    # 获取机器数据
    resource1 = os.path.join("%s%s" % (parentDir, "/resource/3"))
    getMoveOPData(resource1)
    # 获取普通数据
    resource2 = os.path.join("%s%s" % (parentDir, "/resource/2"))
    getMoveOPData(resource2, 1)
    # 读取完数据后对数据作打乱操作
    shuffle_data()
    model_name = "test"
    model_path = os.path.join(parentDir, 'tensorflow_model_save', model_name)

    tnn = TensorNN4C(model_path, hidden_layers=1, hidden_nodes=[200], in_len=20, out_len=2,func=["sigmoid","softmax"])
    tnn.set_train_param(0.01)
    test_data = get_test_data(900)
    origin_rate = tnn.correct_rate(test_data["data"], test_data["label"],"初始数据在测试集")



    for i in range(5000):
        train_data = get_data(100)
        tnn.batch_train(train_data["data"], train_data["label"])
        if (i+1) % 100 == 0:
            tnn.correct_rate(test_data["data"], test_data["label"], "i = %d 测试集" % (i+1))
    advance = tnn.correct_rate(test_data["data"], test_data["label"]) - origin_rate
    print("\n\n训练集正确率提升了 %.2f %%" % (advance*100))


    # 是否无视结果进行保存
    save_ignore = True
    if advance >= 0 or save_ignore:
        tnn.save(tnn.path_name)
        print("模型保存完毕！")
    # 最后进行绘图操作
    tnn.result_scatter_plot(test_data["data"], test_data["label"])

if __name__ == "__main__":
    main()
