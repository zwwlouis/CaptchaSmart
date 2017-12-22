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


def readPathForMoveOP(filePath, num=0, type="both"):
    """
    从文件中获取滑动验证信息
    :param filePath: 文件路径
    :param num: 读取文件数量
    :param type 读取操作类型 "hum"-只读用户数据 "mach"-只读机器数据 "both"-都读取
    :return:
    """
    global hum_op, hum_test, mach_op, mach_test
    if not os.path.exists(filePath):
        print("not exist")
        return
    # 获得该路径下所有的文件
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
            # 读取文件
            result = readFile(fileName)
            # 添加数据
            if (type == "hum") or (type == "both"):
                hum_op += result["hum_op"]
                hum_test += result["hum_test"]
            if (type == "mach") or (type == "both"):
                mach_op += result["mach_op"]
                mach_test += result["mach_test"]
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
    file_op_illegal_num = 0
    file_null_data_num = 0
    fopen = open(fileName, 'r')
    file_hum = []
    file_mach = []
    file_hum_test = []
    file_mach_test = []
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
                    # print(err)
                    continue
                op_array = DataReform.d2toline(op_array)
                rand = random.random()
                if len(op_array) == 0:
                    file_op_illegal_num += 1
                elif opObj['label'] == 1:
                    if rand > test_rate:  # 切割一部分数据到测试集
                        file_hum.append(op_array)
                    else:
                        file_hum_test.append(op_array)
                elif opObj['label'] == 2:
                    if rand > test_rate:
                        file_mach.append(op_array)
                    else:
                        file_mach_test.append(op_array)
                continue
        file_null_data_num += 1
    hum_inc = len(file_hum) + len(file_hum_test)
    mach_inc = len(file_mach) + len(file_mach_test)
    print("读取：%s 文件包含用户数据 %d条，机器数据 %d条" % (fileName, hum_inc, mach_inc))
    # print("错误数据  %d条" % (file_null_data_num + file_op_illegal_num))
    return {
        "hum_op": file_hum,
        "hum_test": file_hum_test,
        "mach_op": file_mach,
        "mach_test": file_mach_test,
    }


def filterPathForHumOp(tnn, filePath, targetPath=None):
    """
    通过输入的模型对路径下的文件进行筛选，如果文件中快速标签为“用户数据”的操作召回率过高，则移动该文件至targetPath所指定的路径
    :param tnn: tensorflow神经网络模型
    :param filePath: 读取路径
    :param targetPath: 可疑文件存放路径（如果为空则不移动）
    :return:
    """
    if not os.path.exists(filePath):
        print("not exist")
        return
        # 获得该路径下所有的文件
    pathDir = os.listdir(filePath)
    # 统计所有用户操作数
    total_hum_num = 0
    # 统计所有用户操作被正确识别的数量
    total_hum_correct_num = 0
    for i in range(len(pathDir)):
        each = pathDir[i]
        if each.startswith("move"):
            print("读取文件: %s" % each)
            fileName = os.path.join('%s/%s' % (filePath, each));
            # 读取文件
            result = readFile(fileName,0)
            test_hum_data = result["hum_op"]
            test_hum_label = [[0, 1]] * len(test_hum_data)
            rate = tnn.correct_rate(test_hum_data, test_hum_label, "该文件所有用户数据")
            data_len = len(test_hum_label)
            total_hum_num += data_len
            total_hum_correct_num += rate * data_len
    print("所有用户数据上的整体正确率为 %.2f %%" % (total_hum_correct_num / total_hum_num * 100))


def testPathForOp(tnn, filePath,type="hum"):
    """
    通过输入的模型对路径下的文件进行测试
    :param tnn: tensorflow神经网络模型
    :param filePath: 读取路径
    :param type: "hum"-用户   "mach"-机器
    :return:
    """
    if not os.path.exists(filePath):
        print("not exist")
        return
        # 获得该路径下所有的文件
    pathDir = os.listdir(filePath)
    total_num = 0
    total_correct_num = 0
    for i in range(len(pathDir)):
        each = pathDir[i]
        if each.startswith("move"):
            print("读取文件: %s" % each)
            fileName = os.path.join('%s/%s' % (filePath, each));
            # 读取文件
            result = readFile(fileName,0)
            test_data = result[type+"_op"]
            data_len = len(test_data)
            if(type == "hum"):
                test_label = [[0, 1]] * data_len
            else:
                test_label = [[1, 0]] * data_len
            rate = tnn.correct_rate(test_data, test_label, "该文件所有 %s数据"%type)
            total_num += data_len
            total_correct_num += rate * data_len
    print("所有%s数据上的整体正确率为 %.2f %%" % (type, total_correct_num / total_num * 100))

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
    # 生成label
    batch_label = [[0, 1]] * len(batch_data)
    return {
        "data": batch_data,
        "label": batch_label
    }


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
    # 生成label
    batch_label = [[1, 0]] * len(batch_data)
    return {
        "data": batch_data,
        "label": batch_label
    }


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
    # 生成label
    batch_label += [[0, 1]] * len(batch_data)

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

    # 生成机器label(补足剩余长度)
    batch_label += [[1, 0]] * (len(batch_data) - len(batch_label))

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
    batch_data = []
    batch_label = []
    if hum_len < len(hum_test):
        batch_data += hum_test[:hum_len]
    else:
        batch_data += hum_test

    # 生成label
    batch_label += [[0, 1]] * len(batch_data)
    return {
        "data": batch_data,
        "label": batch_label
    }


def get_mach_test_data(bath_size):
    """
    只获取机器操作的测试数据
    :param bath_size:
    :return:
    """
    # 测试数据总是从头开始取
    mach_len = bath_size
    batch_data = []
    batch_label = []

    if mach_len < len(mach_test):
        batch_data += mach_test[:mach_len]
    else:
        batch_data += mach_test
    # 生成label
    batch_label += [[1, 0]] * len(batch_data)
    return {
        "data": batch_data,
        "label": batch_label
    }


def get_test_data(bath_size, start=0):
    """
    获取测试集数据
    :param bath_size:
    :param start 开始位置
    :return:
    """
    global hum_test
    global mach_test
    hum_len = int(bath_size / 2)
    mach_len = bath_size - hum_len
    batch_data = []
    batch_label = []

    if (hum_len + start) < len(hum_test):
        batch_data += hum_test[start:hum_len + start]
        add_len = hum_len
    else:
        batch_data += hum_test
        add_len = len(hum_test)
    # 生成用户label
    batch_label += [[0, 1]] * add_len

    if (mach_len + start) < len(mach_test):
        batch_data += mach_test[start:mach_len + start]
        add_len = mach_len
    else:
        batch_data += mach_test
        add_len = len(mach_test)
    # 生成机器label
    batch_label += [[1, 0]] * add_len
    return {
        "data": batch_data,
        "label": batch_label
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


def test_all_data(tnn):
    """
    测试读取到的所有数据
    :param tnn:
    :return:
    """
    test_data_hum = get_hum_data(10000)
    data_len = len(test_data_hum["label"])
    tnn.correct_rate(test_data_hum["data"], test_data_hum["label"], "%d用户数据训练集" % data_len)

    test_data_mach = get_mach_data(10000)
    data_len = len(test_data_mach["label"])
    tnn.correct_rate(test_data_mach["data"], test_data_mach["label"], "%d机器数据训练集" % data_len)

    test_data_hum = get_hum_test_data(10000)
    data_len = len(test_data_hum["label"])
    tnn.correct_rate(test_data_hum["data"], test_data_hum["label"], "%d用户数据测试集" % data_len)

    test_data_mach = get_mach_test_data(10000)
    data_len = len(test_data_mach["label"])
    tnn.correct_rate(test_data_mach["data"], test_data_mach["label"], "%d机器数据测试集" % data_len)


def main():
    print('hello world!')
    # 获得当前文件路径
    curDir = os.path.dirname(__file__)
    parentDir = os.path.dirname(curDir)
    # 获取机器数据
    resource1 = os.path.join("%s%s" % (parentDir, "/resource/op_logs/test/train"))
    readPathForMoveOP(resource1)
    # 获取普通数据
    resource2 = os.path.join("%s%s" % (parentDir, "/resource/op_logs/prod/origin"))
    readPathForMoveOP(resource2, 1)

    # 读取完数据后对数据作打乱操作
    shuffle_data()
    model_name = "tnn"
    model_path = os.path.join(parentDir, 'tensorflow_model_save', model_name)

    tnn = TensorNN4C(model_path, hidden_layers=1, hidden_nodes=[200], in_len=20, out_len=2, func=["sigmoid", "softmax"])
    tnn.set_train_param(0.01)
    test_data = get_test_data(1000)
    origin_rate = tnn.correct_rate(test_data["data"], test_data["label"], "初始数据在测试集")

    # for i in range(10000):
    #     train_data = get_data(50)
    #     tnn.batch_train(train_data["data"], train_data["label"])
    #     if (i + 1) % 100 == 0:
    #         tnn.correct_rate(test_data["data"], test_data["label"], "i = %d 测试集" % (i + 1))
    print("*******************训练完成*********************")
    # 测试所有数据
    test_all_data(tnn)

    advance = tnn.correct_rate(test_data["data"], test_data["label"]) - origin_rate
    print("\n\n训练集正确率提升了 %.2f %%" % (advance * 100))

    # 是否无视结果进行保存！慎改
    save_ignore = False
    if advance > 0 or save_ignore:
        tnn.save(tnn.path_name)
        print("模型保存完毕！")
    else:
        print("模型未保存！")

    # 获取待筛选的用户数据
    # userOpRes = os.path.join("%s%s" % (parentDir, "/resource/op_logs/prod/train"))
    # filterPathForHumOp(tnn, userOpRes)

    userOpRes = os.path.join("%s%s" % (parentDir, "/resource/op_logs/click/1"))
    machOpRes = os.path.join("%s%s" % (parentDir, "/resource/op_logs/test/train"))
    testRes =  os.path.join("%s%s" % (parentDir, "/resource/op_logs/test/other"))
    # filterPathForHumOp(tnn, userOpRes)
    testPathForOp(tnn, machOpRes, "mach")
    testPathForOp(tnn,testRes,"mach")
    # 最后进行绘图操作
    # tnn.result_scatter_plot(test_data["data"], test_data["label"])

    # for i in range(2):
    #     test_mini = get_test_data(10, 10 * i)
    #     tnn.plot_feature(test_mini["data"], test_mini["label"])
    # # 进入作图模式
    # tnn.plot_mode(test_data["data"]);

    # while(True):


if __name__ == "__main__":
    main()
