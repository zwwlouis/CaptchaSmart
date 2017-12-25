import os
from util import OpFileUtil


def dataprocess(from_path, to_path, label="hum", sample_num=10):
    """
    文件预处理函数
    :param from_path:
    :param to_path:
    :param lebel: 处理的数据标签类型 "hum"-只处理用户数据 "mach"-只处理机器数据
    :param sample_num:  数据采样率
    :return:
    """
    if not os.path.exists(from_path):
        print("from_path not exist")
        return
    if not os.path.exists(to_path):
        print("to_path not exist")
        os.makedirs(to_path)
    to_files = os.listdir(to_path)
    to_files_dict = dict();
    # 根据目标文件夹已有的文件建立字典
    for file in to_files:
        to_files_dict[file] = file
    from_files = os.listdir(from_path)
    for file in from_files:
        if file not in to_files_dict:
            # 检查log文件是否已经预处理过了，如果有则跳过，没有则新建文件写入处理结果
            from_filename = os.path.join(from_path, file)
            to_filename = os.path.join(to_path, file)
            # 预处理不分测试集
            result = OpFileUtil.readFile(from_filename, test_rate=0, sample_num=sample_num)
            data = []
            if label == "hum":
                data = result["hum_op"]
            elif label == "mach":
                data = result["mach_op"]
            else:
                # FIXME 抛出异常
                print("参数错误")
                return
            OpFileUtil.writeFile(to_filename, data)


modelname = "tnn"
handle_mapping = {"resource/oplogs/prod/origin": ["resource/oplogs_%s/prod/origin" % modelname, "hum"],
                  "resource/oplogs/prod/train/trust": ["resource/oplogs_%s/prod/train" % modelname, "hum"],
                  "resource/oplogs/test/train": ["resource/oplogs_%s/test/train" % modelname, "mach"],
                  }


def main():
    path = os.getcwd()
    parentpath = os.path.dirname(path)

    modelname = "tnn"
    for key in handle_mapping.keys():
        from_path = os.path.join(parentpath, key)
        value = handle_mapping[key]
        to_path = os.path.join(parentpath, value[0])
        dataprocess(from_path, to_path, value[1], 10)


if __name__ == '__main__':
    main()
