import os
from util import OpFileUtil
from svdd.tnnsvdd4captcha import tnnsvdd4C
pos = [1, 1, 1]


def main():
    print('hello world!')
    # 获得当前文件路径
    curDir = os.path.dirname(__file__)
    parentDir = curDir
    # 获取机器数据
    resource1 = os.path.join("%s%s" % (parentDir, "/resource/3"))
    OpFileUtil.getMoveOPData(resource1)
    # 获取普通数据
    resource2 = os.path.join("%s%s" % (parentDir, "/resource/2"))
    OpFileUtil.getMoveOPData(resource2, 1)
    # 读取完数据后对数据作打乱操作
    OpFileUtil.shuffle_data()
    model_name = "test"
    model_path = os.path.join(parentDir, 'tensorflow_model_save', model_name,"model.ckpt")
    test_data = OpFileUtil.get_test_data(900)


    tsvdd = tnnsvdd4C()
    tsvdd.getTnnAccuracy(test_data["data"],test_data["label"]);

    hum_data = OpFileUtil.get_hum_data(6000);

    tsvdd.svddFit(hum_data["data"],nu = 0.05,gamma = 0.1)
    print("svdd 拟合完成")
    tsvdd.svddAccuracy(hum_data["data"], hum_data["label"], "用户训练数据")
    tsvdd.svddAccuracy(test_data["data"],test_data["label"],"测试数据")
    mach_test_data = OpFileUtil.get_mach_data(400)
    tsvdd.svddAccuracy(mach_test_data["data"],mach_test_data["label"],"机器测试数据")




if __name__ == '__main__':
    main()

