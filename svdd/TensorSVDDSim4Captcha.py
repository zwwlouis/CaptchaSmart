import tensorflow as tf
import os
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from util import DataReform, OpFileUtil
import numpy
import matplotlib.pyplot as plt


def weight_variable(name, shape, seed=10, stddev=0.1):
    # 生成权重变量
    initial = tf.truncated_normal(shape, stddev=stddev, seed=seed)
    return tf.Variable(initial, name=name)


def bias_variable(name, shape):
    # 生成阈值变量
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def get_weight_variable(name, shape, seed, stddev=0.1):
    # 读取已有的权重变量
    initial = tf.truncated_normal(shape, stddev=stddev, seed=seed)
    return tf.get_variable(name, shape=shape)


def get_bias_variable(name, shape):
    # 读取已有的阈值变量
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, shape=shape)


"""
简化版单分类支持向量机
"""


class TensorSVDDSim4C:
    inputLen = 20
    hiddNum_1 = 100
    outputLen = 2

    # input = None
    # output = None
    # label = None
    # loss = None
    # accuracy = None
    # sess = None

    def __init__(self, modelPath, hiddNum=[100, 100], in_len=20, out_len=2):
        """
        初始化网络模型
        :param modelPath: 用于定义命名空间及存储模型的名字
        :param hidden_layers: 隐层数目
        :param hidden_nodes: 隐层节点数目——存储节点数目的数组，数组长度于隐层数目一致
        :param in_len: 输入数据长度
        :param out_len: 输出数据长度
        """
        self.name = "svdd"
        modelPath = os.path.join(modelPath, self.name)
        self.path_name = modelPath + "/model.ckpt"
        # 输入标签
        self.label = tf.placeholder('float', [None, out_len])
        # 输入操作数据
        self.input = tf.placeholder('float', [None, in_len])
        # 第一层
        self.W1 = weight_variable("W1", [in_len, hiddNum[0]], 10, 0.01)
        self.b1 = bias_variable("b1", [hiddNum[0]])
        self.y1 = tf.nn.sigmoid(tf.matmul(self.input, self.W1) + self.b1)
        # 第二层
        self.W2 = weight_variable("W2", [hiddNum[0], hiddNum[1]], 10, 0.01)
        self.b2 = bias_variable("b2", [hiddNum[1]])
        self.y2 = tf.nn.sigmoid(tf.matmul(self.y1, self.W2) + self.b2)

        self.center = bias_variable("center", [hiddNum[1]])
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        """python
           # 'x' is [[1., 1.]
           #         [2., 2.]]
           tf.reduce_mean(x) == > 1.5
           tf.reduce_mean(x, 0) == > [1.5, 1.5]
           tf.reduce_mean(x, 1) == > [1., 2.]
           """
        self.distance = tf.sqrt(tf.reduce_sum(tf.square(self.y2 - self.center), 1))
        self.mean_dis = tf.reduce_mean(self.distance)

        # 设置学习速率
        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.mean_dis)
        # # 尝试加载模型
        # is_success = self.load(self.path_name)
        # if is_success:
        #     print("加载模型成功！")
        # else:
        #     print("未找到对应模型！")

    def batch_train(self, input_data):
        """
        一次只训练一个batch
        :param input_data:
        :param input_label:
        :return:
        """
        self.sess.run(self.train_step, feed_dict={self.input: input_data})

    def active_func(self, tensor, func_name):
        """
        根据所需的激活函数类型返回tensor对象
        :param tensor:
        :param func_name:
        :return:
        """
        if func_name == "relu":
            return tf.nn.relu(tensor)
        elif func_name == "sigmoid":
            return tf.nn.sigmoid(tensor)
        elif func_name == "softmax":
            return tf.nn.softmax(tensor)

    def test(self, test_in):
        text_mat = [test_in]
        test_label = self.sess.run(self.out[-1], feed_dict={self.input: text_mat})
        return test_label

    def save(self, path="save/model.ckpt"):
        dir = os.path.dirname(path)
        # 如果目录不存在则提前创建目录
        if not os.path.exists(dir):
            os.makedirs(dir)
        save = tf.train.Saver()
        save_path = save.save(sess=self.sess, save_path=path)

    def load(self, path="save/model.ckpt"):
        try:
            save = tf.train.Saver()
            save.restore(sess=self.sess, save_path=path)
            return True
        except (IOError, InvalidArgumentError) as err:
            # print(err)
            return False

    def get_radius(self, data, rate=0.95):
        """
        得到输入数据的分布半径
        :param rate:　数据透过率
        :return:
        """
        distance = self.sess.run(self.distance, feed_dict={self.input: data})
        if (isinstance(distance, numpy.ndarray)):
            distance.sort()
            data_len = len(distance)
            index = int(data_len * rate)
            radius = distance[index]
            print("当前数据在透过率%.2f 下的半径为 ：%.5f" % (rate, radius))
            return radius
        else:
            print("获取数据半径出错");

    def get_mean_dis(self, data):
        """
        得到当前数据的平均距离
        :param data:
        :return:
        """
        mean_dis = self.sess.run(self.mean_dis, feed_dict={self.input: data})
        print("当前数据平均距离为 ：%.5f" % (mean_dis))

    def correct_rate(self, data,radius,label=[], name=""):
        """
        获得当前模型在特定输入下的正确率
        :param data:
        :param label:
        :param radius:数据半径
        :return:
        """
        self.out = tf.nn.sigmoid(self.distance - radius)
        ret_label = self.sess.run(self.out, feed_dict={self.input: data})
        print(name)
        print(ret_label)

    def get_variable(self, var, name=""):
        """
        获得及打印当前变量
        :param var:
        :param name: 打印的变量名
        :return:
        """
        if isinstance(var, tf.Variable):
            # 获取矩阵型状
            shape = var.get_shape()
            if len(shape) > 1 and shape[0] > shape[1]:
                var = tf.transpose(var)
            array = self.sess.run(var)
            # 变量名不为空则进行输出
            if not name == "":
                print("变量 %s :" % name)
                print(array)
            return array
        else:
            return None

    def print_all(self):
        """
        打印所有变量
        :return:
        """
        # 打印所有权重信息
        for w in self.weight:
            self.get_variable(w, w.name)
        for b in self.bias:
            self.get_variable(b, b.name)

    def result_scatter_plot(self, test_data, test_label):
        """
        根据测试数据，做出结果散点图
        :param data:
        :param label:
        :return:
        """
        output = self.sess.run(self.out[-1], feed_dict={self.input: test_data, self.label: test_label})
        # 根据输入数据最大值位置
        label_max = tf.argmax(test_label, 1)
        # 确定用户数据的位置
        hum_index = self.sess.run(label_max) == 1
        # 提取用户数据
        hum_out = output[hum_index]
        hum_right_index = numpy.argmax(hum_out, 1) == 1
        hum_jdg_right = hum_out[hum_right_index]
        hum_jdg_wrong = hum_out[~hum_right_index]
        # 提取机器数据
        mach_index = ~hum_index
        mach_out = output[mach_index]
        mach_right_index = numpy.argmax(mach_out, 1) == 0
        mach_jdg_right = mach_out[mach_right_index]
        mach_jdg_wrong = mach_out[~mach_right_index]

        # 作图
        my_fig = plt.figure("human or machine")
        # 画出用户数据点，判断正确-蓝色 错误-紫色
        plt.scatter(hum_jdg_right[:, 0], hum_jdg_right[:, 1], marker='.', color="blue", label="hum right")
        plt.scatter(hum_jdg_wrong[:, 0], hum_jdg_wrong[:, 1], marker='.', color="blueviolet", label="hum wrong")

        plt.scatter(mach_jdg_right[:, 0], mach_jdg_right[:, 1], marker='.', color="red", label="mach right")
        plt.scatter(mach_jdg_wrong[:, 0], mach_jdg_wrong[:, 1], marker='.', color="orange", label="mach wrong")
        plt.xlabel('Machine indice')
        plt.ylabel('Human indice')
        plt.legend()
        plt.show()


def main():
    print('hello world!')
    # 获得当前文件路径
    curDir = os.path.dirname(__file__)
    parentDir = os.path.dirname(curDir);
    # 获取机器数据
    resource1 = os.path.join("%s%s" % (parentDir, "/resource/3"))
    OpFileUtil.readPathForMoveOP(resource1)
    # 获取普通数据
    resource2 = os.path.join("%s%s" % (parentDir, "/resource/2"))
    OpFileUtil.readPathForMoveOP(resource2, 1)
    # 读取完数据后对数据作打乱操作
    OpFileUtil.shuffle_data()

    model_path = os.path.join(parentDir, 'tensorflow_model_save')
    test_data = OpFileUtil.get_hum_test_data(1000)
    test_mach_data = OpFileUtil.get_mach_data(500);
    svdd = TensorSVDDSim4C(model_path)
    radius = 1;
    for i in range(5000):
        train_data = OpFileUtil.get_hum_data(200)
        svdd.batch_train(train_data)
        if (i + 1) % 100 == 0:
            svdd.get_mean_dis(test_data)
            radius = svdd.get_radius(test_data, 0.9)
            # advance = tnn.correct_rat
            # e(test_data["data"], test_data["label"]) - origin_rate
    svdd.correct_rate(test_mach_data, radius = radius,name="机器数据")
    svdd.correct_rate(test_data, radius = radius,name="用户数据")

if __name__ == "__main__":
    main()
