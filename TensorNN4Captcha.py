import tensorflow as tf
import os
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from util import DataReform
from util.DBConnector import CaptchDBConn
import numpy
import matplotlib.pyplot as plt

# 生成权重矩阵
# from util.DBConnector import CaptchDBConn


def weight_variable(name, shape, seed, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev, seed=seed)
    return tf.Variable(initial, name=name)


# 生成
def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# 读取已有的变量
def get_weight_variable(name, shape, seed, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev, seed=seed)
    return tf.get_variable(name, shape=shape)


# 生成
def get_bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, shape=shape)


class TensorNN4C:
    inputLen = 20
    hiddNum_1 = 100
    outputLen = 2

    # input = None
    # output = None
    # label = None
    # loss = None
    # accuracy = None
    # sess = None

    def get_default_hyper_param(self, hidden_layers):
        """
        定义默认的超参数
        :return:
        """
        # 设置激活函数
        if self.func is None:
            self.func = ["sigmoid"] * (hidden_layers + 1)
            self.func[-1] = "softmax"
        # 设置系数初始的标准差
        if self.dev is None:
            self.dev = [0.1] * (hidden_layers + 1)
            self.dev[-1] = 1e-10

    def __init__(self, modelPath, hidden_layers=1, hidden_nodes=[300], in_len=20, out_len=2, func=None, dev=None):
        """
        初始化网络模型
        :param modelPath: 用于定义命名空间及存储模型的名字
        :param hidden_layers: 隐层数目
        :param hidden_nodes: 隐层节点数目——存储节点数目的数组，数组长度于隐层数目一致
        :param in_len: 输入数据长度
        :param out_len: 输出数据长度
        """
        self.name = "test"
        self.weight = [None] * (hidden_layers + 1)
        self.bias = [None] * (hidden_layers + 1)
        self.out = [None] * (hidden_layers + 1)
        # 输入标签
        self.label = tf.placeholder('float', [None, out_len])
        # 输入操作数据
        self.input = tf.placeholder('float', [None, in_len])
        self.func = func
        self.dev = dev
        # 设置默认的超参数
        self.get_default_hyper_param(hidden_layers)
        # 计算得到模型存储路径，模型文件存放于根目录的save/name文件夹下
        self.path_name = modelPath + "/model.ckpt"

        self.sess = tf.Session()
        self.loss = None
        self.accuracy = None
        # 初始的正确率，对应加载的模型的情况，如果训练使得正确率提升才进行保存
        self.orgin_accuracy = 0
        # 生成模型
        self.gen_model(hidden_layers, hidden_nodes, in_len, out_len)
        # self.loss = tf.reduce_mean(tf.square(self.label - self.out[-1]))
        self.loss = -tf.reduce_mean(self.label * tf.log(self.out[-1]))
        correct_test = tf.equal(tf.argmax(self.out[-1], 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_test, "float"))
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        # 设置学习速率
        self.train_step = tf.train.GradientDescentOptimizer(0.005).minimize(self.loss)
        # # 打印参数
        # self.print_all()
        # 尝试加载模型
        is_success = self.load(self.path_name)
        if is_success:
            print("加载模型成功！")
        else:
            print("未找到对应模型！")


            # self.input = tf.placeholder('float', [None, self.inputLen])
            # self.W1 = weight_variable([self.inputLen, self.hiddNum_1], 10)
            # self.b1 = bias_variable([self.hiddNum_1])
            # self.y1 = tf.nn.sigmoid(tf.matmul(self+5.input, self.W1) + self.b1)
            # self.W2 = weight_variable([self.hiddNum_1, self.outputLen], 10, 1e-10)
            # self.b2 = bias_variable([self.outputLen])
            # self.label = tf.placeholder('float', [None, 2])
            # self.output = tf.nn.sigmoid(tf.matmul(self.y1, self.W2) + self.b2)
            # # self.loss = -tf.reduce_mean(self.label * tf.log(self.output))
            # self.loss = tf.reduce_mean(tf.square(self.label-self.output))
            # correct_test = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.label, 1))
            # self.accuracy = tf.reduce_mean(tf.cast(correct_test, "float"))
            #
            # init = tf.global_variables_initializer()
            # self.sess = tf.Session()
            # self.sess.run(init)
            # print("W1 = " + self.sess.run(self.W1).__str__())
            # print("b1 = " + self.sess.run(self.b1).__str__())
            # print("W2 = " + self.sess.run(self.W2).__str__())
            # print("b1 = " + self.sess.run(self.b2).__str__())

    def gen_model(self, hidden_layers, hidden_nodes, in_len, out_len):
        """
        生成模型
        :param hidden_layers:
        :param hidden_nodes:
        :param in_len:
        :param out_len:
        :return:
        """
        # 将输出层算入隐层统一处理
        hidden_nodes.append(out_len)
        with tf.variable_scope(self.name):
            # 依次创建各层隐层节点
            for i in range(hidden_layers + 1):
                with tf.variable_scope("hiden_%d" % i):
                    if i == 0:
                        # 处理输入层节点
                        self.weight[0] = get_weight_variable("weight_%d" % 0, [in_len, hidden_nodes[0]],
                                                             stddev=self.dev[0], seed=10)
                        self.bias[0] = get_bias_variable("bias_%d" % 0, [hidden_nodes[0]])
                        self.out[0] = self.active_func(tf.matmul(self.input, self.weight[0]) + self.bias[0],
                                                       self.func[0])
                    else:
                        # 对剩下各层依次统一处理，输出层为out[-1]
                        self.weight[i] = get_weight_variable("weight_%d" % i, [hidden_nodes[i - 1], hidden_nodes[i]],
                                                             stddev=self.dev[i],
                                                             seed=10)
                        self.bias[i] = get_bias_variable("bias_%d" % i, [hidden_nodes[i]])
                        self.out[i] = self.active_func(tf.matmul(self.out[i - 1], self.weight[i]) + self.bias[i],
                                                       self.func[i])
    def set_train_param(self, learn_rate):
        # 设置学习速率
        self.train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)


    def train(self, data_source, test_in, test_label, batch_size = 100, loop_num = 1000):
        """
        进行训练
        :param data_source: 训练数据源，需要包含函数 get_data(batch_size)
        :param test_in:
        :param test_label:
        :return:
        """
        self.orgin_accuracy = self.correct_rate(test_in, test_label)

        for i in range(loop_num):
            data_batch = data_source.get_data(batch_size)
            self.sess.run(self.train_step, feed_dict={self.input: data_batch["data"], self.label: data_batch["label"]})
            if i % 10 == 0:
                # # FIXME
                # print("i = %d" % i)
                # print("W2 = " + self.sess.run(self.weight[1]).__str__())
                # oneInput = []
                # oneInput.append(input_data[j * batch_size])
                # oneLabel = input_label[j * batch_size]
                # print("out = " + self.sess.run(self.out[-1], feed_dict={self.input: oneInput}).__str__())
                # print("2ndout = " + self.sess.run(self.out[-2], feed_dict={self.input: oneInput}).__str__())
                self.correct_rate(test_in, test_label, "i = %d 测试集"%i)

                # if correct_rate > 0.85:
                #     break
                # print("w1",self.sess.run(self.W1))
                # print("w2",self.sess.run(self.W2))

        for j in range(10):
            batch_in = test_in[(j * 3): ((j + 1) * 3)]
            batch_label = test_label[(j * 3): ((j + 1) * 3)]
            print("\n第%d组数据，每组三个" % (j))
            print(self.sess.run(self.out[-1], feed_dict={self.input: batch_in, self.label: batch_label}))
            print(batch_label)

        # 训练完成后对结果进行保存
        advance = self.correct_rate(test_in, test_label) - self.orgin_accuracy
        if advance > 0:
            # 如果正确率有改善才进行保存
            print("\n\n训练集正确率提升了 %.2f %%" % advance)
            self.save(self.path_name)
        else:
            print("\n\n训练集正确率提升了 %.2f %%，模型未保存" % advance)

    def batch_train(self, input_data, input_label):
        """
        一次只训练一个batch
        :param input_data:
        :param input_label:
        :return:
        """
        data_len = len(input_data)
        label_len = len(input_label)
        self.sess.run(self.train_step, feed_dict={self.input: input_data, self.label: input_label})
        # nloss = self.sess.run(self.loss, feed_dict={self.input: input_data, self.label: input_label})


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
        tf.add_to_collection("output", self.out[-1])
        save_path = save.save(sess=self.sess, save_path=path)

    def load(self, path="save/model.ckpt"):
        try:
            save = tf.train.Saver()
            save.restore(sess=self.sess, save_path=path)
            return True
        except (IOError, InvalidArgumentError) as err:
            # print(err)
            return False

    def correct_rate(self, data, label, name=""):
        """
        获得当前模型在特定输入下的正确率
        :param data:
        :param label:
        :return:
        """
        rate = self.sess.run(self.accuracy, feed_dict={self.input: data, self.label: label})
        if not name == "":
            print("%s 上的正确率 ：%.2f %%" % (name, rate*100))
        return rate


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
        label_max = tf.argmax(test_label,1)
        # 确定用户数据的位置
        hum_index = self.sess.run(label_max) == 1
        # 提取用户数据
        hum_out = output[hum_index]
        hum_right_index = numpy.argmax(hum_out,1) == 1
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
        plt.scatter(hum_jdg_right[:,0],hum_jdg_right[:,1],marker='.',color="blue",label="hum right")
        plt.scatter(hum_jdg_wrong[:,0],hum_jdg_wrong[:,1],marker='.',color="blueviolet",label="hum wrong")

        plt.scatter(mach_jdg_right[:, 0], mach_jdg_right[:, 1], marker='.',color="red",label="mach right")
        plt.scatter(mach_jdg_wrong[:, 0], mach_jdg_wrong[:, 1], marker='.',color="orange",label="mach wrong")
        plt.xlabel('Machine indice')
        plt.ylabel('Human indice')
        plt.legend()
        plt.show()
        input()




def main():
    # 主函数
    print('hello world!')
    captcha = CaptchDBConn()
    cur = captcha.get_op_by_type(10)
    # print(cur.fetchall())
    all = cur.fetchall()
    entry = DataReform.fetch_data(all, ["op", "feature", "label", "date"])
    input = DataReform.toline(entry['input'])
    label = entry['label']

    # for i in range(len(input)):
    #     print(input[i])
    #     print(label[i])
    print("共有 %d 条人工操作" % (entry['humlabel']))
    print("共有 %d 条机器操作" % (entry['machlabel']))
    testinput = DataReform.data_cut(input, 0.1)
    testlabel = DataReform.data_cut(label, 0.1)
    print("训练集合：%d条  测试集合：%d条" % (len(input), len(testinput)))
    tnn = TensorNN4C("test", hidden_layers=1, hidden_nodes=[200], in_len=20, out_len=2)
    tnn.correct_rate(input, label, "训练集")

    # tnn.load()
    # print("########################加载完成#####################")
    # tnn.print_all()
    # tnn.correct_rate(input, label, "训练集")

    tnn.train(input, label, testinput, testlabel)
    #作图
    tnn.result_scatter_plot(testinput,testlabel)


if __name__ == "__main__":
    main()
