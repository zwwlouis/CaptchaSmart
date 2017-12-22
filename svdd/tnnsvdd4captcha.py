import tensorflow as tf
from sklearn import svm
from tensorflow.python.framework.errors_impl import InvalidArgumentError
import os
import numpy as np


class tnnsvdd4C:
    tnn_model_path = "tensorflow_model_save/test_20"
    """
    单分类SVM实现
    """

    def __init__(self):
        # 获得当前文件路径
        curDir = os.path.dirname(__file__)
        parentDir = os.path.dirname(curDir)
        self.model_path = os.path.join(parentDir, self.tnn_model_path)

        self.sess = tf.Session();
        try:
            save = tf.train.import_meta_graph(self.model_path + "/model.ckpt.meta")
            save.restore(sess=self.sess, save_path=self.model_path + "/model.ckpt")
            print("tnn模型加载完成")
            self.input = tf.get_collection('input')[0]
            self.output = tf.get_collection('output')[0]
            self.feature = tf.get_collection('feature')[0]
            self.tnnaccu = tf.get_collection('accuracy')[0]
            self.label = tf.get_collection('label')[0]
        except (IOError, InvalidArgumentError) as err:
            print(err)

    def getTnnAccuracy(self, test_data, test_label):
        rate = self.sess.run(self.tnnaccu, feed_dict={self.input: test_data, self.label: test_label})
        print("tnn模型正确率为 %.2f %%" % (rate * 100))

    def svddFit(self, hum_data, nu=0.1, gamma=0.1):
        """
        :param nu: 测试集错误率
        :param gamma: 针对RBF核函数，e指数上的系数
        :return:
        """
        # FIXME
        hum_feature = self.sess.run(self.feature,feed_dict={self.input:hum_data})
        # hum_feature = hum_data
        # print("svdd拟合输入的特征空间")
        # print(hum_feature)
        """
          kernel : string, optional (default='rbf')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
         a callable.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.
        """
        # self.clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)
        self.clf = svm.OneClassSVM(nu=nu, kernel="sigmoid")
        # self.clf = svm.OneClassSVM(nu=nu, kernel="linear")
        # self.clf = svm.OneClassSVM(nu=nu, kernel="poly",degree = 2)
        self.clf.fit(hum_feature)

    def svddPredict(self, data):
        return self.clf.predict(data)

    def svddAccuracy(self, data, label, name=""):
        feature = self.sess.run(self.feature,feed_dict={self.input: data})
        # feature = data
        predict = self.clf.predict(feature)
        label = np.array(label)
        result = label[:, 1] - label[:, 0]
        equal = (predict == result)
        right_sum = np.sum(equal)
        data_len = len(data)
        rate = right_sum / data_len
        if name != "":
            print("svdd在 %s 上的预测正确率为 %.2f %%" % (name, rate * 100))
