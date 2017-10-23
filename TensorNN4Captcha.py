import tensorflow as tf


# 生成权重矩阵
def weight_variable(shape, seed):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=seed)
    return tf.Variable(initial)


# 生成
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


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

    def __init__(self):
        self.input = tf.placeholder('float', [None, self.inputLen])
        self.W1 = weight_variable([self.inputLen, self.hiddNum_1], 10)
        self.b1 = bias_variable([self.hiddNum_1])

        self.y1 = tf.nn.sigmoid(tf.matmul(self.input, self.W1) + self.b1)

        self.W2 = weight_variable([self.hiddNum_1, self.outputLen], 10)
        self.b2 = bias_variable([self.outputLen])

        self.label = tf.placeholder('float', [None, 2])
        self.output = tf.nn.softmax(tf.matmul(self.y1, self.W2) + self.b2)
        self.loss = -tf.reduce_mean(self.label * tf.log(self.output))
        # self.loss = tf.reduce_mean(tf.square(self.label-self.output))
        correct_test = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_test, "float"))

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        # print(self.sess.run(self.W1))
        # print(self.sess.run(self.b1))
        # print(self.sess.run(self.W2))
        # print(self.sess.run(self.b2))

    def train(self, input_data, label_data, test_in, test_lab):
        # 设置学习速率
        train_step = tf.train.GradientDescentOptimizer(0.0002).minimize(self.loss)

        batch_size = 10
        for i in range(500):
            for j in range(int(len(input_data) / batch_size)):
                batch_in = input_data[(j * batch_size): ((j + 1) * batch_size)]
                batch_label = label_data[(j * batch_size): ((j + 1) * batch_size)]
                self.sess.run(train_step, feed_dict={self.input: batch_in, self.label: batch_label})
                nloss = self.sess.run(self.loss, feed_dict={self.input: batch_in, self.label: batch_label})
            if i % 10 == 0:
                test_correct_rate = self.sess.run(self.accuracy, feed_dict={self.input: test_in, self.label: test_lab})
                train_correct_rate = self.sess.run(self.accuracy, feed_dict={self.input: input_data, self.label: label_data})
                print(i, "test_CorrectRate: ", test_correct_rate, "train_CorrectRate: ", train_correct_rate)
                # if correct_rate > 0.85:
                #     break
                    # print("w1",self.sess.run(self.W1))
                    # print("w2",self.sess.run(self.W2))

        batch_size = 3
        for j in range(int(len(test_in) / batch_size)):
            batch_in = test_in[(j * batch_size): ((j + 1) * batch_size)]
            batch_label = test_lab[(j * batch_size): ((j + 1) * batch_size)]
            print("\n第%d组数据，每组三个" % (j))
            print(self.sess.run(self.output, feed_dict={self.input: batch_in, self.label: batch_label}))
            print(batch_label)
            # saver = tf.train.Saver()
            # saver.save(sess=sess, save_path = "zww/model.ckpt")

    def test(self, test_in):
        text_mat = [test_in]
        test_label = self.sess.run(self.output, feed_dict={self.input: text_mat})
        return test_label


def main():
    a = tf.constant([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
    sess = tf.Session()
    print(sess.run(a))


if __name__ == "__main__":
    main()
