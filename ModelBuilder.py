import TensorNN4Captcha
from util import DataReform
from util.DBConnector import CaptchDBConn


class Builder:
    tnn = None

    def buildModel(self):
        # 主函数
        print('hello world!')
        captcha = CaptchDBConn()
        cur = captcha.get_op_by_type(10)
        # print(cur.fetchall())
        all = cur.fetchall()
        entry = DataReform.fetch_data(all, ["op", "feature", "label", "date"])
        input = DataReform.toline(entry['input'])
        label = entry['label']

        for i in range(len(input)):
            print(input[i])
            print(label[i])
        print("共有 %d 条人工操作" % (entry['humlabel']))
        print("共有 %d 条机器操作" % (entry['machlabel']))
        testinput = DataReform.data_cut(input, 0.1)
        testlabel = DataReform.data_cut(label, 0.1)
        print("训练集合：%d条  测试集合：%d条" % (len(input), len(testinput)))
        self.tnn = TensorNN4Captcha.TensorNN4C()
        self.tnn.train(input, label, testinput, testlabel)
