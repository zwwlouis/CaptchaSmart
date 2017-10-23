from CaptchaSmart import TensorNN4Captcha,DataReform,CaptchDBConn

class Builder:
    tnn = None
    def buildModel(self):
        # 主函数
        print('hello world!')
        captcha = CaptchDBConn()
        cur = captcha.get_op_by_type(10)
        # print(cur.fetchall())
        all = cur.fetchall()
        entry = DataReform.fetch_data(all)
        input = DataReform.toline(entry['input'])
        label = entry['label']

        for i in range(len(input)):
            print(input[i])
            print(label[i])
        print("共有 %d 条人工操作" % (entry['humlabel']))
        print("共有 %d 条机器操作" % (entry['machlabel']))
        testentry = DataReform.testdata_cut(input, label, 0.2)
        testinput = testentry['testinput']
        testlabel = testentry['testlabel']
        print("训练集合：%d条  测试集合：%d条" % (len(input), len(testinput)))
        self.tnn = TensorNN4Captcha.TensorNN4C()
        self.tnn.train(input, label, testinput, testlabel)