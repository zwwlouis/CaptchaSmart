from TensorNN4Captcha import TensorNN4C
from util import DataReform
from util.DBConnector import CaptchDBConn
import os


class Builder:
    @staticmethod
    def buildModel():
        path = "/tensorflow_model_save/captcha_smart/"
        model_name = "test"
        model_path = os.path.join(path,model_name)
        tnn = TensorNN4C("test", hidden_layers=1, hidden_nodes=[200], in_len=20, out_len=2)
        return tnn

def main():
    b = Builder()
    b.buildModel()

if __name__ == '__main__':
    main()


