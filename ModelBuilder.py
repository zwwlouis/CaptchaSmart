from TensorNN4Captcha import TensorNN4C
import os


class Builder:
    # @staticmethod
    def buildModel(self):
        path = os.getcwd()
        model_name = "test"
        model_path = os.path.join(path,'CaptchaSmart/tensorflow_model_save/captcha_smart',model_name)
        print(os.path.exists(model_path))
        print(os.listdir(model_path))
        tnn = TensorNN4C(model_path, hidden_layers=1, hidden_nodes=[200], in_len=20, out_len=2)
        return tnn

def main():
    b = Builder()
    b.buildModel()

if __name__ == '__main__':
    main()


