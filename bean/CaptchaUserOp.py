import json
class CaptchaUserOp:

     def __init__(self):
         self.id = 1
         self.appId = 2



def main():
    # 主函数
    op = CaptchaUserOp()
    op.id = 2
    op.appId=4
    opDict = op.__dict__
    print(type(op))
    print(opDict)
    print(type(opDict))
    opJson = json.dumps(opDict)
    print(opJson)
    print(type(opJson))
    opLoadDict = json.loads(opJson)
    print(opLoadDict)
    print(type(opLoadDict))
    print(opLoadDict['id'])
    print(opLoadDict['appId'])
    # print(opLoadDict['op'])


if __name__ == "__main__":
    main()


