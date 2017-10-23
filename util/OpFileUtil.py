import os
import json
from util import DataReform

# 正常人为操作数组
global humOp
# 机器操作数组
global machOp
# 包含null值得数据数目
global nullDataNum
# opStr长度过短的情况
global opIllegalNum
humOp = []
machOp = []
nullDataNum = 0
opIllegalNum = 0


def getMoveOPData(filePath):
    if not os.path.exists(filePath):
        print("not exist")
        return
    pathDir = os.listdir(filePath)
    for each in pathDir:
        print("读取文件: %s" % each)
        fileName = os.path.join('%s/%s' % (filePath, each));
        readFile(fileName)


def readFile(fileName):
    fopen = open(fileName, 'r')
    for eachline in fopen:
        # 剪切开头的日期得到操作对象的json字符串
        jsonStr = eachline[19:]
        # 生成操作对象
        opObj = json.loads(jsonStr)
        if ('op' in opObj) and (opObj['op'] != 'NULL'):
            if ('label' in opObj) and (opObj['label'] != 'NULL'):
                opArray = DataReform.json_reform(opObj['op'], 30)
                if len(opArray) == 0:
                    opIllegalNum += 1
                elif opObj['label'] == 1:
                    humOp.append([opObj['op'], opObj['label']])
                elif opObj['label'] == 2:
                    machOp.append([opObj['op'], opObj['label']])
                continue
        nullDataNum += 1;


def main():
    print('hello world!')
    # 获得当前文件路径
    curDir = os.path.dirname(__file__)
    parentDir = os.path.dirname(curDir)
    print(parentDir)
    resource = os.path.join("%s%s" % (parentDir, "/resource"))
    print(resource)
    getMoveOPData(resource)

    print(humOp)
    print(machOp)
    print(nullDataNum)


if __name__ == "__main__":
    main()
