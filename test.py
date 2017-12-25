import json
import os
fileName = "resource/test.txt"
path = os.getcwd()
fopen = open(fileName, 'w')
# lines = fopen.readlines()
# lines.append("testtest")
# print(lines)
# str = json.dumps(lines)
fopen.write("56")
fopen.flush()
