import json

fileName = "resource/test.txt"

fopen = open(fileName, 'a+')
lines = fopen.readlines()
lines.append("testtest")
print(lines)
str = json.dumps(lines)
fopen.write(str)
fopen.flush()
