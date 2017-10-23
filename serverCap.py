# python web 服务，用于接收验证操作字符串，返回验证结果
import os  # Python的标准库中的os模块包含普遍的操作系统功能
import re  # 引入正则表达式对象
import urllib  # 用于对URL进行编解码
from http.server import HTTPServer, BaseHTTPRequestHandler  # 导入HTTP处理相关的模块
import CaptchaSmart as capSmt
from CaptchaSmart.ModelBuilder import Builder
from CaptchaSmart import DataReform
from urllib import parse, request


# 自定义处理程序，用于处理HTTP请求
class TestHTTPHandler(BaseHTTPRequestHandler):
    tnn = None
    port = 8000

    # 处理GET请求
    def do_GET(self):
        # 页面输出模板字符串
        templateStr = '''<html>   
<head>   
<title>QR Link Generator</title>   
</head>   
<body>   
%s 
<br>   
<br>   
<form action="/qr" name=f method="GET"><input maxLength=1024 size=70   
name=s value="" title="Text to QR Encode"><input type=submit   
value="Show QR" name=qr>   
</form> 
</body>   
</html> '''

        # 将正则表达式编译成Pattern对象 r""未防止字符串转义
        pattern = re.compile(r"/checkOp/\?op=(.+)")
        # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
        match = pattern.match(self.path)
        # respond = b'';
        if match:
            # 使用Match获得分组信息
            op_str = match.group(1)
            # FIXME
            op_str = request.unquote(op_str)
            # 数据采样处理
            op_array = DataReform.json_reform(op_str, 10)
            # 输入变形为行向量
            op_line = DataReform.d2toline(op_array)
            label = self.tnn.test(op_line)
            print(label)
            thres = 0.5
            if (label[0][1] > thres):
                lNum = "HUMAN"
            else:
                lNum = "MACHINE"
            respond = str(lNum)
            respond = respond.encode(encoding='utf-8')

        self.protocal_version = 'HTTP/1.1'  # 设置协议版本
        self.send_response(200)  # 设置响应状态码
        self.send_header("Welcome", "Contect")  # 设置响应头
        self.end_headers()
        self.wfile.write(respond)


def start_server(port):
    # 初始化模型
    build = Builder()
    build.buildModel()
    TestHTTPHandler.tnn = build.tnn;

    # 启动服务器
    http_server = HTTPServer(('', int(port)), TestHTTPHandler)
    http_server.serve_forever()  # 设置一直监听并接收请求


def main():
    # 主函数
    os.chdir('static')  # 改变工作目录到 static 目录
    start_server(8000)  # 启动服务，监听8000端口
    print("服务启动成功！")
    # pattern = re.compile('www')
    # # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
    # match = pattern.match("aabccwwwd")
    # print(match.group(0))
    # print(match.group(1))
    # tnn = capSmt.main()
    # label = tnn.test([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    # print(label)


if __name__ == "__main__":
    main()
