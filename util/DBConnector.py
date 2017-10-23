import MySQLdb


class CaptchDBConn:
    __host = '10.3.246.37'
    __port = 3302
    __username = 'java'
    __password = 'z226java'
    __db = 'captchas'
    __table = 'captcha_user_op'
    __charset = 'utf8'

    def __init__(self):
        self.__conn = MySQLdb.Connection(host=self.__host, port=self.__port, user=self.__username,
                                         passwd=self.__password,
                                         db=self.__db, charset=self.__charset)
        self.__cur = self.__conn.cursor()

    def get_op_by_type(self, ctype):
        sql = 'select op_str,feature_str,label,create_time from '
        sql += self.__table
        sql += " where captcha_type = " + str(ctype)
        sql += " and label is not null"
        self.__cur.execute(sql)
        return self.__cur


def main():
    print('DBConnector')

if __name__ == "__main__":
    main()
