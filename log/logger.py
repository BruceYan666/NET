import logging
from logging import handlers

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):

        self.logger = logging.getLogger()#创建一个logger
        self.logger.setLevel(self.level_relations.get(level))#Log等级总开关
        format_str = logging.Formatter(fmt)#设置日志格式

        #创建一个handler,用于输出到控制台
        ch = logging.StreamHandler()#往屏幕上输出
        ch.setLevel(logging.INFO)  # 输出到console的log等级的开关
        ch.setFormatter(format_str) #设置屏幕上显示的格式

        #创建一个handler，用于写入日志文件
        fh = logging.FileHandler(filename=filename,mode='a')
        fh.setLevel(logging.INFO) # 用于写到file的等级开关
        fh.setFormatter(format_str)#设置文件里写入的格式

        self.logger.addHandler(ch) #把对象加到logger里
        self.logger.addHandler(fh)

if __name__ == '__main__':
    log = Logger('test.log',level='debug')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('警告')
    log.logger.error('报错')
    log.logger.critical('严重')


