import logging
import inspect

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class MyLogger(metaclass=Singleton):
    logger = None

    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s][%(threadName)s][%(message)s]",
            handlers=[
                logging.StreamHandler()
            ])

        self.logger = logging.getLogger(__name__ + '.logger')

    @staticmethod
    def __get_call_info():
        stack = inspect.stack()

        fn = stack[2][1][-30:]
        ln = stack[2][2]
        func = stack[2][3]

        return fn, func, ln
    
    def info(self, message, *args):
        message = "[{}][{}][{}]: {}".format(*self.__get_call_info(), message)
        self.logger.info(message, *args)
        
    def error(self, message, *args):
        message = "[{}][{}][{}]: {}".format(*self.__get_call_info(), message)
        self.logger.error(message, *args)
    def warning(self, message, *args):
        message = "[{}][{}][{}]: {}".format(*self.__get_call_info(), message)
        self.logger.warning(message, *args)
    def debug(self, message, *args):
        message = "[{}][{}][{}]: {}".format(*self.__get_call_info(), message)
        self.logger.debug(message, *args)
        