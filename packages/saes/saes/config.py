import importlib
import sys
from multiprocessing import cpu_count
from time import time
from typing import Dict
from functools import wraps
from types import FunctionType
import datetime as dt
from datetime import timedelta as td

from colorama import Back, Fore

FC = Fore.CYAN
FR = Fore.RESET
BG = Back.GREEN
BRED = Back.RED
BRES = Back.RESET

PRINT_DETAILS = True
FILE_OUTPUT = False
# NCPUS = cpu_count()
NCPUS = 24

KEY_LIMIT = 100
PT_LIMIT = 10


def timed(f):
    @wraps(f)
    def wrapped(self, *args, **kwargs):
        start = time()
        result = f(self, *args, **kwargs)
        self.t_log[f.__name__] = time() - start
        return result

    return wrapped


def time_log(self):
    return {
        i: td(seconds=td(seconds=v) // td(seconds=1))
        for i, v in self.t_log.items()
    }


class Timer(type):
    def __new__(cls, name, bases, attrs):
        new_attrs = {}
        for attr_name, attr in attrs.items():
            if isinstance(attr, FunctionType):
                attr = timed(attr)
            new_attrs[attr_name] = attr

        new_attrs['t_log'] = {}
        new_attrs['time_log'] = time_log

        return super().__new__(cls, name, bases, new_attrs)


def reload(module_name: str, *names: str):
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    else:
        __import__(module_name, fromlist=names)
    for name in names:
        globals()[name] = getattr(sys.modules[module_name], name)
