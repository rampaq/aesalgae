import typing as t
from collections import defaultdict
from functools import wraps
from pprint import PrettyPrinter
from time import time

from sage.all import Integer, Rational
from termcolor import cprint

pp = PrettyPrinter(compact=True)

V = t.TypeVar("V")
P = t.ParamSpec("P")
R = t.TypeVar("R")


class Logger:
    name: str
    _log: dict[str, list[t.Any]]
    _timings: dict[str, dict[str, float]]
    _data: dict[str, dict[str, t.Any]]
    _names_order: list

    def __init__(self, name=None, log=None, timings=None, data=None, names_order=None):
        """
        Logger for strings and timings which can be subdivided into children loggers via `sublogger`.
        Warning: do not create sublogger in parallel code
        """

        self._names_order = names_order
        if names_order is None:
            self._names_order = []

        if name is None:
            self.name = ""
        else:
            self.name = name
            self._names_order.append(name)

        self._log = log
        if log is None:
            self._log = defaultdict(list)

        self._timings = timings
        if timings is None:
            self._timings = defaultdict(dict)

        self._data = data
        if data is None:
            self._data = defaultdict(dict)

    def sublogger(self, name: str) -> "Logger":
        """Create named logger. When enabled=False, the logger wont log anything."""
        assert "." not in name
        new_name = f"{self.name or ''}.{name}"
        return Logger(
            name=new_name,
            log=self._log,
            timings=self._timings,
            data=self._data,
            names_order=self._names_order,
        )

    # def timefn(self, fn: t.Callable[..., V], *args, **kwargs) -> V:
    #    """Save running time and return results of function call"""
    #    return self.timefn_id(fn.__name__, fn, *args, **kwargs)

    def timefn(
        self, fn: t.Callable[P, R], name: str | None = None, sublog: bool = False
    ) -> t.Callable[P, R]:
        """
        sublog: whether to pass 'log' keyword arg to function with sublogger
        """
        if name is None:
            name = fn.__name__

        @wraps(fn)
        def inner(*args: P.args, **kwargs: P.kwargs):
            time0 = time()
            if sublog and "log" not in kwargs:
                kwargs["log"] = self.sublogger(name)
            out = fn(*args, **kwargs)
            time1 = time()
            dtime = time1 - time0
            self._timings[self.name][name] = dtime
            if dtime > 1:
                self._print_time(name, dtime)

            return out

        return inner

    def timefn_log(
        self, fn: t.Callable[P, R], name: str | None = None, sublog: bool = False
    ) -> t.Callable[P, R]:
        return self.timefn(fn, name, sublog=True)

    # def timefn_id(self, name: str, fn: t.Callable[..., V], *args, **kwargs) -> V:
    #    """Save running time and return results of function call, supply own name for the saved result"""
    #    time0 = time()
    #    out = fn(*args, **kwargs)
    #    time1 = time()
    #    self._timings[self.name][name] = time1 - time0
    #    return out

    def time(self, name: str, dtime: float):
        """Log manually timed function"""
        self._timings[self.name][name] = dtime
        if dtime > 1:
            self._print_time(name, dtime)

    def _print_time(self, name, dtime):
        cprint(f"{self.name}::{name} - time: {dtime:.2f}", "green")

    def data(self, name: str, data: t.Any, show: bool = True, pprint: bool = False):
        """Log manual data"""
        if isinstance(data, Integer):
            data = int(data)
        elif isinstance(data, Rational):
            data = float(data)

        self._data[self.name][name] = data
        if show:
            if pprint:
                print(f"{self.name}::{name} - data: --- 👇")
                pp.pprint(data)
            else:
                print(f"{self.name}::{name} - data: {data}")

    def log(self, *args, show=True, **kwargs):
        """Log message"""
        if len(kwargs) == 0 and len(args) == 1:
            self._log[self.name].append(args[0])
            if show:
                print(f"{self.name} - log: {args[0]}")
        else:
            self._log[self.name].append((args, kwargs))
            if show:
                print(f"{self.name} - log: {args, kwargs}")

    def get_logs(self) -> dict:
        """Return log and timings of all loggers below and equal to my level"""
        out = {}
        log = {
            key: content
            for key, content in self._log.items()
            if key.startswith(self.name)
        }
        if log:
            out["log"] = log

        timing = {
            key: content
            for key, content in self._timings.items()
            if key.startswith(self.name)
        }
        if timing:
            out["time"] = timing

        data = {
            key: content
            for key, content in self._data.items()
            if key.startswith(self.name)
        }
        if data:
            out["data"] = data

        return out


class NoLogger_base(Logger):
    def __init__(self, *args, **kwargs):
        pass

    def sublogger(self, *args, **kwargs):
        return self

    def timefn(self, fn, *args, **kwargs):
        return fn

    def timefn_log(self, fn, *args, **kwargs):
        return fn

    def timefn_id(self, name, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def time(self, *args, **kwargs):
        pass

    def data(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def get_logs(self):
        return {}


nolog = NoLogger_base()


class Experiment:
    experiment_name: str
    log: Logger

    def __init__(self, name: str, logger: Logger | None = None):
        self.experiment_name = name
        if logger is None:
            self.log = Logger(name)
        else:
            self.log = logger
