# fmt: off

import ctypes
import sys
from math import ceil, floor

import _ctypes
from numpy import array, float64, int64, random, zeros

from codered import __name__ as pkg_name
from codered import __path__ as pkg_path

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

def c_char_ptr(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_char))

def c_long_ptr(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_long))

def ham(v):
    return sum(v)



# CodeRedLib: a python wrapper for the c++ coreredlib.cpp

# coreredlib is compiled many time with various value of maxn
# make sure the value you want to use are listed in compile_cpp_core.sh         

# Functions names match with the paper. They all act on the internal state:
# self.B : the basis
# self.E : The epipodal matrix
# self.P : The cumulative projector matrix P[i] = &_{j<i} ~ B[j] (has length k+1) 
# (P[i] is the boolean complement of s_i from the paper)
# self.l : Epipodal length

class CodeRedLib(object):
    def __init__(self, B, seed=None):
        k, n = B.shape
        self.k, self.n = k, n

        if seed is None:
            seed = random.randint(0,2**63)

        nmax = 256 * int(ceil(n/256.))
        ns = [256, 384, 512, 768, 1024, 1280, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 24576, 32768, 49152, 65536]
        nmax = next(filter(lambda x: n <= x, sorted(ns)), None)
        if nmax is None:
            raise ValueError(f"Compile larger library for n={n}")
        self.lib = ctypes.cdll.LoadLibrary(pkg_path[0] + "/bin/coderedlib-%d.so"%nmax)
        
        self.lib._setup(k, n, c_char_ptr(B), ctypes.c_long(seed))

        self.B = zeros((k  , n), dtype='bool')
        self.E = zeros((k  , n), dtype='bool')
        self.P = zeros((k+1, n), dtype='bool')
        self.l = zeros( k, dtype='int64')
        self.update()

    def update(self):
        self.lib._export_all(c_char_ptr(self.B), 
                             c_char_ptr(self.E), 
                             c_char_ptr(self.P), 
                             c_long_ptr(self.l))
        # Check that the code is of full length
        assert(sum(self.l)==self.n)


    def LLL(self):
        self.lib._LLL()
        self.update()


    def Randomize(self, light=True):
        self.lib._Randomize(light)
        self.update()


    def Systematize(self):
        self.lib._Systematize()
        self.update()


    def EpiSort(self):
        self.lib._EpiSort()
        self.update()


    def SizeRedBasis(self):
        self.lib._SizeRedBasis()
        self.update()


    def SemiSystematize(self):
        self.lib._SemiSystematize()
        self.update()
        for k1 in range(self.k)[::-1]:
            if self.l[k1] > 1:
                return k1+1
        return 0

    def KillTwos(self):
        self.lib._KillTwos()
        self.update()


    # Used to speed up LB/LBB experiments in large dimension by only
    # vistining a (1+skip)^{1-w2} fraction of the enumerated space.
    def set_skip(self, skip):
        return self.lib._set_skip(int(floor(skip)))


    def SizeRed(self, t):
        return self.lib._SizeRed(c_char_ptr(t))


    def LB(self, w2, goal_w=None, t=None, stats=False):
        tt     = zeros(self.n, dtype='bool') if t is None else 1 * t
        _stats = zeros(self.n+1, dtype='int64') if stats else None

        success = self.lib._LB(c_char_ptr(tt), w2, 
                               0 if goal_w is None else goal_w,
                               c_long_ptr(_stats) if stats else None)

        if stats:
            return _stats
        if success or goal_w is None:
            return tt


    def LBB(self, k1, w2, goal_w=None, t=None, stats=False):
        tt     = zeros(self.n, dtype='bool') if t is None else 1 * t
        _stats = zeros(self.n+1, dtype='int64') if stats else None

        success = self.lib._LBB(c_char_ptr(tt), k1, w2,
                                0 if goal_w is None else goal_w,
                                c_long_ptr(_stats) if stats else None)

        if stats:
            return _stats
        if success or goal_w is None:
            return tt
