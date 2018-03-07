import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

def _test(x):
    n = len(x)
    ret = [n]
    if n>0 and nest.is_sequence(x[0]):
        ret.extend(map(_test, x))
    return ret
            
def test(x):
    if nest.is_sequence(x):
        return _test(x)
    return None
    
def _maxlens(x,d=0):
    n = len(x)
    ret = [(n,d)]
    if n>0 and nest.is_sequence(x[0]):
        ret.extend(map(lambda y: _maxlens(y,d+1), x))
    return nest.flatten(ret)

def maxlens(x):
    v = _maxlens(x)
    n = v[-1] + 1
    u = [[] for _ in range(n)]
    for i in range(len(v)/2):
        u[v[2*i+1]].append(v[2*i])
    return tuple(map(max,u))
        
    
a = [1,2,3,4]
b = [[1,2],[3,4,5,6],[7,8,9]]
c = [[[1,2,3,4,5],[1,2,3,4,5,6,7]],[[1,2,3],[1,2],[1,2],[1,2]],[[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,9]]]

aa = test(a)
bb = test(b)
cc = test(c)
