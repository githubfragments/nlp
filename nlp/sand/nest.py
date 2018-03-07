import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
    
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
    
def pad(seq, shape, p=None, dtype='int32', value=0.):
    n = len(seq)
    d = len(shape)
    x = (np.ones(shape) * value).astype(dtype)
    if d==1:
        if p and p[0] and p[0]=='pre':
            x[-len(seq):]==seq
        else:
            x[:len(seq)] = seq
        return x
    j = (0 if (p==None or p[0]==None or p[0]=='post') else shape[0]-n)
    for i,s in enumerate(seq):
        y = pad(s, shape[1:], p[1:], dtype=dtype, value=value)
        x[i+j,:] = y
    return x
        
def pad_sequences(seq, p=None, dtype='int32', value=0.):
    shape = maxlens(seq)
    return pad(seq, shape, p=p, dtype=dtype, value=value)
    

a = [1,2,3,4]
b = [[1,2],[3,4,5,6],[7,8,9]]
c = [[[1,2,3,4,5],[1,2,3,4,5,6,7]],[[1,2,3],[1,2],[1,2],[1,2]],[[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,9]]]

seq=c
p = (None,'pre',None)
