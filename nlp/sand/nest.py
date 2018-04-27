import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
    
def nested_lens(x,d=0):
    n = len(x)
    if d == None:
        ret = [n]
        dd = None
    else:
        ret = [(n,d)]
        dd = d+1
    if n>0 and nest.is_sequence(x[0]):
        ret.extend(map(lambda y: nested_lens(y,d=dd), x))
    return ret

def max_lens(x):
    v = nest.flatten(nested_lens(x))
    n = v[-1] + 1
    u = [[] for _ in range(n)]
    for i in range(len(v)/2):
        u[v[2*i+1]].append(v[2*i])
    #return tuple(map(max,u))
    return map(max,u)

def _seq_lens(x, axis=1):
    if axis==0:
        return x[0]
    return map(lambda y: _seq_lens(y, axis-1), x[1:])
    
def seq_lens(x, axis=1, p=None):
    shape = max_lens(x)
    u = _seq_lens(nested_lens(x, None), axis=axis)
    v = pad(u, shape=shape[0:axis], p=p)
    return v
    
def pad(seq, shape, p=None, dtype='int32', value=0.):
    n = len(seq)
    d = len(shape)
    x = (np.ones(shape) * value).astype(dtype)
    if p==None: p=tuple([None]*n)
    if d==1:
        if p[0] and p[0]=='pre':
            x[-len(seq):]= seq
        else:
            x[:len(seq)] = seq
        return x
    j = (0 if (p[0]==None or p[0]=='post') else shape[0]-n)
    for i,s in enumerate(seq):
        y = pad(s, shape[1:], p[1:], dtype=dtype, value=value)
        x[i+j,:] = y
    return x
        
def pad_sequences(seq, p=None, dtype='int32', value=0.):
    shape = max_lens(seq)
    seq_lengths = [seq_lens(seq, axis=i+1, p=p) for i in range(len(shape)-1)]
    return pad(seq, shape, p=p, dtype=dtype, value=value), tuple(seq_lengths)

def batch(inputs):
    batch_size = len(inputs)
    
    document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
    document_size = document_sizes.max()
    
    sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs]
    sentence_size = max(map(max, sentence_sizes_))
    sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
    
    b = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32) # == PAD
    for i, document in enumerate(inputs):
        for j, sentence in enumerate(document):
            sentence_sizes[i, j] = sentence_sizes_[i][j]
            for k, word in enumerate(sentence):
                b[i, j, k] = word
    return b, document_sizes, sentence_sizes
    
def slyce(x,i,j,pad='post'):
    if pad=='post':
        return x[i][:j]
    else:
        return x[i][-j:]
        
def tensor2list(b,sl,wl,p):
    z = []
    (n,m,o) = b.shape
    for i in range(n):
        zz=[]
        s = slyce(wl,i,sl[i],p[0])
        k = 0 if p[0]=='post' else m-sl[i]
        for j in range(len(s)):
            t = list(slyce(b[i],j+k,s[j],p[1]))
            zz.append(t)
        z.append(zz)
    return z
        
def list2list(x,f):
    if isinstance(x, (list,)):
        return map(lambda y: list2list(y,f), x)
    return f(x)
    
a = [1,2,3,4]
b = [[1,2],[3,4,5,6],[7,8,9]]
c = [[[1,2,3,4,5],[1,2,3,4,5,6,7]],[[1,2,3],[1,2],[1,2],[1,2]],[[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,9]]]

x=c
p=(None,'pre','post')

shape = max_lens(x)
(doc_size, sent_size, word_size) = shape
#b, sent_lens, word_lens = batch(x)
b, (sent_lens, word_lens) = pad_sequences(x,p)

word_level_lengths = np.reshape(word_lens, [doc_size * sent_size])

wl = word_lens
sl = sent_lens
z = tensor2list(b,sl,wl,p[1:])

zz = slyce(wl,0,sl[0],p[1])