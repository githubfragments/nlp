from __future__ import print_function
from __future__ import division

import sys
import os, errno
import logging
import numpy as np
import pandas as pd
import random
import codecs
import collections
import itertools
import glob
import pickle
import time
import re
from timeit import default_timer as timer
import tensorflow as tf

def softmask(x, axis=-1, mask=None, T=None):
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    x = x - x_max
    
    if T!=None:
#         if not tf.is_numeric_tensor(T):
#             T = tf.get_variable('T', shape=[1], initializer=tf.constant_initializer(T), dtype=tf.float32, trainable=True)
        x = x/T
    
    ex = tf.exp(x)
    
    if mask!=None:
        ex = tf.multiply(ex, mask)
        
    es = tf.reduce_sum(ex, axis=axis, keep_dims=True)
    
#     if mask!=None:
#         ez = tf.cast(tf.reduce_sum(mask, axis=-1, keep_dims=True)==0, tf.float32)
#         es = es + ez
    ez = tf.cast(es==0, tf.float32)
    
    ret = ex/(es + ez)
    
    return ret, ex, es, ez

#######################################################################
## NESTED SEQUENCES ##

from tensorflow.python.util import nest

def nested_lens(x, d=0):
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
    for i in range(int(len(v)/2)):
        u[v[2*i+1]].append(v[2*i])
    #return tuple(map(max,u))
    return map(max,u)

def _seq_lens(x, axis=1):
    if axis==0: return x[0]
    return map(lambda y: _seq_lens(y, axis-1), x[1:])
    
def seq_lens(x, axis=1, p=None):
    shape = max_lens(x)
    u = _seq_lens(nested_lens(x, d=None), axis=axis)
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
        
def pad_sequences(seq, p=None, m=None, dtype='int32', value=0.):
    shape = max_lens(seq)
    ## necessary?? ##
    if m:
        ss = list(shape)
        for i in range(len(m)):
            if m[i]:
                ss[i] = m[i]
        shape = tuple(ss)
    #################
    seq_lengths = [seq_lens(seq, axis=i+1, p=p) for i in range(len(shape)-1)]
    return pad(seq, shape, p=p, dtype=dtype, value=value), tuple(seq_lengths)

def is_sequence(x):
    return nest.is_sequence(x)
#######################################################################
## NLP / TOKENIZATION ##

punc = '(),'# ?!
word_pattern = r'([0-9]+[0-9,.]*[0-9]+|[\w]+|[{}])'.format(punc)

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(word_pattern)

def tokenize_OLD(string):
    tokens = tokenizer.tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens

from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
sent_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
from sentence_tokenizer import split_into_sentences

def clean(s):
    s = re.sub("([0-9]+),([0-9]+)", "\\1COMMA\\2", s)
    for c in punc:
        a = '[{0}][\s{0}]*[{0}]*'.format(c)
        b = ' {0} '.format(c)
        s = re.sub('[{0}][\s{0}]*[{0}]*'.format(c), ' {0} '.format(c), s)
    s = re.sub("COMMA", ",", s)
    return s
    
def tokenize_NEW(string):
    string = re.sub("[.]\s*[.]\s*[.]"," . ", string)
    string = re.sub("([\w']+)-([\w']+)", "\\1 - \\2", string)
    string = re.sub("[+]","", string)
    
    sents = sent_tokenize(string)
    #sents2 = sent_tokenizer.tokenize(string)
    #sents3 = split_into_sentences(string)
    
    #words = [word_tokenize(clean(s)) for s in sents if len(s)>1]
    #tokens = [tokenizer.tokenize(clean(s)) + [u'.'] for s in sents if len(s)>1]
    tokens = [tokenizer.tokenize(clean(s)) for s in sents if len(s)>1]
    
    #return list(itertools.chain(*tokens))
    return tokens

def tokenize(string):
    #return tokenize_OLD(string.lower())
    return tokenize_NEW(string.lower())

def read_col(file, col, sep="\t", header=None, type='int32'):
    df = pd.read_csv(file, sep=sep, header=header)#.sort_values(by=col)
    vals = df[df.columns[col]].values.astype(type)
    return vals

def write_sequence(file, x, sep='\n'):
    with open(file, 'w') as output_file:
        for item in x:
            output_file.write(str(item) + sep)
    
class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

def ps(x, s):
    print('{} : {}'.format(s, x.shape))
    
def softmask(x, axis=-1, mask=None):
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    x = x - x_max
    ex = tf.exp(x)
    
    if mask!=None:
        ex = tf.multiply(ex, mask)
        
    es = tf.reduce_sum(ex, axis=axis, keep_dims=True)
    return ex/es

def shuffle_lists(*ls):
    l =list(zip(*ls))
    random.shuffle(l, random=rng)
    return zip(*l)

def shuffle_arrays(*xx):
    n = xx[0].shape[0]
    p = rng.permutation(n)#p = np.random.permutation(n)
    yy=()
    for x in xx:
        yy += (x[p],)
    return yy

def lindexsplit(x, idx):
    return [x[start:end] for start, end in zip(idx, idx[1:])]

    # For a little more brevity, here is the list comprehension of the following
    # statements:
    #    return [some_list[start:end] for start, end in zip(args, args[1:])]
#     my_list = []
#     for start, end in zip(args, args[1:]):
#         my_list.append(some_list[start:end])
#     return my_list

def memoize(f):
    """ Memoization decorator for a function taking one or more arguments. """
    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)
        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret
    return memodict().__getitem__

T={}
def tic(K=0):
    global T
    if isinstance(K,list):
        for k in K:
            tic(k)
    else:
        T[K]=timer()
def toc(K=0, reset=True):
    global T
    t=timer()
    tt=t-T[K]
    if reset:
        T[K]=t
    return tt

def dot(x,y):
    x = tf.transpose(x)
    #y = tf.transpose(y)
    return tf.matmul(x,y)

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
def kappa_loss(labels, predictions, scope=None):
    with ops.name_scope(scope, "kappa_loss", (predictions, labels)) as scope:
        predictions = math_ops.to_float(predictions)
        labels = math_ops.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        losses = math_ops.squared_difference(predictions, labels)
        u = math_ops.reduce_sum(losses)
        a = math_ops.reduce_mean(labels)
        b = math_ops.multiply(predictions, labels - a)
        v = math_ops.reduce_sum(b)
        k = u / (2*v + u)
        return k

def nkappa(t,x):
    t=np.array(t,np.float32)
    x=np.array(x,np.float32)
    u = 0.5 * np.sum(np.square(x - t))
    v = np.dot(np.transpose(x), t - np.mean(t))
    return v / (v + u)

def interleave(a,b):
    return list(itertools.chain.from_iterable(zip(a,b)))

def arrayfun(f,A):
    return list(map(f,A))

def isnum(a):
    try:
        float(repr(a))
        ans = True
    except:
        ans = False
    return ans

def get_seed():
    t = int( time.time() * 1000.0 )
    seed = ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >>  8) + ((t & 0x0000ff00) <<  8) + ((t & 0x000000ff) << 24)
    return seed % 2**30-1#    2**32-1

rng = 0
def seed_random(seed=None):
    global rng
    if seed==None or seed<=0:
        seed = get_seed()
    print(b_green('RAND_SEED == {}'.format(seed)))
    random.seed(seed)
    np.random.seed(seed=seed)
    rng = np.random.RandomState(seed)
    return seed

def string2rand(s):
    return abs(hash(s)) % (10 ** 8)



def get_hostname():
    import socket
    return socket.gethostname()

def get_loc(check='home', default='work'):
    if check in get_hostname():
        return check
    return default

def mkdirs(s):
    import errno
    try:
        os.makedirs(s)
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(s):
            pass

''' add root to path IF path not absolute'''
def try_abs(path, root):
    if os.path.isabs(path):
        return path
    return os.path.join(root,path)

def make_abs(path):
    return os.path.abspath(path)

# sort 2-D numpy array by col
def sortrows(x,col=0,asc=True):
    n=-1
    if asc:
        n=1
    x=x*n
    return n*x[x[:,col].argsort()]


def set_logger(out_dir=None):
    console_format = BColors.OKBLUE + '[%(levelname)s]' + BColors.ENDC + ' (%(name)s) %(message)s'
    #datefmt='%Y-%m-%d %Hh-%Mm-%Ss'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(console_format))
    logger.addHandler(console)
    if out_dir:
        file_format = '[%(levelname)s] (%(name)s) %(message)s'
        log_file = logging.FileHandler(out_dir + '/log.txt', mode='w')
        log_file.setLevel(logging.DEBUG)
        log_file.setFormatter(logging.Formatter(file_format))
        logger.addHandler(log_file)

import re

class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    WHITE = '\033[37m'
    YELLOW = '\033[33m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    RED = '\033[31m'
    MAGENTA = '\033[35m'
    BLACK = '\033[30m'
    BHEADER = BOLD + '\033[95m'
    BOKBLUE = BOLD + '\033[94m'
    BOKGREEN = BOLD + '\033[92m'
    BWARNING = BOLD + '\033[93m'
    BFAIL = BOLD + '\033[91m'
    BUNDERLINE = BOLD + '\033[4m'
    BWHITE = BOLD + '\033[37m'
    BYELLOW = BOLD + '\033[33m'
    BGREEN = BOLD + '\033[32m'
    BBLUE = BOLD + '\033[34m'
    BCYAN = BOLD + '\033[36m'
    BRED = BOLD + '\033[31m'
    BMAGENTA = BOLD + '\033[35m'
    BBLACK = BOLD + '\033[30m'
    
    @staticmethod
    def cleared(s):
        return re.sub("\033\[[0-9][0-9]?m", "", s)

def red(message):
    return BColors.RED + str(message) + BColors.ENDC

def b_red(message):
    return BColors.BRED + str(message) + BColors.ENDC

def blue(message):
    return BColors.BLUE + str(message) + BColors.ENDC

def b_yellow(message):
    return BColors.BYELLOW + str(message) + BColors.ENDC

def green(message):
    return BColors.GREEN + str(message) + BColors.ENDC

def b_green(message):
    return BColors.BGREEN + str(message) + BColors.ENDC