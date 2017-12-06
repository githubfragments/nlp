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
from timeit import default_timer as timer
import tensorflow as tf

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
def tokenize(string):
    tokens = tokenizer.tokenize(string.lower())
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens

def read_col(file, col, sep="\t", header=None, type='int32'):
    df = pd.read_csv(file, sep=sep, header=header)#.sort_values(by=col)
    vals = df[col].values.astype(type)
    return vals
    
class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

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
    return seed

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