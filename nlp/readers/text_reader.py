
from __future__ import print_function
from __future__ import division

import os
import codecs
import collections
import numpy as np
import pandas as pd
import random
import glob
import re
import pickle
import time
import itertools
from tensorflow.python.util import nest

from vocab import Vocab
from nlp.util.utils import adict, get_seed, arrayfun, adict
from nlp.util import utils as U

#REGEX_NUM = r'^[0-9]*\t[0-9]\t[0-9]\t[0-9]\t(?!\s*$).+'
#REGEX_MODE = r'^[0-9]*\tm\tm\tm\t(?!\s*$).+'
REGEX_NUM = r'^[0-9]*\t([0-9]\t)+(?!\s*$).+'
REGEX_MODE = r'^[0-9]*\t(m\t)+(?!\s*$).+'

## for (possibly) nested lists
def isListEmpty(inList):
    if isinstance(inList, list): # Is a list
        return all( map(isListEmpty, inList) )
    return False # Not a list

'''
READ FILE IN CHUNKS
- reads chunk by chunk
- yields line by line
- filter by regex
- shuffles each chunk
'''
class ChunkReader(object):
    def __init__(self, file_name, chunk_size=1000, shuf=True, regex=None, seed=None, bucket=None, verbose=True):
        self.file_name = file_name
        self.chunk_size = chunk_size
        if chunk_size==None or chunk_size<=0: # read entire file as one chunk
            self.chunk_size=np.iinfo(np.int32).max
        self.shuf = shuf
        self.bucket = bucket
        self.verbose = verbose
        self.regex = regex
        if regex:
            self.regex = re.compile(regex)
        if seed==None or seed<=0:
            self.seed = U.get_seed()
        else:
            self.seed = seed
        self.rng = np.random.RandomState(self.seed + U.string2rand('ChunkReader'))
    
    def next_chunk(self, file_stream):
        lines = []
        try:
            for i in xrange(self.chunk_size):
                lines.append(next(file_stream).strip())
        except StopIteration:
            self.eof = True
            
        if self.regex:
            lines = filter(self.regex.search, lines)
            
        if self.bucket!=None:
            lens = [len(line) for line in lines]
            m = float(max(lens))
            lens = [float(ell)/m for ell in lens]
            lens = [ell + self.bucket * self.rng.rand() for ell in lens]
            pp = np.argsort(lens)
            lines = [lines[p] for p in pp]
            
        elif self.shuf:
            self.rng.shuffle(lines)
            
        return lines
        
    def chunk_stream(self, stop=True):
        while True:
            self.eof = False
            if self.verbose:
                print('READING... ', self.file_name)
            with codecs.open(self.file_name, "r", "utf-8") as f:
                while not self.eof:
                    yield self.next_chunk(f)
            if stop:
                break
    
    def line_stream(self, stop=True):
        for chunk in self.chunk_stream(stop=stop):
            for line in chunk:
                yield line
                
    def sample(self, sample_every=100, stop=True):
        i=0
        for line in self.line_stream(stop=stop):
            i=i+1
            if i % 100 == 0:
                print('{} | {}'.format(i,line))

class GlobReader(object):
    def __init__(self, file_pattern, chunk_size=1000, shuf=True, regex=None, seed=None, bucket=None, verbose=True):
        self.file_pattern = file_pattern
        self.file_names = glob.glob(self.file_pattern)
        self.file_names.sort()
        self.chunk_size = chunk_size
        self.shuf = shuf
        self.bucket = bucket
        self.verbose = verbose
        self.regex = regex
        if seed==None or seed<=0:
            self.seed = get_seed()
        else:
            self.seed = seed
        self.rng = np.random.RandomState(self.seed + U.string2rand('GlobReader'))
        self.num_files = None
        self.bpf = None
        self.prev_file = ''
        
    def new_file(self):
        new = (self.cur_file != self.prev_file) and (len(self.prev_file)>0)
        self.prev_file = self.cur_file
        return new
        
    def file_stream(self):
        if self.shuf:
            self.rng.shuffle(self.file_names)
        if self.num_files is None:
            self.num_files = len(self.file_names)
        for file_name in self.file_names:
            yield file_name
    
    ''' reads files in sequence (NOT parallel) '''
    def chunk_stream(self, stop=True):
        while True:
            for file in self.file_stream():
                self.cur_file = file
                chunk_reader =  ChunkReader(file, 
                                            chunk_size=self.chunk_size, 
                                            shuf=self.shuf,
                                            bucket=self.bucket,
                                            verbose=self.verbose,
                                            regex=self.regex, 
                                            seed=self.seed)
                for chunk in chunk_reader.chunk_stream(stop=True):
                    yield chunk
            if stop:
                break
    
    def line_stream(self, stop=True):
        for chunk in self.chunk_stream(stop=stop):
            for line in chunk:
                yield line

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
def is_number(token):
    return bool(num_regex.match(token))

'''
returns dictionary: words->[word indices]
                    chars->[char indices]
set words=None, chars=None if not desired
'''

#punc = set(['-','--',',',':','.','...','\'','(',')','&','#','$'])
#punc = set(['-','--',':','...','\'','(',')','&','#','$'])
punc = set(['-','--',':','...','\'','&','#','$'])

class TextParser(object):
    def __init__(self, word_vocab=None, char_vocab=None, max_word_length=None, reader=None, words='w', chars='c', eos='+', sep=' ', tokenize=True, keep_unk=True):
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.max_word_length = max_word_length
        self.reader = reader
        self.words = words
        self.chars = chars
        self.ws = 'ws'
        self.cs = 'cs'
        self.eos = eos
        self.sep = sep
        self._tokenize = tokenize
        self.keep_unk = keep_unk
        if tokenize:
            self._tokenize = U.tokenize
        
    def tokenize(self, str):
        if self._tokenize:
            return self._tokenize(str)
        return str.split()
    
    ''' parses line into word/char tokens, based on vocab(s) '''
    def _parse_line(self, line, word_tokens, char_tokens):
        line = Vocab.clean_line(line)
        
        sentences = self.tokenize(line)
        #toks = list(itertools.chain(*sentences))
        
        ws, cs = [0],[0]
        
        for toks in sentences:
            for word in toks:
                word = Vocab.clean(word, self.max_word_length, lower=(self.char_vocab==None))
                
                if self.char_vocab is None:
                    if word in punc:
                        continue
    #                 if is_number(word):
    #                     word='1'
                
                if self.word_vocab:
                    word_idx = self.word_vocab.get(word)
                    if word_idx>self.word_vocab.unk_index or self.keep_unk:
                        word_tokens.append(word_idx)
                
                if self.char_vocab:
                    char_array = Vocab.get_char_aray(word, self.char_vocab, self.word_vocab)
                    char_tokens.append(char_array)
                    
            if self.eos:
                if self.word_vocab: word_tokens.append(self.word_vocab.get(self.eos))
                if self.char_vocab: char_tokens.append(self.char_vocab.get_tok_array(self.eos))
            
            ws.append(len(word_tokens))
            cs.append(len(char_tokens))
        
        #ww = U.lindexsplit(word_tokens, ws)
        #cc = U.lindexsplit(char_tokens, cs)
            
        return word_tokens, char_tokens, ws, cs
    
    def parse_line(self, line):
        return self.package(*self._parse_line(line, word_tokens=[], char_tokens=[]))
    
    def package(self, word_tokens, char_tokens, ws, cs):
        return adict( { self.words:word_tokens , self.chars:char_tokens, self.ws:ws , self.cs:cs } )
    
    def chunk_stream(self, reader=None, stop=True):
        if reader==None:
            reader=self.reader
        if reader!=None:
            for chunk in reader.chunk_stream(stop=stop):
                word_tokens, char_tokens = [], []
                for line in chunk:
                    self._parse_line(line, word_tokens, char_tokens)
                yield self.package(word_tokens, char_tokens)
    
    def line_stream(self, reader=None, stop=True):
        if reader==None:
            reader=self.reader
        if reader!=None:
            for line in reader.line_stream(stop=stop):
                yield self.parse_line(line)
    
    def new_file(self):
        return self.reader.new_file()
    
    @property
    def num_files(self):
        return self.reader.num_files
             
    def sample(self, sample_every=100, reader=None, stop=True):
        i=0
        for d in self.line_stream(reader=reader, stop=stop):
            i=i+1
            if i % 100 == 0:
                print('{} | {}'.format(i, d.w))

def sample_table(c, min_cut=0.5, r=0.95):
    n = float(c.sum())
    t = float(c.min())/c
    while sum(t*c)/n < min_cut:
        t=1.-r*(1.-t)
    return t

def sample_dict(v, c, min_cut=0.5, r=0.95):
    t = sample_table(c, min_cut=min_cut, r=r)
    return dict(zip(v, t))
    
class FieldParser(object):
    def __init__(self, fields, reader=None, sep='\t', seed=None):
        self.fields = fields
        self.reader = reader# GlobReader!!
        self.sep = sep
        self.seed = seed
        if seed==None or seed<=0:
            self.seed = U.get_seed()
        else:
            self.seed = seed
        self.rng = np.random.RandomState(self.seed + U.string2rand('FieldParser'))
    
    def parse_line(self, line, t=None):
        rec = line.strip().split(self.sep)
        d = {}
        for k,v in self.fields.items():
            if isinstance(v, basestring):
                d[v] = rec[k].strip()
            else:
                d.update(v.parse_line(rec[k].strip()))
            if t and v=='y':
                y = float(d[v])
                p = t[y]
                if self.rng.rand()>p:
                    #print('sample NO\t[{},{}]'.format(y,p))
                    return None
                #print('sample YES\t[{},{}]'.format(y,p))
        return adict(d)
    
    def line_stream(self, reader=None, stop=True, t=None):
        if reader==None:
            reader=self.reader
        if reader!=None:
            for line in self.reader.line_stream(stop=stop):
                d = self.parse_line(line, t=t)
                if d: 
                    yield d
    
    def get_maxlen(self):
        n = 0
        for d in self.line_stream(stop=True):
            n = max(n,len(d.w))
        return n
    
    def get_all_fields(self, key):
        x = []
        for d in self.line_stream(stop=True):
            x.append(d[key])
        return x
    
    def get_ystats(self):
        y = self.get_all_fields(key='y')
        y = np.array(y,dtype=np.float32)
        v, c = np.unique(y, return_counts=True)
        d = {}
        d['mean'] = np.mean(y)
        d['std'] = np.std(y)
        d['min'] = np.min(y)
        d['max'] = np.max(y)
        d['n'] = len(y)
        d['v'] = v
        d['c'] = c
        return adict(d)
                       
    def sample(self, sample_every=100, reader=None, stop=True):
        i=0
        for d in self.line_stream(reader=reader, stop=stop):
            i=i+1
            if i % 100 == 0:
                print('{} | {}\t{}\t{}'.format(i, d.id, d.y, d.w))
                #print('{} | {}'.format(i, d.w))
        print('{} LINES'.format(i))
                
## reader=TextParser
class TextBatcher(object):
    def __init__(self, reader, batch_size, num_unroll_steps, batch_chunk=100, trim_chars=False):
        self.reader = reader
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.max_word_length = reader.max_word_length# reader=TextParser
        self.batch_chunk = batch_chunk
        if batch_chunk==None or batch_chunk<=0:
            self.batch_chunk=np.iinfo(np.int32).max
        self.trim_chars = trim_chars
        self.wpb = self.batch_size * self.num_unroll_steps
    
    @property
    def bpf(self):
        return 3057
    @property
    def bps(self):
        return self.bpf
    
    def new_file(self):
        return self.reader.new_file()
    def new_shard(self):
        return self.new_file()
    
    @property
    def num_files(self):
        return self.reader.num_files
    
    def length(self):
        if not self.num_files is None and not self.bpf is None:
            return self.num_files*self.bpf
        return None
    
    def make_batches(self, tok_stream):
        word_toks, char_toks, N = [], [], 0
        for d in tok_stream:
            word_toks.extend(d.w)
            char_toks.extend(d.c)
            N = N + len(d.w)
            if N > self.batch_chunk * self.wpb:
                break
        
        word_tensor = np.array(word_toks, dtype=np.int32)
        char_tensor = np.zeros([len(char_toks), self.max_word_length], dtype=np.int32)
        for i, char_array in enumerate(char_toks):
            char_tensor [i,:len(char_array)] = char_array
        
        length = word_tensor.shape[0]
        assert char_tensor.shape[0] == length
        
        # round down length to whole number of slices
        reduced_length = (length // (self.batch_size * self.num_unroll_steps)) * self.batch_size * self.num_unroll_steps
        if reduced_length==0:
            return None
        
        word_tensor = word_tensor[:reduced_length]
        char_tensor = char_tensor[:reduced_length, :]
        
        ydata = np.zeros_like(word_tensor)
        ydata[:-1] = word_tensor[1:].copy()
        ydata[-1] = word_tensor[0].copy()

        x_batches = char_tensor.reshape([self.batch_size, -1, self.num_unroll_steps, self.max_word_length])
        y_batches = ydata.reshape([self.batch_size, -1, self.num_unroll_steps])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))
        
        return list(x_batches), list(y_batches)
    
    ## trims zero-padding off 3rd (last) dimension (characters)
    def trim_batch(self, x):
        s = np.sum(np.sum(x,axis=1), axis=0)
        i = np.nonzero(s)[0][-1]+1
        return x[:,:,:i]
    
    ## x: char indices
    ## y: word indices
    def batch_stream(self, stop=False):
        tok_stream = self.reader.chunk_stream(stop=stop)
        
        while True:
            batches = self.make_batches(tok_stream)
            if batches is None:
                break
            for c, w in zip(batches[0], batches[1]):
                if self.trim_chars:
                    c = self.trim_batch(c)
                yield adict( { 'w':w , 'c':c } )

def nest_depth(x):
    #depth = lambda L: isinstance(L, list) and max(map(depth, L))+1
    #return depth(x)
    depths = []
    for item in x:
        if isinstance(item, list):
            depths.append(nest_depth(item))
    if len(depths) > 0:
        return 1 + max(depths)
    return 1

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
    for i in range(int(len(v)/2)):
        u[v[2*i+1]].append(v[2*i])
    return map(max,u) 

def pad_sequences_better(sequences, m=None, p=None, dtype='int32', value=0.):
    sample_shape = maxlens(sequences)
    if m:
        ss = list(sample_shape)
        for i in range(len(m)):
            if m[i]:
                ss[i] = m[i]
        sample_shape = tuple(ss)
        
    x = (np.ones(sample_shape) * value).astype(dtype)
    
    for i,s in enumerate(sequences):
        if d > 2:# <-- indicates char sequence
            y = (np.ones((max_text_length,) + sample_shape) * value).astype(dtype)
            k = (0 if wpad=='post' else max_text_length-len(s))
            for j,t in enumerate(s):
                if j>= max_text_length:
                    break
                if cpad == 'post':
                    y[j+k,:len(t)] = t
                else:
                    y[j+k,-len(t):] = t
            x[i,:] = y
        else:# <-- otherwise word sequence
            s = s[:max_text_length]
            if wpad == 'post':
                x[i,:len(s)] = s
            else:
                x[i,-len(s):] = s
    
#     if d<3 and wpad!='post':
#         seq_lengths = [max_text_length for i in seq_lengths]
        
    return x, seq_lengths
        
def pad_sequences(sequences, trim_words=False, max_text_length=None, max_word_length=None, dtype='int32', wpad='post', cpad='post', value=0.):
    num_samples = len(sequences)
    seq_lengths = map(len, sequences)
    if trim_words:
        max_text_length = max(seq_lengths)
        #max_text_length+=100
    
    sample_shape = tuple()
    d = nest_depth(sequences)
    if d > 2:# <-- indicates char sequence
        if max_word_length is None:
            max_word_length = max(map(lambda x:max(map(len,x)), sequences))
        sample_shape = (max_word_length,)
        
    x = (np.ones((num_samples, max_text_length) + sample_shape) * value).astype(dtype)
    
    for i,s in enumerate(sequences):
        if d > 2:# <-- indicates char sequence
            y = (np.ones((max_text_length,) + sample_shape) * value).astype(dtype)
            k = (0 if wpad=='post' else max_text_length-len(s))
            for j,t in enumerate(s):
                if j>= max_text_length:
                    break
                if cpad == 'post':
                    y[j+k,:len(t)] = t
                else:
                    y[j+k,-len(t):] = t
            x[i,:] = y
        else:# <-- otherwise word sequence
            s = s[:max_text_length]
            if wpad == 'post':
                x[i,:len(s)] = s
            else:
                x[i,-len(s):] = s
    
#     if d<3 and wpad!='post':
#         seq_lengths = [max_text_length for i in seq_lengths]
        
    return x, seq_lengths

            
## reader=FieldParser
class EssayBatcher(object):
    def __init__(self, reader, batch_size, max_text_length=None, max_word_length=None, trim_words=False, trim_chars=False, ystats=None, verbose=True):
        self.reader = reader
        self.batch_size = batch_size
        self.trim_words = trim_words
        self.trim_chars = trim_chars
        self.max_text_length = max_text_length
        self.max_word_length = max_word_length
        if trim_words:
            self.max_text_length = None
        elif max_text_length==None:
            self.max_text_length = self.reader.get_maxlen()# reader=FieldParser
            print('max essay length: {}'.format(self.max_text_length))
        if trim_chars:
            self.max_word_length = None
        if ystats is None and reader!=None:
            ystats = reader.get_ystats()# for ATS: reader=field_parser
            if verbose:
                print('\nYSTATS (mean,std,min,max,#): {}\n'.format(ystats))
        self.ystats = ystats
    
    ## to interval [0,1]
    def normalize(self, y, min=None, max=None):
        if min==None:
            min = self.ystats.min
        if max==None:
            max = self.ystats.max
        y = (y-min) / (max-min)# --> [0,1]
        #y = y - (self.ystats.mean-self.ystats.min)/(self.ystats.max-self.ystats.min)
        return y
        #return (y - self.ystats.mean) / (self.ystats.max-self.ystats.min)
        
    @property
    def ymean(self):
        return self.normalize(self.ystats.mean)
    
    def word_count(self, reset=True):
        wc = self._word_count
        if reset:
            self._word_count = 0
        return wc
    
    def batch(self, 
              ids=None, 
              labels=None, 
              words=None, 
              chars=None, 
              w=None, 
              c=None, 
              trim_words=None,
              trim_chars=None,
              wpad='pre',
              cpad='post',
              ):
        if ids:
            self.last = (ids, labels, words, chars, w, c)
        else:
            (ids, labels, words, chars, w, c) = self.last
            
        if trim_words==None:
            trim_words=self.trim_words
        if not trim_words and self.max_text_length==None:
            self.max_text_length = self.reader.get_maxlen()
        if trim_chars==None:
            trim_chars=self.trim_chars
            
        n = len(ids)
        b = { 'n' : n }
        
        # if not full batch.....just copy first item to fill
        for i in range(self.batch_size-n):
            ids.append(ids[0])
            labels.append(labels[0])
            words.append(words[0])
            chars.append(chars[0])
        
        b['id'] = ids                              # <-- THIS key ('id') SHOULD COME FROM FIELD_PARSER.fields
                
        y = np.array(labels, dtype=np.float32)
        y = self.normalize(y)
        y = y[...,None]#y = np.expand_dims(y, 1)
        b['y'] = y                                  # <-- THIS key ('y') SHOULD COME FROM FIELD_PARSER.fields
        
        if w and not isListEmpty(words):
            word_tensor, seq_lengths = pad_sequences(words, trim_words=trim_words, max_text_length=self.max_text_length, wpad=wpad)
            b['w'] = word_tensor
            b['x'] = b['w']
        
        if c and not isListEmpty(chars):
            char_tensor, seq_lengths = pad_sequences(chars, trim_words=trim_words, max_text_length=self.max_text_length, max_word_length=self.max_word_length, wpad=wpad)
            b['c'] = char_tensor
            b['x'] = b['c']
        
        b['s'] = seq_lengths    
        
        return adict(b)
    
    '''
    use batch padding!
    https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
    '''
    def batch_stream(self, 
                     stop=False, 
                     skip_ids=None, 
                     hit_ids=None, 
                     w=True, 
                     c=True, 
                     min_cut=1.0, 
                     partial=False, 
                     trim_words=None, 
                     wpad='pre', 
                     ):
        t=None
        if min_cut<1.0:
            t = sample_dict(self.ystats.v, self.ystats.c, min_cut=min_cut)
            
        i, ids, labels, words, chars = 0,[],[],[],[]
        self._word_count = 0
        for d in self.reader.line_stream(stop=stop, t=t):# reader=FieldParser!
            if skip_ids:
                if d.id in skip_ids:
                    continue
            if hit_ids:
                if d.id not in hit_ids:
                    continue
            if len(d.w)==0:
                continue
            
            ids.append(d.id)
            labels.append(d.y)
            self._word_count+=len(d.w)
            
            words.append(d.w)
            chars.append(d.c)
            
            i=i+1
            if i == self.batch_size:
                yield self.batch(ids, labels, words, chars, w, c, trim_words, wpad=wpad)
                i, ids, labels, words, chars = 0,[],[],[],[]
        
        if i>0 and partial:
            yield self.batch(ids, labels, words, chars, w, c, trim_words, wpad=wpad)
            
            
    def batch_stream2(self, x, y, w=True, c=False, partial=True, trim_words=None):
        ## random shuffle entire set
        x,y = U.shuffle_arrays(x,y)
        
        ##
        labels, words, j = [],[],0
        self._word_count = 0
        for i in range(len(y)):
            j+=1
            labels.append(y[i])
            words.append(x[i,:])
            if j == self.batch_size:
                yield self.batch2(labels, words)
                labels, words, j = [],[],0
        
        if j>0 and partial:
            yield self.batch2(labels, words)

    def batch2(self, labels=None, words=None, trim_words=None, shuf=False):
        n = len(labels)
        b = { 'n' : n }
        
        # shuffle
        if shuf:
            p = np.random.permutation(n)
            q = [i+n for i in range(self.batch_size-n)]
            p = np.hstack([p,q]).astype(np.int32)
        
        # if not full batch.....just copy first item to fill
        for i in range(self.batch_size-n):
            labels.append(labels[0])
            words.append(words[0])
        
        ############################################################
        y = np.array(labels, dtype=np.float32)
        if shuf: y = y[p]
        y = y[...,None]#y = np.expand_dims(y, 1)
        b['y'] = y
        
        ############################################################
        word_tensor = np.array(words, dtype=np.float32)
        if shuf: word_tensor = word_tensor[p]
        
        ####################
        ## TRIM KERAS SEQUENCES to LONGEST in BATCH!!!
        nz = first_nonzero(word_tensor)
        mz = min(nz)
        #mz = max(0, mz-100)
        
        ##1
        word_tensor = word_tensor[:,mz:]
        
#         ##2
#         m = word_tensor.shape[1]
#         seq_lengths = [m-z for z in nz]
#         max_text_length = m-mz
#         x = np.zeros((self.batch_size, max_text_length)).astype(np.float32)
#         for i in range(self.batch_size):
#             s = word_tensor[i,nz[i]:]
#             x[i,:len(s)] = s
#         word_tensor = x
        
#         ##3
#         word_tensor = np.fliplr(word_tensor); seq_lengths = [m-z for z in nz]
        
        ####################
        b['w'] = word_tensor
        b['x'] = b['w']
        
        ############################################################
        max_seq_length = word_tensor.shape[1]
        seq_lengths = [max_seq_length for x in labels]
        b['s'] = seq_lengths
        
        ####################
        return adict(b)
    
## reader=FieldParser
class ResponseBatcher(object):
    def __init__(self, reader, batch_size, max_text_length=None, max_word_length=None, trim_words=False, trim_chars=False, ystats=None, verbose=True, normy=True):
        self.reader = reader
        self.batch_size = batch_size
        self.trim_words = trim_words
        self.trim_chars = trim_chars
        self.max_text_length = max_text_length
        self.max_word_length = max_word_length
        if trim_words:
            self.max_text_length = None
        elif max_text_length==None:
            self.max_text_length = self.reader.get_maxlen()# reader=FieldParser
            print('max essay length: {}'.format(self.max_text_length))
        if trim_chars:
            self.max_word_length = None
        if ystats is None and reader!=None:
            ystats = reader.get_ystats()# for ATS: reader=field_parser
            if verbose:
                print('\nYSTATS (mean,std,min,max,#): {}\n'.format(ystats))
        self.ystats = ystats
        self.normy = normy
    
    ## to interval [0,1]
    def normalize(self, y, min=None, max=None):
        if min==None:
            min = self.ystats.min
        if max==None:
            max = self.ystats.max
        y = (y-min) / (max-min)# --> [0,1]
        #y = y - (self.ystats.mean-self.ystats.min)/(self.ystats.max-self.ystats.min)
        return y
        #return (y - self.ystats.mean) / (self.ystats.max-self.ystats.min)
        
    @property
    def ymean(self):
        return self.normalize(self.ystats.mean)
    
    def word_count(self, reset=True):
        wc = self._word_count
        if reset:
            self._word_count = 0
        return wc

    @staticmethod
    def pad(seq, shape, p, dtype='int32', value=0.):
        n = len(seq)
        d = len(shape)
        x = (np.ones(shape) * value).astype(dtype)
        
        if d==1:
            if p[0] and p[0]=='pre':
                x[-len(seq):]==seq
            else:
                x[:len(seq)] = seq
            return x
        
        j = (0 if (p[0]==None or p[0]=='post') else shape[0]-n)
        for i,s in enumerate(seq):
            y = ResponseBatcher.pad(s, shape[1:], p[1:], dtype=dtype, value=value)
            x[i+j,:] = y
        
        return x

    @staticmethod
    def pad_sequences(seq, m=None, p=None, dtype='int32', value=0.):
        seq_lengths = map(len, seq)
        shape = maxlens(seq)
        n = len(shape)
        if m:
            ss = list(shape)
            for i in range(len(m)):
                if m[i]:
                    ss[i] = m[i]
            shape = tuple(ss)
        if not p:
            p = tuple([None]*n)
            
        x = ResponseBatcher.pad(seq, shape, p, dtype=dtype, value=value)
        return x, seq_lengths
    
    def batch(self, 
              ids=None, 
              labels=None, 
              words=None, 
              chars=None, 
              w=None, 
              c=None, 
              trim_words=None,
              trim_chars=None,
              spad='pre', 
              wpad='pre',
              cpad='post',
              split_sentences=False,
              ):
        if ids:
            self.last = (ids, labels, words, chars, w, c)
        else:
            (ids, labels, words, chars, w, c) = self.last
            
        if trim_words==None:
            trim_words=self.trim_words
        if not trim_words and self.max_text_length==None:
            self.max_text_length = self.reader.get_maxlen()
        if trim_chars==None:
            trim_chars=self.trim_chars
            
        n = len(ids)
        b = { 'n' : n }
        
        # if not full batch.....just copy first item to fill
        for i in range(self.batch_size-n):
            ids.append(ids[0])
            labels.append(labels[0])
            words.append(words[0])
            chars.append(chars[0])
        
        b['id'] = ids                              # <-- THIS key ('id') SHOULD COME FROM FIELD_PARSER.fields
                
        y = np.array(labels, dtype=np.float32)
        if self.normy:
            y = self.normalize(y)
        y = y[...,None]#y = np.expand_dims(y, 1)
        b['y'] = y                                  # <-- THIS key ('y') SHOULD COME FROM FIELD_PARSER.fields
        
        if w and not isListEmpty(words):
            m = (self.max_text_length,)
            if trim_words: m = (None,)
            if split_sentences: m = (None,) + m
            m = (None,) + m
            
            p = (wpad,)
            if split_sentences: p = (spad,) + p
            p = (None,) + p
            
            word_tensor, seq_lengths= self.pad_sequences(words, m=m, p=p)
            b['w'] = word_tensor
            b['x'] = b['w']
        
        if c and not isListEmpty(chars):
            m = (self.max_word_length,)
            if trim_chars: m = (None,)
            if trim_words: m = (None,) + m
            else: m = (self.max_text_length,) + m
            if split_sentences: m = (None,) + m
            m = (None,) + m
            
            p = (wpad, cpad)
            if split_sentences: p = (spad,) + p
            p = (None,) + p
            
            char_tensor, seq_lengths = self.pad_sequences(chars, m=m, p=p)
            b['c'] = char_tensor
            b['x'] = b['c']
        
        b['s'] = seq_lengths
        
        return adict(b)
    
    '''
    use batch padding!
    https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
    '''
    def batch_stream(self, 
                     stop=False, 
                     skip_ids=None, 
                     hit_ids=None, 
                     w=True, 
                     c=True, 
                     min_cut=1.0, 
                     partial=False, 
                     trim_words=None,
                     spad='pre',
                     wpad='pre',
                     cpad='post',
                     split_sentences=False,
                     ):
        t=None
        if min_cut<1.0:
            t = sample_dict(self.ystats.v, self.ystats.c, min_cut=min_cut)
            
        i, ids, labels, words, chars = 0,[],[],[],[]
        self._word_count = 0
        for d in self.reader.line_stream(stop=stop, t=t):# reader=FieldParser!
            if skip_ids:
                if d.id in skip_ids:
                    continue
            if hit_ids:
                if d.id not in hit_ids:
                    continue
            if len(d.w)==0:
                continue
            
            ids.append(d.id)
            labels.append(d.y)
            self._word_count+=len(d.w)
            
            dw = d.w
            dc = d.c
            if split_sentences:
                dw = U.lindexsplit(d.w, d.ws)
                dc = U.lindexsplit(d.c, d.cs)
            words.append(dw)
            chars.append(dc)
            
            i=i+1
            if i == self.batch_size:
                yield self.batch(ids, labels, words, chars, w, c, trim_words, spad=spad, wpad=wpad, cpad=cpad, split_sentences=split_sentences)
                i, ids, labels, words, chars = 0,[],[],[],[]
        
        if i>0 and partial:
            yield self.batch(ids, labels, words, chars, w, c, trim_words, spad=spad, wpad=wpad, cpad=cpad, split_sentences=split_sentences)
            
            

def first_nonzero(a):
    di=[]
    for i in range(len(a)):
        idx=np.where(a[i]!=0)
        try:
            di.append(idx[0][0])
        except IndexError:
            di.append(len(a[i]))
    return di

def test_text_reader():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    shard_patt = os.path.join(data_dir, 'holdout', 'ets.2016.heldout-00001-of-00050')
    
    reader =  GlobReader(shard_patt, chunk_size=1000, shuf=False)
    text_parser = TextParser(vocab_file, reader=reader)
        
    for d in text_parser.chunk_stream(stop=True):
        print(len(d.w))

def test_ystats():
    emb_dir = '/home/david/data/embed'
    emb_file = os.path.join(emb_dir, 'glove.6B.100d.txt')
    
    data_dir = '/home/david/data/ets1b/2016'
    id = 70088; essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    reader =  GlobReader(essay_file, chunk_size=10000, regex=REGEX_NUM, shuf=True)
    
    E, word_vocab = Vocab.load_word_embeddings(emb_file, essay_file, min_freq=1)
    text_parser = TextParser(word_vocab=word_vocab)
    
    fields = {0:'id', 1:'y', -1:text_parser}
    field_parser = FieldParser(fields, reader=reader)
    
    #field_parser.sample()
    ystats = field_parser.get_ystats()
    print(ystats)
    
def test_essay_reader():
    data_dir = '/home/david/data/ets1b/2016'
    id = 63986; essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    data_dir = '/home/david/data/ats/ets'
    id = 55433; essay_file = os.path.join(data_dir, '{0}', 'text.txt').format(id)
    
    regex_num = r'^[0-9]*\t([0-9]\t)+(?!\s*$).+'
    
    reader =  GlobReader(essay_file, chunk_size=1000, regex=regex_num, shuf=False)
    for line in reader.line_stream(stop=True):
        print(line)
        break
    
def test_essay_parser():
    emb_dir = '/home/david/data/embed'
    emb_file = os.path.join(emb_dir, 'glove.6B.100d.txt')
    
#     data_dir = '/home/david/data/ets1b/2016'
#     id = 63986; essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    data_dir = '/home/david/data/ats/ets'
    id = 55433; essay_file = os.path.join(data_dir, '{0}', 'text.txt').format(id)
    
    regex_num = r'^[0-9]*\t([0-9]\t)+(?!\s*$).+'
    reader =  GlobReader(essay_file, chunk_size=1000, regex=regex_num, shuf=False)
    
    E, word_vocab = Vocab.load_word_embeddings(emb_file, essay_file, min_freq=10)
    text_parser = TextParser(word_vocab=word_vocab, tokenize=True)
    
    fields = {0:'id', 1:'y', -1:text_parser}
    field_parser = FieldParser(fields, reader=reader)
    
    for d in field_parser.line_stream(stop=True):
        print(d.w)
        #break

def test_text_batcher():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    #shard_patt = os.path.join(data_dir, 'train', 'ets.2016-00001-of-00100')
    shard_patt = os.path.join(data_dir, 'holdout', 'ets.2016.heldout-0000*-of-00050')
    
    reader =  GlobReader(shard_patt, chunk_size=1000, shuf=True)
    
    word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
    text_parser = TextParser(word_vocab=word_vocab, char_vocab=char_vocab, max_word_length=max_word_length, reader=reader)
    
    batcher = TextBatcher(reader=text_parser, batch_size=128, num_unroll_steps=20, batch_chunk=50, trim_chars=True)
    
    #i=1
    for b in batcher.batch_stream(stop=True):
        #print(x)
        #print(y)
        print('{}\t{}'.format(b.w.shape, b.c.shape))
        #print(i);i=i+1

''' CHAR EMBEDDINGS '''
def test_essay_batcher_1():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    id = 62051 # 63986 62051 70088
    essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    reader =  GlobReader(essay_file, chunk_size=1000, regex=REGEX_NUM, shuf=True)
    
    word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
    text_parser = TextParser(word_vocab=word_vocab, char_vocab=char_vocab, max_word_length=max_word_length)
    
    fields = {0:'id', 1:'y', -1:text_parser}
    field_parser = FieldParser(fields, reader=reader)
    
    batcher = EssayBatcher(reader=field_parser, batch_size=128, max_word_length=max_word_length, trim_words=True, trim_chars=False)
    for b in batcher.batch_stream(stop=True):
        print('{}\t{}\t{}'.format(b.w.shape, b.c.shape, b.y.shape))

''' GLOVE WORD EMBEDDINGS '''
def test_essay_batcher_2():
    char = True
#     char = False

    U.seed_random(1234)
    keep_unk = False
    
    emb_dim = 100
    emb_path = '/home/david/data/embed/glove.6B.{}d.txt'
    vocab_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(vocab_dir, 'vocab_n250.txt')
    
    data_dir = '/home/david/data/ats/ets'
    id = 55433; essay_file = os.path.join(data_dir, '{0}', 'text.txt').format(id)
    #     id = 63986; essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    reader =  GlobReader(essay_file, chunk_size=1000, regex=REGEX_NUM, shuf=True)
    
    if char:
        word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
        text_parser = TextParser(word_vocab=word_vocab, char_vocab=char_vocab, keep_unk=keep_unk)
    else:
        E, word_vocab = Vocab.load_word_embeddings_ORIG(emb_path, emb_dim, essay_file, min_freq=5)
        #E, word_vocab = Vocab.load_word_embeddings(emb_file, essay_file, min_freq=2)
        text_parser = TextParser(word_vocab=word_vocab, keep_unk=keep_unk)
    
    fields = {0:'id', 1:'y', -1:text_parser}
    field_parser = FieldParser(fields, reader=reader, seed=1234)
    
    batcher = EssayBatcher(reader=field_parser, batch_size=32, trim_words=True)
    for b in batcher.batch_stream(stop=True, split_sentences=True):
        print('{}\t{}'.format(b.w.shape, b.y.shape))

''' char and word embeddings '''
def test_response_batcher():
    char = True
    char = False

    U.seed_random(1234)
    keep_unk = False
    
    emb_dir = '/home/david/data/embed'; emb_file = os.path.join(emb_dir, 'glove.6B.100d.txt')
    emb_path = '/home/david/data/embed/glove.6B.{}d.txt'
    
    vocab_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(vocab_dir, 'vocab_n250.txt')
    
    data_dir = '/home/david/data/ats/ets'
    id = 55399; 
    #essay_file = os.path.join(data_dir, '{0}/text.txt').format(id)
    essay_file = os.path.join(data_dir, '{0}/{0}.txt').format(id)
    
    reader =  GlobReader(essay_file, chunk_size=1000, regex=REGEX_NUM, shuf=True)
    
    if char:
        word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
        text_parser = TextParser(word_vocab=word_vocab, char_vocab=char_vocab, keep_unk=keep_unk)
    else:
        E, word_vocab = Vocab.load_word_embeddings_ORIG(emb_path, 100, essay_file, min_freq=5)
        #E, word_vocab = Vocab.load_word_embeddings(emb_file, essay_file, min_freq=2)
        text_parser = TextParser(word_vocab=word_vocab, keep_unk=keep_unk)
    
    fields = {0:'id', 1:'y', -1:text_parser}
    field_parser = FieldParser(fields, reader=reader, seed=1234)
    
    tot_vol = 0
    batcher = ResponseBatcher(reader=field_parser, batch_size=64, trim_words=True)
    for b in batcher.batch_stream(stop=True, split_sentences=True):
        vol = np.prod(b.x.shape); tot_vol+=vol
        print('{}\t{}\t{}'.format(vol, b.x.shape, b.y.shape))
    print('tot_vol:\t{}'.format(tot_vol))
                     
if __name__ == '__main__':
    #test_essay_reader()
#     test_essay_parser()
#     test_ystats()
#     test_text_reader()
#     test_text_batcher()
#     test_essay_batcher_1()
#     test_essay_batcher_2()
    test_response_batcher()
    print('done')