
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

from vocab import Vocab
from nlp.util.utils import adict, get_seed, arrayfun


REGEX_NUM = r'^[0-9]*\t[0-9]\t[0-9]\t[0-9]\t(?!\s*$).+'
REGEX_MODE = r'^[0-9]*\tm\tm\tm\t(?!\s*$).+'

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
    def __init__(self, file_name, chunk_size=1000, shuf=True, regex=None, seed=None, rng=None):
        self.file_name = file_name
        self.chunk_size = chunk_size
        if chunk_size==None or chunk_size<=0: # read entire file as one chunk
            self.chunk_size=np.iinfo(np.int32).max
        self.shuf = shuf
        self.regex = regex
        if regex:
            self.regex = re.compile(regex)
        if seed==None or seed<=0:
            self.seed = get_seed()
        else:
            self.seed = seed
            self.shuf = True
        if rng==None:
            self.rng = np.random.RandomState(self.seed)
        else:
            self.rng = rng
            self.shuf = True
    
    def next_chunk(self, file_stream):
        lines = []
        try:
            for i in xrange(self.chunk_size):
                lines.append(next(file_stream).strip())
        except StopIteration:
            self.eof = True
        if self.regex:
            lines = filter(self.regex.search, lines)
        if self.shuf:
            self.rng.shuffle(lines)
        return lines
        
    def chunk_stream(self, stop=True):
        while True:
            self.eof = False
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
    def __init__(self, file_pattern, chunk_size=1000, shuf=True, regex=None, seed=None, rng=None):
        self.file_pattern = file_pattern
        self.file_names = glob.glob(self.file_pattern)
        self.file_names.sort()
        self.chunk_size = chunk_size
        self.shuf = shuf
        self.regex = regex
        if seed==None or seed<=0:
            self.seed = get_seed()
        else:
            self.seed = seed
            self.shuf = True
        if rng==None:
            self.rng = np.random.RandomState(self.seed)
        else:
            self.rng = rng
            self.shuf = True
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
                                            regex=self.regex, 
                                            seed=self.seed,
                                            rng=self.rng)
                for chunk in chunk_reader.chunk_stream(stop=True):
                    yield chunk
            if stop:
                break
    
    def line_stream(self, stop=True):
        for chunk in self.chunk_stream(stop=stop):
            for line in chunk:
                yield line

'''
returns dictionary: words->[word indices]
                    chars->[char indices]
set words=None, chars=None if not desired
'''
class TextParser(object):
    def __init__(self, word_vocab=None, char_vocab=None, max_word_length=None, reader=None, words='w', chars='c', eos='+', sep=' '):
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.max_word_length = max_word_length
        self.reader = reader
        self.words = words
        self.chars = chars
        self.eos = eos
        self.sep = sep
    
    ''' parses line into word/char tokens, based on vocab(s) '''
    def _parse_line(self, line, word_tokens, char_tokens):
        toks = line.split()
        
        for word in toks:
            word = Vocab.clean(word, self.max_word_length)
            
            if self.word_vocab:
                word_idx = self.word_vocab.get(word)
                word_tokens.append(word_idx)
            
            if self.char_vocab:
                char_array = Vocab.get_char_aray(word, self.char_vocab, self.word_vocab)
                char_tokens.append(char_array)
                
        if self.eos:
            if self.word_vocab: word_tokens.append(self.word_vocab.get(self.eos))
            if self.char_vocab: char_tokens.append(self.char_vocab.get_tok_array(self.eos))
            
        return word_tokens, char_tokens
    
    def parse_line(self, line):
        return self.package(*self._parse_line(line, word_tokens=[], char_tokens=[]))
    
    def package(self, word_tokens, char_tokens):
#         d = {}
#         if self.word_vocab:
#             d[self.words] = word_tokens
#         if self.char_vocab:
#             d[self.chars] = char_tokens
        d = {self.words : word_tokens , self.chars : char_tokens }
        return adict(d)
    
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


class FieldParser(object):
    def __init__(self, fields, reader=None, sep='\t'):
        self.fields = fields
        self.reader = reader
        self.sep = sep
    
    def parse_line(self, line):
        rec = line.strip().split(self.sep)
        d = {}
        for k,v in self.fields.items():
            if isinstance(v, basestring):
                d[v] = rec[k].strip()
            else:
                d.update(v.parse_line(rec[k].strip()))
        return adict(d)
    
    def line_stream(self, reader=None, stop=True):
        if reader==None:
            reader=self.reader
        if reader!=None:
            for line in self.reader.line_stream(stop=stop):
                yield self.parse_line(line)
    
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
        x = self.get_all_fields(key='label')
        x = np.array(x,dtype=np.float32)
        return np.mean(x), np.std(x), np.min(x), np.max(x), len(x)
                       
    def sample(self, sample_every=100, reader=None, stop=True):
        i=0
        for d in self.line_stream(reader=reader, stop=stop):
            i=i+1
            if i % 100 == 0:
                print('{} | {}\t{}\t{}'.format(i, d.id, d.label, d.w))
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
            for x, y in zip(batches[0], batches[1]):
                if self.trim_chars:
                    x = self.trim_batch(x)
                yield x, y

def nest_depth(x):
    depth = lambda L: isinstance(L, list) and max(map(depth, L))+1
    return depth(x)
        
def pad_sequences(sequences, max_text_length=None, max_word_length=None, dtype='int32', wpad='post', cpad='post', value=0.):
    num_samples = len(sequences)
    seq_lengths = map(len, sequences)
    if max_text_length is None:
        max_text_length = max(seq_lengths)
    
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
            for j,t in enumerate(s):
                k=0
                if wpad == 'pre':
                    k = max_text_length-len(s)
                if cpad == 'post':
                    y[j+k,:len(t)] = t
                else:
                    y[j+k,-len(t):] = t
            x[i,:] = y
        else:# <-- otherwise word sequence
            if wpad == 'post':
                x[i,:len(s)] = s
            else:
                x[i,-len(s):] = s
                
        
    return x, seq_lengths

            
## reader=FieldParser
class EssayBatcher(object):
    def __init__(self, reader, batch_size, max_text_length=None, max_word_length=None, trim_words=False, trim_chars=False, ystats=None):
        self.reader = reader
        self.batch_size = batch_size
        self.trim_words = trim_words
        self.trim_chars = trim_chars
        self.max_text_length = max_text_length
        self.max_word_length = max_word_length
        if trim_words:
            self.max_text_length = None
        elif max_text_length==None:
            self.max_text_length = reader.get_maxlen()# reader=FieldParser
            print('max essay length: {}'.format(self.max_text_length))
        if trim_chars:
            self.max_word_length = None
        if ystats is None:
            ystats = reader.get_ystats()
            print('\nYSTATS (mean,std,min,max,#): {}\n'.format(ystats))
        self.ystats = ystats
    
    ## to interval [0,1]
    def normalize(self, y):
        y = (y - self.ystats[2]) / (self.ystats[3]-self.ystats[2])# --> [0,1]
        #y = y - (self.ystats[0]-self.ystats[2])/(self.ystats[3]-self.ystats[2])
        return y
        #return (y - self.ystats[0]) / (self.ystats[3]-self.ystats[2])
        
    @property
    def ymean(self):
        return self.normalize(self.ystats[0])
    
    def word_count(self, reset=True):
        wc = self._word_count
        if reset:
            self._word_count = 0
        return wc
    
    '''
    use batch padding!
    https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
    '''
    def batch_stream(self, stop=False, skip_ids=None):
        i, ids, labels, words, chars = 0,[],[],[],[]
        self._word_count = 0
        for d in self.reader.line_stream(stop=stop):
            if skip_ids:
                if d.id in skip_ids:
                    continue
            ids.append(d.id)
            labels.append(d.label)
            words.append(d.w); self._word_count+=len(d.w)
            chars.append(d.c)
            i=i+1
            if i== self.batch_size:
                b = {'ids':ids}
                
                y = np.array(labels, dtype=np.float32)
                y = self.normalize(y)
                y = y[...,None]#y = np.expand_dims(y, 1)
                b['y'] = y
                
                if not isListEmpty(words):
                    word_tensor, seq_lengths = pad_sequences(words, max_text_length=self.max_text_length)
                    b['w'] = word_tensor
                
                if not isListEmpty(chars):
                    char_tensor, seq_lengths = pad_sequences(chars, max_text_length=self.max_text_length, max_word_length=self.max_word_length)
                    b['c'] = char_tensor
                
                b['s'] = seq_lengths
                yield adict(b)
                i, ids, labels, words, chars = 0,[],[],[],[]

         
def test():
    emb_dir = '/home/david/data/embed'
    emb_file = os.path.join(emb_dir, 'glove.6B.100d.txt')
    
    data_dir = '/home/david/data/ets1b/2016'
    id = 63986; essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    reader =  GlobReader(essay_file, chunk_size=1000, regex=REGEX_NUM, shuf=True)
    
    E, word_vocab = Vocab.load_word_embeddings(emb_file, essay_file, min_freq=1)
    text_parser = TextParser(word_vocab=word_vocab)
    
    fields = {0:'id', 1:'label', -1:text_parser}
    field_parser = FieldParser(fields, reader=reader)
    
    field_parser.sample()

def test_text_reader():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    shard_patt = os.path.join(data_dir, 'holdout', 'ets.2016.heldout-00001-of-00050')
    
    reader =  GlobReader(shard_patt, chunk_size=1000, shuf=False)
    text_parser = TextParser(vocab_file, reader=reader)
        
    for d in text_parser.chunk_stream(stop=True):
        print(len(d.w))
        
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
    for x, y in batcher.batch_stream(stop=True):
        #print(x)
        #print(y)
        print('{}\t{}'.format(x.shape, y.shape))
        #print(i);i=i+1

''' GLOVE WORD EMBEDDINGS '''       
def test_essay_batcher_2():
    emb_dir = '/home/david/data/embed'
    emb_file = os.path.join(emb_dir, 'glove.6B.100d.txt')
    
    data_dir = '/home/david/data/ets1b/2016'
    id = 63986; essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    reader =  GlobReader(essay_file, chunk_size=1000, regex=REGEX_NUM, shuf=True)
    
    E, word_vocab = Vocab.load_word_embeddings(emb_file, essay_file, min_freq=1)
    text_parser = TextParser(word_vocab=word_vocab)
    
    fields = {0:'id', 1:'label', -1:text_parser}
    field_parser = FieldParser(fields, reader=reader)
    
    batcher = EssayBatcher(reader=field_parser, batch_size=128, trim_words=True)
    for ids, y, x, seq_lens in batcher.batch_stream(stop=True):
        print('{}\t{}'.format(x.shape, y.shape))

''' CHAR EMBEDDINGS '''
def test_essay_batcher_1():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    id = 62051 # 63986 62051 70088
    essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    reader =  GlobReader(essay_file, chunk_size=1000, regex=REGEX_NUM, shuf=True)
    
    word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
    text_parser = TextParser(word_vocab=word_vocab, char_vocab=char_vocab, max_word_length=max_word_length)
    
    fields = {0:'id', 1:'label', -1:text_parser}
    field_parser = FieldParser(fields, reader=reader)
    
    batcher = EssayBatcher(reader=field_parser, batch_size=128, trim_words=True, trim_chars=True)
    for ids, y, w, c, seq_lens in batcher.batch_stream(stop=True):
        print('{}\t{}\t{}'.format(w.shape, c.shape, y.shape))
            
if __name__ == '__main__':
    test()
#     test_text_reader()
#     test_text_batcher()
#     test_essay_batcher_1()
#     test_essay_batcher_2()
    print('done')