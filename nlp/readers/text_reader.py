
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

'''
READ FILE IN CHUNKS
- reads chunk by chunk
- yields line by line
- filter by regex
- shuffles each chunk
'''
class ChunkReader(object):
    def __init__(self, file_name, chunk_size=1000, shuf=True, regex=None):
        self.file_name = file_name
        self.chunk_size = chunk_size
        if chunk_size==None or chunk_size<=0: # read entire file as one chunk
            self.chunk_size=np.iinfo(np.int32).max
        self.shuf = shuf
        self.regex = regex
        if regex:
            self.regex = re.compile(regex)
    
    def next_chunk(self, file_stream):
        lines = []
        try:
            for i in xrange(self.chunk_size):
                line = next(file_stream)
                lines.append(line.strip())
        except StopIteration:
            self.eof = True
        if self.regex:
            lines = filter(self.regex.search, lines)
        if self.shuf:
            random.shuffle(lines)
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
    def __init__(self, file_pattern, chunk_size=1000, shuf=True, regex=None, seed=None):
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
        self.rng = np.random.RandomState(self.seed)
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
    def line_stream(self, stop=True):
        while True:
            for file in self.file_stream():
                self.cur_file = file
                chunk_reader =  ChunkReader(file, chunk_size=self.chunk_size, shuf=self.shuf, regex=self.regex)
                for line in chunk_reader.line_stream(stop=True):
                    yield line
            if stop:
                break
'''
returns dictionary: words->[word indices]
                    chars->[char indices]
set words=None, chars=None if not desired
'''
class TextParser(object):
    def __init__(self, vocab_file, reader=None, words='words', chars='chars', sep=' ', eos='+'):
        self.vocab_file = vocab_file
        self.reader = reader
        self.sep = sep
        self.eos = eos
        self.words = words
        self.chars = chars
        self.word_vocab, self.char_vocab, self.max_word_length = Vocab.load_vocab(vocab_file)
    
    ''' parses line into word/char tokens, based on vocab(s) '''
    def parse_line(self, line):
        toks = line.split(self.sep)
        word_tokens = []
        char_tokens = []
        
        for word in toks:
            word = Vocab.clean(word, self.max_word_length)
            word_idx = self.word_vocab.get(word)
            word_tokens.append(word_idx)
            
            if self.chars:
                char_array = Vocab.get_char_aray(word, self.char_vocab, self.word_vocab)
                char_tokens.append(char_array)
                
        if self.eos:
            word_tokens.append(self.word_vocab.get(self.eos))
            if self.chars: char_tokens.append(self.char_vocab.get_tok_array(self.eos))
            
        d = {}
        if self.words:
            d[self.words] = word_tokens
            #word_tensors = np.array(word_tokens, dtype=np.int32)
            #d[self.words] = word_tensors
        if self.chars:
            d[self.chars] = char_tokens
            #char_tensors = np.zeros([len(char_tokens), self.max_word_length], dtype=np.int32)
            #for i, char_array in enumerate(char_tokens):
            #    char_tensors [i,:len(char_array)] = char_array
            #d[self.chars] = char_tensors
        return adict(d)
    
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
                print('{} | {}'.format(i, d.words))


class FieldParser(object):
    def __init__(self, fields, reader=None, sep='\t'):#vocab_file, cols=[0,1,4], eos='+'):
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
        n = 0;
        for d in self.line_stream():
            n = max(n,len(d.words))
        return n
                       
    def sample(self, sample_every=100, reader=None, stop=True):
        i=0
        for d in self.line_stream(reader=reader, stop=stop):
            i=i+1
            if i % 100 == 0:
                print('{} | {}\t{}\t{}'.format(i, d.id, d.label, d.words))
                #print('{} | {}'.format(i, d.words))
                
## reader=TextParser
class TextBatcher(object):
    def __init__(self, reader, batch_size, num_unroll_steps, batch_chunk = 100, trim_chars=False):
        self.reader = reader
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.max_word_length = reader.max_word_length# reader=TextParser
        self.batch_chunk = batch_chunk
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
            word_toks.extend(d.words)
            char_toks.extend(d.chars)
            N = N + len(d.words)
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
    
    def trim_batch(self, x):
        s = np.sum(np.sum(x,axis=1), axis=0)
        i = np.nonzero(s)[0][-1]+1
        return x[:,:,:i]
        
    def batch_stream(self, stop=False):
        tok_stream = self.reader.line_stream(stop=stop)
        while True:
            batches = self.make_batches(tok_stream)
            if batches is None:
                break
            for x, y in zip(batches[0], batches[1]):
                if self.trim_chars:
                    x = self.trim_batch(x)
                yield x, y
        
            
## reader=FieldParser
class EssayBatcher(object):
    def __init__(self, reader, batch_size, max_len=None):
        self.reader = reader
        self.batch_size = batch_size
        if max_len==None:
            self.max_len = reader.get_maxlen()# reader=FieldParser
            print('max essay length: {}'.format(self.max_len))
        else:
            self.max_len = max_len
    '''
    use batch padding instead!
    https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
    '''
    def batch_stream(self, stop=False):
        from keras.preprocessing import sequence
        i, labels, words, chars = 0,[],[],[]
        for d in self.reader.line_stream(stop=stop):
            labels.append(d.label)
            words.append(d.words)
            chars.append(d.chars)
            i=i+1
            if i== self.batch_size:
                words = sequence.pad_sequences(words, maxlen=self.max_len)
                word_tensor = np.array(words, dtype=np.int32)
                y_tensor = np.array(labels, dtype=np.float32)
                yield word_tensor, y_tensor
                i, labels, words, chars = 0,[],[],[]
                
        
def test_essay_batcher():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    id = 63986; essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    chunk_reader =  ChunkReader(essay_file, chunk_size=1000, regex=REGEX_NUM, shuf=False)
    text_parser = TextParser(vocab_file, words='words', chars='chars')
    fields = {0:'id', 1:'label', -1:text_parser}
    field_parser = FieldParser(fields, reader=chunk_reader)
    
    batcher = EssayBatcher(reader=field_parser, batch_size=128, max_len=1500)
    for x,y in batcher.batch_stream(stop=True):
        print('{}\t{}'.format(x.shape, y.shape))
         
def test():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    shard_file = os.path.join(data_dir, 'train', 'ets.2016-00001-of-00100')
    id = 63986; essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    ## read sentences only (shard file)
    chunk_reader =  ChunkReader(shard_file, chunk_size=1000, shuf=False)
    text_parser = TextParser(vocab_file, words='words', chars='chars')
    text_parser.sample(100, reader=chunk_reader, stop=True)
    
    ## read essays with id+label
    chunk_reader =  ChunkReader(essay_file, chunk_size=1000, regex=REGEX_NUM, shuf=False)
    fields = {0:'id', 1:'label', -1:text_parser}
    field_parser = FieldParser(fields, reader=chunk_reader)
    field_parser.sample(100, stop=True)
    
def test_text_batcher():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    #shard_file = os.path.join(data_dir, 'holdout', 'ets.2016.heldout-00000-of-00050')
    shard_patt = os.path.join(data_dir, 'holdout', 'ets.2016.heldout-00001-of-00050')
    
    #reader =  ChunkReader(shard_file, chunk_size=1000, shuf=False)
    reader =  GlobReader(shard_patt, chunk_size=1000, shuf=False)
    text_parser = TextParser(vocab_file, reader=reader)
    batcher = TextBatcher(reader=text_parser, batch_size=128, num_unroll_steps=20, batch_chunk = 10)
    
    for x, y in batcher.batch_stream(stop=True):
        #print(x)
        #print(y)
        print('{}\t{}'.format(x.shape, y.shape))
            
if __name__ == '__main__':
#     test()
    test_text_batcher()
#     test_essay_batcher()
    print('done')