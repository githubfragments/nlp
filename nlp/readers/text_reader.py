
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
from nlp.util.utils import adict

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
    def __init__(self, reader, batch_size, num_unroll_steps):
        self.reader = reader
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.max_word_length = reader.max_word_length# reader=TextParser
        self.M = self.batch_size * self.num_unroll_steps
        #self.word_toks, self.char_toks, self.N = [], [], 0
    
    def fill_toks(self, tok_stream):
        self.word_toks, self.char_toks, self.N = [], [], 0
        for d in tok_stream:
            self.word_toks.extend(d.words)
            self.char_toks.extend(d.chars)
            self.N = self.N + len(d.words)
            if self.N >= self.M:
                return True
        return False
    
    def batch_stream(self, stop=True):
        tok_stream = self.reader.line_stream(stop=stop)
        while True:
            if self.fill_toks(tok_stream):
                assert len(self.word_toks) == self.N
                word_tensor = np.array(self.word_toks[0:self.M], dtype=np.int32)
                char_tensor = np.zeros([self.M, self.max_word_length], dtype=np.int32)
                for i, char_array in enumerate(self.char_toks):
                    if i==self.M:
                        break
                    char_tensor [i,:len(char_array)] = char_array
                
                ydata = np.zeros_like(word_tensor)
                ydata[:-1] = word_tensor[1:].copy()
                ydata[-1] = word_tensor[0].copy()
                
                x_batches = char_tensor.reshape([self.batch_size, -1, self.num_unroll_steps, self.max_word_length])
                y_batches = ydata.reshape([self.batch_size, -1, self.num_unroll_steps])
                
                x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
                y_batches = np.transpose(y_batches, axes=(1, 0, 2))
                yield x_batches[0], y_batches[0]
                
            else:
                break
            
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
    
    def batch_stream(self, stop=True):
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

def test_text_batcher():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    #shard_file = os.path.join(data_dir, 'train', 'ets.2016-00001-of-00100')
    shard_file = os.path.join(data_dir, 'holdout', 'ets.2016.heldout-00001-of-00050')
    
    chunk_reader =  ChunkReader(shard_file, chunk_size=1000, shuf=False)
    text_parser = TextParser(vocab_file, reader=chunk_reader)
    
    batcher = TextBatcher(reader=text_parser, batch_size=256, num_unroll_steps=20)
    for x,y in batcher.batch_stream(stop=True):
        print('{}\t{}'.format(x.shape, y.shape))
        
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
    
    
            
if __name__ == '__main__':
#     test()
#     test_text_batcher()
    test_essay_batcher()
    print('done')