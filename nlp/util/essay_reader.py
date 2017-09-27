import os
import codecs
import collections
import numpy as np
import pandas as pd
import random
import glob
import re
from nlp.util.data_reader import *

REGEX_NUM = r'^[0-9]*\t[0-9]\t[0-9]\t[0-9]\t(?!\s*$).+'
REGEX_MODE = r'^[0-9]*\tm\tm\tm\t(?!\s*$).+'

def val_map(y,x=None):
    m = {}
    for v in np.unique(y):
        m[v] = np.where(y==v)
        if x is not None:
            m[v] = x[m[v]]
    return m

class ChunkReader(object):
    def __init__(self, file_name, chunk_size=1000, shuf=True, regex=None):
        self.file_name = file_name
        self.chunk_size = chunk_size
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

class EssayReader(object):
    def __init__(self, file_name, word_vocab, char_vocab, chunk_size=1000, regex=None, cols=[0,1,4], sep='\t', shuf=True, eos='+', max_word_length=60):
        self.file_name = file_name
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.chunk_size = chunk_size
        self.cols = cols
        self.sep = sep
        self.shuf = shuf
        self.eos = eos
        self.max_word_length = max_word_length
        self.chunk_reader =  ChunkReader(file_name, chunk_size=chunk_size, regex=regex, shuf=shuf)
    
    def parse_record(self, rec):
        id = rec[0]
        label = rec[1]
        toks = rec[2].split()
        word_tokens = []
        char_tokens = []
        
        for word in toks:
            word = Vocab.clean(word, self.max_word_length)
            word_idx = self.word_vocab.get(word)
            word_tokens.append(word_idx)
            
            char_array = get_char_aray(word, self.char_vocab, self.word_vocab)
            char_tokens.append(char_array)
                
        if self.eos:
            word_tokens.append(self.word_vocab.get(self.eos))
            char_tokens.append(self.char_vocab.get_tok_array(self.eos))
            
        return id, label, word_tokens, char_tokens
    
    def parse_chunk(self, chunk):
        data = []
        for line in chunk:
            parts = line.split(self.sep)
            if self.cols:
                parts = [parts[i] for i in self.cols]
            data.append(parts)
        #return map(list, zip(*data))
        return data
    
    def chunk_stream(self, stop=True):
        for chunk in self.chunk_reader.chunk_stream(stop=stop):
            yield self.parse_chunk(chunk)
        
    def record_stream(self, stop=True):
        for table in self.chunk_stream(stop=stop):
            for record in table:
                yield record
                
    def data_stream(self, stop=True):
        for record in self.record_stream(stop=stop):
            yield self.parse_record(record)


class EssaySetReader(object):
    def __init__(self, files, word_vocab, char_vocab, chunk_size=1000, regex=None, cols=[0,1,4], sep='\t', shuf=True, eos='+', max_word_length=60):
        file_names = files
        if isinstance(files, basestring):
            file_names = glob.glob(files)
        self.readers = []
        for file_name in file_names:
            self.readers.append(EssayReader(file_name, word_vocab, char_vocab, 
                                            chunk_size=chunk_size, regex=regex, 
                                            cols=cols, sep=sep, shuf=shuf, eos=eos, 
                                            max_word_length=max_word_length)
                                )
            
    def _stream(self, streams):
        while True:
            if len(streams)==0:
                break
            i = random.randint(0,len(streams)-1)
            try:
                yield next(streams[i])
            except StopIteration:
                del streams[i]
    
    def record_stream(self, stop=True):
        streams = [r.record_stream(stop=stop) for r in self.readers]
        return self._stream(streams)
    
    def data_stream(self, stop=True):
        streams = [r.data_stream(stop=stop) for r in self.readers]
        return self._stream(streams)
    

def test3():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    essay_files = []
    ids = [54183, 54703, 55381]
#     ids = [55381]
    for id in ids:
        idstr = '{}'.format(id)
        essay_file = os.path.join(data_dir, idstr, idstr + '.txt.clean.tok')
        essay_files.append(essay_file)
    
    word_vocab, char_vocab, max_word_length = load_vocab(vocab_file)
    reader = EssaySetReader(essay_files, word_vocab, char_vocab, chunk_size=2000, regex=REGEX_MODE)

#     n = 0
#     for r in reader.record_stream(stop=True):
#         print('{}\t{}'.format(r[0], r[1]))
#         n += 1
#     print('n={}'.format(n))
    
    n = 0
    for id, label, word_tokens, char_tokens in reader.data_stream(stop=True):
        print('{}\t{}'.format(id, label))
        n += 1
    print('n={}'.format(n))

def test1():
    id = 54183
    idstr = '{}'.format(id)
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    essay_file = os.path.join(data_dir, idstr, idstr + '.txt.clean.tok')
    
#     cr = ChunkReader(essay_file, 2000)
#     for chunk in cr.chunk_stream(stop=True):
#         print(len(chunk))

#     lines = filter(regex_num.search, chunk)
#     print(len(lines))
#     lines = filter(regex_mode.search, chunk)
#     print(len(lines))
    
    cr = ChunkReader(essay_file, 2000, regex=REGEX_NUM)
    cs = cr.chunk_stream(stop=True)
    chunk = next(cs)
    print(len(chunk))
    
    cr = ChunkReader(essay_file, 2000, regex=REGEX_MODE)
    cs = cr.chunk_stream(stop=True)
    chunk = next(cs)
    print(len(chunk))
    
def test2():
    id = 54183
    idstr = '{}'.format(id)
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    essay_file = os.path.join(data_dir, idstr, idstr + '.txt.clean.tok')
    
    word_vocab, char_vocab, max_word_length = load_vocab(vocab_file)
    er = EssayReader(essay_file, word_vocab, char_vocab, chunk_size=2000, regex=REGEX_MODE)
#     cs = er.chunk_stream()
#     data = next(cs)

    for r in er.record_stream(stop=True):
        print('{}\t{}'.format(r[0], r[1]))
    print('done')
    
#     for id, label, word_tokens, char_tokens in er.data_stream(stop=True):
#         print('{}\t{}'.format(id, label))
#     print('done')



if __name__ == '__main__':
#     test1()
#     test2()
    test3()

    
    print('DONE')