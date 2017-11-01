from __future__ import print_function
from __future__ import division

import os
import codecs
import collections
import numpy as np
import pandas as pd
import random
import glob
import pickle
import time

from nlp.util.utils import get_seed

ALL_CHARS = True

class Vocab:
    def __init__(self, token2index=None, index2token=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=0):#default=None
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)
            
    def get_tok_array(self, toks):
        tok_array = []
        for tok in '{' + toks + '}':
            t = self.get(tok)
            if t>0:
                tok_array.append(t)
        return tok_array

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)
        return cls(token2index, index2token)
    
    @staticmethod
    def clean(word, max_word_length, eos='+'):
        word = word.strip().replace('}', '').replace('{', '').replace('|', '')
        if eos:
            word = word.replace(eos, '')
        if len(word) > max_word_length - 2:  # space for 'start' and 'end' chars
            word = word[:max_word_length-2]
        return word
    
def get_char_aray(word, char_vocab, word_vocab, all_chars=True):
    if all_chars:
        char_array = char_vocab.get_tok_array(word)
    else:
        char_array = char_vocab.get_tok_array(word_vocab.token(word_vocab.get(word)))
    return char_array

def load_vocab(vocab_file, max_word_length=60, eos='+'):
    char_vocab = Vocab()
    char_vocab.feed(' ')  # blank is at index 0 in char vocab
    char_vocab.feed('{')  # start is at index 1 in char vocab
    char_vocab.feed('}')  # end   is at index 2 in char vocab
    char_vocab.feed('|')
    if eos:
        char_vocab.feed(eos)
        
    word_vocab = Vocab()
    word_vocab.feed('|')  # <unk> is at index 0 in word vocab
    if eos:
        word_vocab.feed(eos)

    actual_max_word_length = 0
    with codecs.open(vocab_file, "r", "utf-8") as f:
        for line in f:
            word, count = line.strip().split()
            word = Vocab.clean(word, max_word_length, eos)
            
            word_vocab.feed(word)
            
            for c in word:
                char_vocab.feed(c)
                
            actual_max_word_length = max(actual_max_word_length, len(word)+2)
    
    assert actual_max_word_length <= max_word_length
    
    print()
    print('actual longest token length is:', actual_max_word_length)
    print('size of word vocabulary:', word_vocab.size)
    print('size of char vocabulary:', char_vocab.size)
    return word_vocab, char_vocab, actual_max_word_length

def load_shard(shard_file, word_vocab, char_vocab, max_word_length, eos='+', seed=0, shuf=True):
    word_tokens = []
    char_tokens = []

    print('READING... ', shard_file)
    with codecs.open(shard_file, 'r', 'utf-8') as f:
        lines = [line.strip() for line in f]
        if shuf:
            if seed==None or seed<=0:
                seed = get_seed()
            rng = np.random.RandomState(seed)
            rng.shuffle(lines)

        for line in lines:
            for word in line.split():
                word = Vocab.clean(word, max_word_length, eos)
                
                word_idx = word_vocab.get(word)
                word_tokens.append(word_idx)
                
                char_array = get_char_aray(word, char_vocab, word_vocab, all_chars=ALL_CHARS)
                char_tokens.append(char_array)

            if eos:
                word_tokens.append(word_vocab.get(eos))
                char_tokens.append(char_vocab.get_tok_array(eos))

    # now we know the sizes, create tensors
    word_tensors = np.array(word_tokens, dtype=np.int32)
    char_tensors = np.zeros([len(char_tokens), max_word_length], dtype=np.int32)

    for i, char_array in enumerate(char_tokens):
        char_tensors [i,:len(char_array)] = char_array
        
    return word_tensors, char_tensors

def load_essays(essay_file, word_vocab, char_vocab, max_word_length, eos='+', shuf=True, I=0):
    essays = []
    lines = []

    print('READING... ', essay_file)
    with codecs.open(essay_file, 'r', 'utf-8') as f:
        #lines = [line.strip() for line in f]
        n = 0
        for line in f:
            lines.append(line.strip())
            n +=1
            if n>=100:
                break
            
        if shuf:
            random.shuffle(lines)
        for line in lines:
            word_tokens = []
            char_tokens = []
            if I>0:
                extra_cols = []
            if I>0:
                parts = line.split('\t')
                line = '\t'.join(parts[I:])
                extra_cols = parts[0:I]
            for word in line.split():
                word = Vocab.clean(word, max_word_length, eos)
                
                word_idx = word_vocab.get(word)
                word_tokens.append(word_idx)
                
                char_array = get_char_aray(word, char_vocab, word_vocab, all_chars=ALL_CHARS)
                char_tokens.append(char_array)

            if eos:
                word_tokens.append(word_vocab.get(eos))
                char_tokens.append(char_vocab.get_tok_array(eos))
            
            if I>0:
                essay = (word_tokens, char_tokens, extra_cols)
            else:
                essay = (word_tokens, char_tokens)
            essays.append(essay)
            
    essays = map(list, zip(*essays))
    return essays[0], essays[1], essays[2]

class DataReader:
    def __init__(self, word_tensor, char_tensor, batch_size, num_unroll_steps):
        length = word_tensor.shape[0]
        assert char_tensor.shape[0] == length

        max_word_length = char_tensor.shape[1]

        # round down length to whole number of slices
        reduced_length = (length // (batch_size * num_unroll_steps)) * batch_size * num_unroll_steps
        word_tensor = word_tensor[:reduced_length]
        char_tensor = char_tensor[:reduced_length, :]

        ydata = np.zeros_like(word_tensor)
        ydata[:-1] = word_tensor[1:].copy()
        ydata[-1] = word_tensor[0].copy()

        x_batches = char_tensor.reshape([batch_size, -1, num_unroll_steps, max_word_length])
        y_batches = ydata.reshape([batch_size, -1, num_unroll_steps])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        assert len(self._x_batches) == len(self._y_batches)
        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
        self.wpb = np.mean([np.prod(y.shape) for y in self._y_batches])
        
    def iter(self):
        for x, y in zip(self._x_batches, self._y_batches):
            yield x, y

class Dataset(object):
    def __init__(self, word_vocab, char_vocab, file_pattern, 
                 batch_size, num_unroll_steps, max_word_length, 
                 seed=0, shuf=True):
        self._word_vocab = word_vocab
        self._char_vocab = char_vocab
        self._file_pattern = file_pattern
        self._batch_size = batch_size
        self._num_unroll_steps = num_unroll_steps
        self._max_word_length = max_word_length
        if seed==None or seed<=0:
            self._seed = get_seed()
        else:
            self._seed = seed
        self._rng = np.random.RandomState(self._seed)
        self._shuf = shuf
        self.num_shards = None
        self.bps = None
        self.wpb = None
        self.prev_file = ''
    
    def length(self):
        if not self.num_shards is None and not self.bps is None:
            return self.num_shards*self.bps
        return None
    
    def new_shard(self):
        new = (self.cur_file != self.prev_file) and (len(self.prev_file)>0)
        self.prev_file = self.cur_file
        return new
    
    def file_stream(self):
        file_names = glob.glob(self._file_pattern)
        file_names.sort()
        if self._shuf:
            self._rng.shuffle(file_names)
        if self.num_shards is None:
            self.num_shards = len(file_names)
        for file_name in file_names:
            yield file_name
            
    def iter(self):
        while True:
            for file in self.file_stream():
                self.cur_file = file
                word_tensors, char_tensors = load_shard(shard_file=file, 
                                                        word_vocab=self._word_vocab, 
                                                        char_vocab=self._char_vocab, 
                                                        max_word_length=self._max_word_length, 
                                                        seed=self._rng.randint(np.iinfo(np.int32).max),
                                                        shuf=self._shuf)
                
                shard_reader = DataReader(word_tensors, char_tensors, self._batch_size, self._num_unroll_steps)
                
                if self.bps is None:
                    self.bps = shard_reader.length
                if self.wpb is None:
                    self.wpb = shard_reader.wpb
                for x, y in shard_reader.iter():
                    yield x, y
                    
            #break
    
    def batch_stream(self):
        return self.iter()
    
def load_test():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    #shard_file = os.path.join(data_dir, 'train', 'ets.2016-00001-of-00100')
    shard_file = os.path.join(data_dir, 'holdout', 'ets.2016.heldout-00000-of-00050')
    
    word_vocab, char_vocab, max_word_length = load_vocab(vocab_file, 65)
    print('Done loading vocab.')
    
    word_tensors, char_tensors = load_shard(shard_file, word_vocab, char_vocab, max_word_length)
    print('Done loading shard.')

def val_map(y,x=None):
    m = {}
    for v in np.unique(y):
        m[v] = np.where(y==v)
        if x is not None:
            m[v] = x[m[v]]
    return m
    
def load_mode_data_test():
    id = 54183
    idstr = '{}'.format(id)
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    essay_file = os.path.join(data_dir, idstr, idstr + '.txt.clean.tok')
    
    word_vocab, char_vocab, max_word_length = load_vocab(vocab_file, 65)
    print('Done loading vocab.')
    
    words, chars, feats = load_essays(essay_file, word_vocab, char_vocab, max_word_length, I=4)
    print('Done loading essays.')
    
    ####
    df = pd.DataFrame(feats)
    ids = np.array(df[0].values.astype('int32'))
    
    dfm = df.loc[df[1]=='m']
    idx_mode = np.array(dfm.index.values.astype('int32'))
    ids_mode = np.array(dfm[0].values.astype('int32'))
    
    y = np.array(pd.to_numeric(df[1].values, errors='coerce'))
    idx_num = np.squeeze(np.argwhere(~ np.isnan(y)))
    ids_num = ids[idx_num]
    y = y[idx_num].astype('int32')
    
    # maps score-point to essays
    idx_dict = val_map(y,idx_num)
#     idx_dict[CCN] =  idx_mode
    
    dd = {}
    for k,v in idx_dict.iteritems():
        dd[k] = ([words[i] for i in v], [chars[i] for i in v])

def test_data_reader():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    word_vocab, char_vocab, max_word_length = load_vocab(vocab_file)
    
    #patt = os.path.join(data_dir, 'holdout', 'ets.2016.heldout-00001-of-00050')
    patt = os.path.join(data_dir, 'train', 'ets.2016-00001-of-00100')
    
    reader = Dataset(word_vocab, char_vocab, patt, 128, 20, max_word_length, shuf=True)
    
    i=1
    for x, y in reader.batch_stream():
        #print(x)
        #print(y)
        #print('{}\t{}'.format(x.shape, y.shape))
        print(i);i=i+1
     
if __name__ == '__main__':
#     load_test()
#     load_mode_data_test()
    test_data_reader()
    print('done')
    