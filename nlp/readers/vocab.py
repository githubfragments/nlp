from __future__ import print_function

import os
import codecs
import numpy as np
import re
from future_builtins import map  # Only on Python 2
from collections import Counter
from itertools import chain
from nlp.util import utils as U


def word_counts(filename, min_freq=1):
    with open(filename) as f:
        d = Counter(chain.from_iterable(map(str.split, f)))
    if min_freq>1:
        for k in list(d):
            if d[k] < min_freq:
                del d[k]
    return d

def check_header(filename):
    with open(filename) as f:
        line = f.readline().rstrip().split()
        if len(line)!=2:
            return False
        try:
            float(line[0])
            float(line[1])
            return True
        except ValueError:
            return False
        return True

def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return str(''.join(stripped))

def clean_tags(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def corr(s):
    return re.sub(r'\.(?! )', ' . ', re.sub(r' +', ' ', s))
  
def load_word_list(file):
    has_header = check_header(file)
    words = []
    with codecs.open(file, "r", "utf-8") as f:
        for line in f:
            if has_header:
                has_header=False
                continue
            word = line.split(None, 1)[0]
            words.append(strip_non_ascii(word))
    return words

def load_embeddings(file, filter_words=None, verbose=True):
    has_header = check_header(file)
    word2emb = {}
    with codecs.open(file, "r", "utf-8") as f:
        for line in f:
            if has_header:
                has_header=False
                continue
            word = strip_non_ascii(line.split(None, 1)[0])
            if filter_words:
                if not word in filter_words:
                    continue

            v = line.split()[1:]
            word2emb[word] = np.array(v, dtype='float32')
    if verbose:
        print('Loaded {} word embeddings of dim {}'.format(len(word2emb), word2emb[next(iter(word2emb))].size))
    return word2emb

class Vocab:
    def __init__(self, 
                 token2index=None, 
                 index2token=None,
                 unk_index=0):
        self._token2index = token2index or {}
        self._index2token = index2token or []
        self.unk_index = unk_index
        self.reset_counts()
        
    def reset_counts(self):
        self._unk = 0
        self._tot = 0
        
    def unk2tot(self):
        return float(self._unk)/float(self._tot)

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

    ''' returns index 0 for unknown tokens! '''
    def get(self, token):
        idx = self._token2index.get(token, self.unk_index)
        self._tot += 1
        self._unk += (1 if idx==self.unk_index else 0)
        return idx
    
    def get_index(self, token):
        return self.get(token)

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
    def get_char_aray(word, char_vocab, word_vocab=None, all_chars=True):
        if all_chars:
            char_array = char_vocab.get_tok_array(word)
        else:
            char_array = char_vocab.get_tok_array(word_vocab.token(word_vocab.get(word)))
        return char_array
    
    @staticmethod
    def clean(word, max_word_length=None, eos='+', lower=False):
        word = word.strip().replace('}', '').replace('{', '').replace('|', '')
        if lower:
            word = word.lower()
        if eos:
            word = word.replace(eos, '')
        if max_word_length and len(word) > max_word_length - 2:  # space for 'start' and 'end' chars
            word = word[:max_word_length-2]
        return word
    
    @staticmethod
    def clean_line(line):
        #line = clean_tags(line)
        #line = corr(line)
        # LOOKUP ets_reader.py 'is_number' for number processing!!!
        return corr(clean_tags(line))
        

    @staticmethod
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
        has_header = check_header(vocab_file)
        with codecs.open(vocab_file, "r", "utf-8") as f:
            for line in f:
                if has_header:
                    has_header=False
                    continue
                word = line.split(None, 1)[0]
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

    @staticmethod
    def load_word_embeddings_ORIG(emb_path, emb_dim, data_file, min_freq=1, unk='<unk>', verbose=True):
        wc = word_counts(data_file, min_freq)
        words = set(wc)
        words.discard(unk)
        
        emb_file = emb_path.format(emb_dim)
        word2emb = load_embeddings(emb_file, filter_words=words, verbose=verbose)
        #words = list(word2emb)
        
        #z = np.zeros_like(word2emb[next(iter(word2emb))])
        n = len(word2emb) + 1
        d = word2emb[next(iter(word2emb))].size
        E = np.zeros([n, d], dtype=np.float32)# <unk> is given all-zero embedding... at E[0,:]
        
        word_vocab = Vocab()
        word_vocab.feed(unk)    # <unk> is at index 0 in word vocab --> so idx=0 returned for unknown toks
        for word in list(word2emb):
            idx = word_vocab.feed(word)
            E[idx,:] = word2emb[word]
            #print(word)
        
        # returns embedding matrix, word_vocab
        return E, word_vocab
    
    @staticmethod
    def load_word_embeddings(emb_path, emb_dim, data_file, min_freq=1, verbose=True):
        ## pre-load emb words
        from deepats import ets_reader
        from deepats.w2vEmbReader import W2VEmbReader as EmbReader
        
        emb_reader = EmbReader(emb_path, emb_dim)
        emb_words = emb_reader.load_words()
        
        text = U.read_col(data_file, col=-1, type='string')
        vocab = ets_reader.create_vocab(text, tokenize_text=True, to_lower=True, min_word_freq=min_freq, emb_words=emb_words)
        #  vocab = {'<pad>':0, '<unk>':1, '<num>':2, .....}
        
        #######################################################
        pad='<pad>';unk='<unk>';num='<num>'
        words = set(vocab)
        words.discard(pad);words.discard(unk);words.discard(num)
        
        emb_file = emb_path.format(emb_dim)
        word2emb = load_embeddings(emb_file, filter_words=words, verbose=verbose)
        
        n = len(word2emb) + 3
        d = word2emb[next(iter(word2emb))].size
        E = np.zeros([n, d], dtype=np.float32)# <unk> is given all-zero embedding... at E[0,:]
        
        word_vocab = Vocab(unk_index=1)
        word_vocab.feed(pad)    # <pad> is at index 0 in word vocab
        word_vocab.feed(unk)    # <unk> is at index 1 in word vocab --> so idx=1 returned for unknown toks
        word_vocab.feed(num)    # <num> is at index 2 in word vocab
        for word in list(word2emb):
            idx = word_vocab.feed(word)
            E[idx,:] = word2emb[word]
            #print(word)
        
        return E, word_vocab
            
def test1():
    data_dir = '/home/david/data/ets1b/2016'
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
    
def test2():
    data_dir = '/home/david/data/embed'
    vocab_file = os.path.join(data_dir, 'glove.6B.50d.txt')
    word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
        
def test3():
    data_dir = '/home/david/data/ets1b/2016'
    #vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    #word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
    id = 63986; essay_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    d = word_counts(essay_file, 2)
    return d

def test4():
    emb_path = '/home/david/data/embed/glove.6B.{}d.txt'
    emb_dim = 100
    
#     data_dir = '/home/david/data/ets1b/2016'
#     id = 63986; data_file = os.path.join(data_dir, '{0}', '{0}.txt.clean.tok').format(id)
    
    data_dir = '/home/david/data/ats/ets'
    id = 61190; essay_file = os.path.join(data_dir, '{0}', 'text.txt').format(id)
    
    #words = set(word_counts(data_file, min_freq=1))
    #word2emb = load_embeddings(emb_file, filter_words=words)
    E, word_vocab = Vocab.load_word_embeddings(emb_path, emb_dim, essay_file, min_freq=2)
    print(E.shape)
    
if __name__ == '__main__':
    #test1()
    #test2()
    #d = test3()
    test4()
    print('done')    