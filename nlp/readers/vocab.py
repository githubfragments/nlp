from __future__ import print_function

import os
import codecs


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
        with codecs.open(vocab_file, "r", "utf-8") as f:
            for line in f:
                #word, count = line.strip().split()
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
    def get_char_aray(word, char_vocab, word_vocab, all_chars=True):
        if all_chars:
            char_array = char_vocab.get_tok_array(word)
        else:
            char_array = char_vocab.get_tok_array(word_vocab.token(word_vocab.get(word)))
        return char_array
    

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
    vocab_file = os.path.join(data_dir, 'vocab_n250.txt')
    word_vocab, char_vocab, max_word_length = Vocab.load_vocab(vocab_file)
    
    
    
if __name__ == '__main__':
    test1()
    test2()
    print('done')    