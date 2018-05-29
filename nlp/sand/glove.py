from __future__ import print_function
import os
import codecs
import gzip
import numpy as np
import urllib2
import pprint
import zipfile

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
       
class Glove:
    """
    Glove -- loads in GloVe pretrained models as provided on the Stanford NLP GloVe 
    website. Does cosine similarity nearest neighbors.
    """
    def __init__(self, glove_path=None, max_entries=20000, num_components=200):
        self.wd_to_i = {}
        self.i_to_wd = {}
        self.v = None
        if glove_path is not None:
            self.load_glove(glove_path, 
                            max_entries=max_entries, 
                            num_components=num_components)
    
    def load_glove(self, glove_path, gz=False, max_entries=20000, num_components=200):
        self.v = np.empty((max_entries, num_components))
        has_header = check_header(glove_path)
        i=-1
        with codecs.open(glove_path, "r", "utf-8") as f:
            for line in f:
                if has_header:
                    has_header=False
                    continue
                if i >= max_entries-1:
                    #print "Stopping: ", i, max_entries
                    break
                if i % 10000 == 0:
                    print(i)
                i+=1
                word = strip_non_ascii(line.split(None, 1)[0])
                self.wd_to_i[word] = i
                self.i_to_wd[i] = word
                self.v[i,:] = tuple(float(j) for j in line.split()[1:])             
                    
    def load_glove_OLD(self, glove_path, gz=False, max_entries=20000, num_components=200):
        """Load GloVe vectors into memory from txt file. Returns a dict where the keys are 
        the headwords from the model and the values are the n-dimensional vectors
        representing the word.
    
        Arguments
        glove_path: path to glove vectors (txt file or gzipped txt file)
        gz: is path to a gzipped file (default True)"""

        self.v = np.empty((max_entries, num_components))
        if gz:
            with gzip.open(glove_path, "r") as glove_file:
                utfreader = codecs.getreader("utf-8")
                for i, glove_entry in enumerate(utfreader(glove_file)):
                    if i >= max_entries:
                        break
                    self.split_and_store(glove_entry, i)
        else:
            with codecs.open(glove_path, "r", "utf-8") as glove_file:
                for i, glove_entry in enumerate(glove_file):
                    if i >= max_entries:
                        #print "Stopping: ", i, max_entries
                        break
                    self.split_and_store(glove_entry, i)


    def split_and_store(self, glove_entry, i):
        glove_components = glove_entry.split(' ')
        self.wd_to_i[glove_components[0]] = i
        self.i_to_wd[i] = glove_components[0]
        try:
            self.v[i,:] = tuple(float(i) for i in glove_components[1:])
        except (ValueError, IndexError) as e:
            print('poop')
            #print "Bad entry", i, ", GloVe components: \"", glove_components, "\""

    def nearest_to_vec(self, vec, n=10):
        similarities = np.dot(self.v, vec) / (np.linalg.norm(self.v,axis=1) * 
                                                np.linalg.norm(vec))
        # sort similarities largest to smallest
        simil_i = np.argsort(-similarities)
        return [(self.i_to_wd[i], similarities[i]) for i in simil_i[:n]]
    
    def nearest_euclidean(self, vec, n=10):
        sse = np.sum((self.v-vec) ** 2, axis=1)
        distances = np.sqrt(sse)
        # sort distances smallest to largest
        simil_i = np.argsort(distances)
        return [(self.i_to_wd[i], distances[i]) for i in simil_i[:n]]
    
    def get_nearest(self, keyword, n=10):
        kw_vector = self.v[self.wd_to_i[keyword]]
        return self.nearest_to_vec(kw_vector, n)
        
    def syn(self,w,n=10):
        [print(i[0]) for i in self.get_nearest(w,n=n)]
    
    def sims(self, w, ws):
        v = self.v[self.wd_to_i[w]]
        vs = np.array([self.v[self.wd_to_i[w]] for w in ws])
        sims = np.dot(vs,v)/(np.linalg.norm(vs,axis=1)*np.linalg.norm(v))
        return sims
    
    def plot_vec(self,vec):
        plt.bar(range(0, len(vec)), vec)

    def plot_wd(self,wd):
        self.plot_vec(self.v[self.wd_to_i[wd]])

    def get_vec(self,wd):
        return self.v[self.wd_to_i[wd]]
    
    def add_2_minus_1(self, to_add, to_subtract):
        return np.sum([self.get_vec(wd) for wd in to_add],axis=0) - self.get_vec(to_subtract)
    
    def __getitem__(self, key):
        # Pull some type awfulness
        if type(key) in (unicode, str):
            return self.get_vec(key)
        elif type(key) in (float, np.ndarray):
            return self.nearest_to_vec(key)
        else:
            raise IndexError()
###############################################################################
            
dim=300
vocab_size=100000
glove_folder = '/home/david/data/embed'

glove_path = os.path.join(glove_folder, 'glove.6B.{}d.txt'.format(dim))
myg = Glove(glove_path, max_entries=vocab_size, num_components=dim)

#######
myg.syn('apparently')
myg.syn('affront')

w = 'adamant'
ws = ['oblique','obscure','obsolete','obstinate']
sims = myg.sims(w,ws)