from __future__ import print_function
import os
import codecs
import gzip
import numpy as np
import urllib2
import pprint
import zipfile
import matplotlib.pyplot as plt
import pandas as pd

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
    def __init__(self, embed_path=None, max_entries=20000, dim=200):
        self.wd_to_i = {}
        self.i_to_wd = {}
        self.v = None
        if embed_path is not None:
            self.load_glove(embed_path, 
                            max_entries=max_entries, 
                            dim=dim)
    
    def load_glove(self, embed_path, gz=False, max_entries=20000, dim=200):
        self.v = np.empty((max_entries, dim))
        has_header = check_header(embed_path)
        i=-1
        with codecs.open(embed_path, "r", "utf-8") as f:
            for line in f:
                if has_header:
                    has_header=False
                    continue
                if i >= max_entries-1:
                    #print "Stopping: ", i, max_entries
                    break
                if i % 10000 == 0:
                    print(i)
                try:
                    word = strip_non_ascii(line.split(None, 1)[0])
                    self.v[i,:] = tuple(float(j) for j in line.split()[1:])
                except (ValueError,IndexError), e:
                    continue
                self.wd_to_i[word] = i
                self.i_to_wd[i] = word
                i+=1             

    def nearest_to_vec(self, vec, n=10):
        sims = np.dot(self.v,vec)/(np.linalg.norm(self.v,axis=1)*np.linalg.norm(vec))
        simil_i = np.argsort(-sims)
        return [(self.i_to_wd[i], sims[i]) for i in simil_i[:n]]
    
    def nearest_euclidean(self, vec, n=10):
        sse = np.sum((self.v-vec) ** 2, axis=1)
        distances = np.sqrt(sse)
        simil_i = np.argsort(distances)
        return [(self.i_to_wd[i], distances[i]) for i in simil_i[:n]]
        
    def cos(self, vec, v=None):
        eps=1e-15
        if v is None: v=self.v
        x = 1.0-np.dot(v,vec)/(np.linalg.norm(v,axis=1)*np.linalg.norm(vec))
        x[x<eps]=0
        x[x>1.0]=1.0
        return x
    
    def euc(self, vec, v=None):
        if v is None: v=self.v
        sse = np.sum((v-vec) ** 2, axis=1)
        distances = np.sqrt(sse)
        return distances
        
    def dist(self, vec, v=None, f='cos'):
        if f is 'cos': return self.cos(vec,v=v)
        else: return self.euc(vec,v=v)
        
    def nearest(self, vec, n=10, f='cos'):
        dist = self.dist(vec,f=f)
        simil_i = np.argsort(dist)
        return [(self.i_to_wd[i], dist[i]) for i in simil_i[:n]]
    
    def get_nearest(self, keyword, n=10, f='cos'):
        kw_vector = self.v[self.wd_to_i[keyword]]
        return self.nearest(kw_vector, n=n, f=f)
        
    def syn(self,w,n=10,f='cos'):
        ans = self.get_nearest(w,n=n,f=f)
        print('\n{}'.format(w))
        self.pprint(ans)
        return ans
    
    def sims(self, ws, w=None, f='cos',s=''):
        if w is None:
            w=ws[0]
            ws=ws[1:]
        vec = self.v[self.wd_to_i[w]]
        v = np.array([self.v[self.wd_to_i[ww]] for ww in ws])
        dist = self.dist(vec, v=v, f=f)
        ans = zip(ws, dist)
        print('\n{}{}'.format(w,s))
        j=self.pprint(ans)
        return ans,j
        
    def pprint(self, x):
        m=min(zip(*x)[1])
        i=0
        for r in x:
            s=''
            if m==r[1]:
                s='**'
                j=i
            i+=1
            print('{1:0.4g}\t{0}{2}'.format(r[0],r[1],s))
        return j
    
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

embed_folder = '/home/david/data/embed'

dim=300
#corp=6; vocab_size=200000
#corp=42; vocab_size=120000
corp=840; vocab_size=340000
fn = 'glove.{}B.{}d.txt'.format(corp, dim)

#dim=300; vocab_size=300000; fn='wiki.en.vec'
#dim=300; vocab_size=400000; fn='w2v/GoogleNews-vectors-negative300.txt'

###############################################################################
def scan_words():
    x = 3

def syn_test():
    vocab_size=340000
    myg = Glove(os.path.join(embed_folder,fn), max_entries=vocab_size, dim=dim)
    
    fxn='cos'
    #fxn='euc'
    
    path = '/media/david/Elements/AIG/items_for_psychometrics_use/txt'
    low = 'lower_vr_S.txt'
    mid = 'middle_vr_S.txt'
    files = [low,mid]
    
    n,m=0,0
    x,y=[],[]
    for f in files:
        print(f)
        df = pd.read_csv(os.path.join(path,f), sep='\t')
        for i, row in df.iterrows():
            r = row['RASCH']
            #if not r<100: continue
            ws = [row['prompt'],row['key'],row['distractor1'],row['distractor2'],row['distractor3']]
            try:
                sims,j=myg.sims(ws,f=fxn,s='\t{}'.format(r))
            except KeyError:
                continue
            m+=1; n+=(1 if j==0 else 0)
            #x.append(r); y.append(sims[0][1])
    
    print('\n{0}/{1} = {2:0.4g}%'.format(n,m,100*float(n)/float(m)))
    
    #r = np.corrcoef(np.array(x),np.array(y))[1][0]

###############################################################################
            

#######

#myg.syn('apparently',f=f)
#myg.syn('affront',f=f)

#pprint.pprint(myg[myg['talkin']-myg['talking']+myg['going']][:5])

#######
#myg.sims(['adamant','oblique','obscure','obsolete','obstinate'],f=f)
#myg.sims(['despair','anguish','abuse','deception','tragedy'],f=f)
#myg.sims(['peculiar','remarkable','dignified','familiar','superior'],f=f)

if __name__ == '__main__':
    syn_test()
    
