from __future__ import print_function

import os,sys
import glob
import docx2txt
from nltk.tokenize import word_tokenize

path = '/media/david/Elements/AIG/forms_for_AIG/2009/syn'

files = glob.glob(os.path.join(path,'*.docx'))
files.sort()
ss = []
ii = [3,8,12,16,20]
for f in files:
    text = docx2txt.process(f)
    toks = word_tokenize(text)
    
    s = ''
    for i in ii:
        tok = toks[i]
        t='\t'
        if len(tok)<8: t='\t\t'
        s = s + tok + t
    ss.append(s)
    #break
    
ss = list(set(ss))
ss.sort()
[print(s) for s in ss]