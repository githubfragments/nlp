import os
import re
# import tempfile
from backports import tempfile

import numpy as np
import tensorflow as tf
from distutils.dir_util import copy_tree

def get_temp_dir():
    dirpath = tempfile.mkdtemp()
    return dirpath
    
def get_temp_filename():
    default_tmp_dir = tempfile._get_default_tempdir()
    temp_name = next(tempfile._get_candidate_names())
    return os.path.join(default_tmp_dir, temp_name)

def copy_chkpt(src_chkpt_dir, dst_chkpt_dir):
    copy_tree(src_chkpt_dir, dst_chkpt_dir)

def reverse(s):
    return s[::-1]

def common_suffix(s1, s2):
    return reverse(os.path.commonprefix([reverse(s1), reverse(s2)]))

def longest_common_suffix(s, slist):
    sxs = [common_suffix(s, ss) for ss in slist]
    idx = np.argmax(map(len, sxs))
    return slist[idx], sxs[idx]

def map_rename_vars(new_vars, old_vars):
    vmap = {}
    for new_var in new_vars:
        old_var, suffix = longest_common_suffix(new_var, old_vars)
        vmap[old_var] = new_var
    return vmap

def rename_vars(varnames_to_restore, src_chkpt_dir, dst_chkpt_dir):
    copy_chkpt(src_chkpt_dir, dst_chkpt_dir)
    rename(dst_chkpt_dir, varnames_to_restore)
    
def rename(chkpt_dir, varnames_to_restore, dry_run=False):
#     latest_chkpt_filename = tf.train.latest_checkpoint(chkpt_dir)
    checkpoint = tf.train.get_checkpoint_state(chkpt_dir)#, latest_filename=latest_chkpt_filename)
    
    varnames_to_restore = [re.sub(':[0-9]$', '', v) for v in varnames_to_restore]
    chkpt_vars = [var_name for var_name, _ in tf.contrib.framework.list_variables(chkpt_dir)]
    
    vmap = map_rename_vars(varnames_to_restore, chkpt_vars)
#     for k in vmap.keys(): print('{}\t=> {}'.format(k,vmap[k]))
    
    with tf.Session() as sess:
        for old_name in vmap.keys():
            var = tf.contrib.framework.load_variable(chkpt_dir, old_name)
            new_name = vmap[old_name]
            if dry_run:
                print('%s would be renamed to %s.' % (old_name, new_name))
            else:
                print('Renaming %s to %s.' % (old_name, new_name))
                # Rename the variable
                var = tf.Variable(var, name=new_name)

        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint.model_checkpoint_path)
            
    print(chkpt_dir)

if __name__ == '__main__':
#     src_chkpt_dir = '/home/david/code/python/tf-ats/tf-ats/lm_char/mod2_600-15'
#     dst_chkpt_dir = '/home/david/code/python/tf-ats/tf-ats/lm_char/tmp'
    chkpt_dir = '/home/david/code/python/tf-ats/tf-ats/lm_char/mod2_600-15'
    
    varnames_to_restore = [
        'Model/FlatModel/char_embed_b/internal_embed/embeddings:0',
        'Model/FlatModel/TDNN/conv_2d/w:0',
        'Model/FlatModel/TDNN/conv_2d/b:0',
        'Model/FlatModel/TDNN/conv_2d_1/w:0',
        'Model/FlatModel/TDNN/conv_2d_1/b:0'
        ]
    
    with tempfile.TemporaryDirectory() as tmp_chkpt_dir:
        rename_vars(varnames_to_restore, 
                    src_chkpt_dir=chkpt_dir,
                    dst_chkpt_dir=tmp_chkpt_dir)
            
    print('done')
           
# Model/FlatModel/char_embed_b/internal_embed/embeddings:0
# Model/FlatModel/TDNN/conv_2d/w:0
# Model/FlatModel/TDNN/conv_2d/b:0
# Model/FlatModel/TDNN/conv_2d_1/w:0
# Model/FlatModel/TDNN/conv_2d_1/b:0

# varnames_to_restore = ['Model/FlatModel/TDNN/conv_2d/w:0','Model/FlatModel/TDNN/conv_2d/b:0','Model/FlatModel/TDNN/conv_2d_1/w:0','Model/FlatModel/TDNN/conv_2d_1/b:0']

# /tmp/tmpkyVUpL
# /tmp/tmpy7dhTP
# /tmp/tmpxvCboJ

# /tmp/tmpBiksCs