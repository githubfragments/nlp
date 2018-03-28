from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import time
import collections
import numpy as np
import tensorflow as tf
import sonnet as snt

from nlp.util import utils as U
from nlp.util.utils import ps

from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import variable_scope as vs

def softmax_rescale(x, mask, axis=-1):
    u = tf.multiply(x, mask)
    v = tf.reduce_sum(u, axis=axis, keepdims=True)
    ## fix div by 0
    v = v + tf.cast( tf.equal( v, tf.constant( 0, dtype=tf.float32 ) ), tf.float32)
    ##
    return u/v

def default_initializers(std=None, bias=None):
    if std != None:
        w_init = tf.truncated_normal_initializer(stddev=std)
    else:
        w_init = tf.glorot_uniform_initializer(dtype=tf.float32)
    
    if bias != None:
        b_init = tf.constant_initializer(bias)
    else:
        b_init = w_init
        
    return w_init, b_init

class Linear(snt.AbstractModule):
    def __init__(self, output_dim, w_init=None, b_init=None, name="linear"):
        super(Linear, self).__init__(name=name)
        self.output_dim = output_dim
        self.name = name
        self.w_init = w_init
        self.b_init = b_init
                
    def _build(self, inputs):
        d = inputs.shape[-1].value
        W = tf.get_variable('W', shape=[d, self.output_dim], initializer=self.w_init)
        b = tf.get_variable('b', shape=[self.output_dim], initializer=self.b_init)
        
        ''' build einsum index equation '''
        dim = len(inputs.shape)
        q1 = 'ij,jk'
        q2 = 'ik'
        for i in range(dim-2):
            c = chr(i+97)
            q1 = c+q1
            q2 = c+q2
        q = '{}->{}'.format(q1,q2)
        #print(q)
        
        ''' multiply '''
        output = tf.einsum(q, inputs, W) + b
        #tf.summary.histogram('{}_output'.format(self.name), output)
        return output

class Attention(snt.AbstractModule):
    def __init__(self, FLAGS,
                 name="attention"):
        super(Attention, self).__init__(name=name)
        self.FLAGS = FLAGS
    
    ''' https://github.com/ilivans/tf-rnn-attention/blob/master/attention.py
    '''
    def build(self, inputs):
        A = self.FLAGS.att_size
        D = inputs.shape[-1].value # D value - hidden size of the RNN layer
        mask = tf.cast(tf.abs(tf.reduce_sum(inputs, axis=2))>0, tf.float32)
        
        w_init, b_init = default_initializers(std=self.FLAGS.attn_std, bias=self.FLAGS.attn_b)
        lin_module = Linear(output_dim=A, w_init=w_init, b_init=b_init)
        u = tf.get_variable('u', shape=[A], initializer=w_init)
            
        ''' Linear Layer '''
        v = tf.tanh(lin_module(inputs))
        
        #beta = tf.tensordot(v, u, axes=1)
        beta = tf.einsum('ijk,k->ij', v, u)
        
        #T = tf.get_variable('T', shape=[1], initializer=tf.constant_initializer(1.0), dtype=tf.float32, trainable=True)
        
        ''' softmax '''
        with vs.variable_scope("alpha"):
            #alpha = mc.softmask(beta, mask=mask)
            alpha = tf.nn.softmax(beta, axis=1); 
            alpha = softmax_rescale(alpha, mask=mask, axis=1)
        
        ''' apply attn weights '''
        #w = inputs * tf.expand_dims(alpha, -1); output = tf.reduce_sum(w, 1)
        output = tf.reduce_sum(inputs * tf.expand_dims(alpha, -1), 1)
        
        #############################
        self._z = {}
#         self._z['inputs'] = inputs
#         self._z['u'] = u
#         self._z['v'] = v
#         self._z['w'] = w
#         self._z['beta'] = beta
#         self._z['alpha'] = alpha
#         self._z['output'] = output
#         self._z['mask'] = mask
        
        return output
    
    def _build(self, inputs):
        return self.build(inputs)
    
    @property
    def z(self):
        self._ensure_is_connected()
        try:
            return self._z
        except AttributeError:
            return []