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
import tensorflow.contrib.layers as layers

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
    def __init__(self, output_dim,
                 act=tf.tanh,
                 w_init=None, 
                 b_init=None, 
                 name="linear"):
        super(Linear, self).__init__(name=name)
        self.output_dim = output_dim
        self.act = act
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
        
        ''' activation '''
        if self.act is not None:
            output = self.act(output)
      
        return output

''' https://github.com/ilivans/tf-rnn-attention/blob/master/attention.py '''
class Attention(snt.AbstractModule):
    def __init__(self, FLAGS,
                 name="attention"):
        super(Attention, self).__init__(name=name)
        self.FLAGS = FLAGS
    
    def build(self, inputs):
        A = self.FLAGS.att_size
        D = inputs.shape[-1].value # D value - hidden size of the RNN layer
        mask = tf.cast(tf.abs(tf.reduce_sum(inputs, axis=2))>0, tf.float32)
        
        w_init, b_init = default_initializers(std=self.FLAGS.attn_std, bias=self.FLAGS.attn_b)
        lin_module = Linear(output_dim=A, w_init=w_init, b_init=b_init)
        q = tf.get_variable('q', shape=[A], initializer=w_init)
            
        ''' Linear Layer '''
        k = lin_module(inputs)
        
        beta = tf.einsum('ijk,k->ij', k, q)#beta = tf.tensordot(k, q, axes=1)
        
        ''' softmax '''
        with vs.variable_scope("alpha"):
            #alpha = mc.softmask(beta, mask=mask)
            alpha = tf.nn.softmax(beta, axis=1) 
            alpha = softmax_rescale(alpha, mask=mask, axis=1)
        
        ''' apply attn weights '''
        #w = inputs * tf.expand_dims(alpha, -1); output = tf.reduce_sum(w, 1)
        output = tf.reduce_sum(inputs * tf.expand_dims(alpha, -1), 1)
        
        ''' inputs == v '''
        
        #############################
        self._z = {}
#         self._z['inputs'] = inputs
#         self._z['q'] = q
#         self._z['k'] = k
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
 
class Attention2D(snt.AbstractModule):
    def __init__(self, FLAGS,
                 name="Attention2D"):
        super(Attention2D, self).__init__(name=name)
        self.FLAGS = FLAGS
    
    def build(self, inputs):
        A = self.FLAGS.att_size
        R = self.FLAGS.attn_depth
        D = inputs.shape[-1].value # D value - hidden size of the RNN layer
        mask = tf.cast(tf.abs(tf.reduce_sum(inputs, axis=2))>0, tf.float32)
        
        ''' Linear Projection Layer (Key, Ws1) '''
        w_init, b_init = default_initializers(std=self.FLAGS.attn_std, bias=self.FLAGS.attn_b)
        lin_module = Linear(output_dim=A, w_init=w_init, b_init=b_init)
        K = lin_module(inputs)
        
        ''' Query '''
        Q = tf.get_variable('Q', shape=[A,R], initializer=w_init)
        beta = tf.einsum('bij,jk->bik', K, Q)
        
        ''' softmax '''
        with vs.variable_scope("alpha"):
            alpha = tf.nn.softmax(beta, axis=1)
            alpha = softmax_rescale(alpha, mask=mask, axis=1)
        
        ''' apply attn weights '''
        #w = inputs * tf.expand_dims(alpha, -1); output = tf.reduce_sum(w, 1)
        output = tf.reduce_sum(inputs * tf.expand_dims(alpha, -1), 1)
        
        ''' inputs == v '''
        
        #############################
        self._z = {}
#         self._z['inputs'] = inputs
#         self._z['q'] = q
#         self._z['k'] = k
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
        
def task_specific_attention(inputs, output_size,
                            initializer=layers.xavier_initializer(),
                            activation_fn=tf.tanh, scope=None):
    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None
    
    with tf.variable_scope(scope or 'attention') as scope:
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[output_size],
                                                   initializer=initializer,
                                                   dtype=tf.float32)
        input_projection = layers.fully_connected(inputs, output_size,
                                                  activation_fn=activation_fn,
                                                  scope=scope)
        
        keepdims=False
        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keepdims=keepdims)
        mask = tf.cast(tf.abs(tf.reduce_sum(input_projection, axis=2, keepdims=keepdims))>0, tf.float32)
        
        ''' softmax '''
        attention_weights = tf.nn.softmax(vector_attn, axis=1)
        #attention_weights = tf.contrib.sparsemax.sparsemax(vector_attn)
        
        attention_weights = softmax_rescale(attention_weights, mask=mask, dim=1)
        
        if not keepdims:
            attention_weights = tf.expand_dims( attention_weights, -1)
        outputs = tf.reduce_sum(tf.multiply(inputs, attention_weights), axis=1)
        
        tf.summary.histogram('{}_outputs'.format('task_specific_attention'), outputs)
        
        return outputs


''' softmax with 0-padding re-normalization '''  
def softmask(x, axis=-1, mask=None, T=None):
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x = x - x_max
#     if T!=None:
#         if not tf.is_numeric_tensor(T):
#             T = tf.get_variable('T', shape=[1], initializer=tf.constant_initializer(T), dtype=tf.float32, trainable=True)
#             T = tf.get_variable('T', shape=[1], initializer=tf.constant_initializer(1.0), dtype=tf.float32, trainable=True)
#         x = x/T
    ex = tf.exp(x)
    if mask!=None: ex = tf.multiply(ex, mask)
    es = tf.reduce_sum(ex, axis=axis, keepdims=True)
    ## fix div by 0
    es = es + tf.cast( tf.equal( es, tf.constant( 0, dtype=tf.float32 ) ), tf.float32)
    #     if mask!=None: es = es + tf.cast(tf.reduce_sum(mask, axis=-1, keep_dims=True)==0, tf.float32)
    return ex/es