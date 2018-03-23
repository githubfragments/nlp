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

import nlp.tf_tools.model_components as mc

from nlp.util import utils as U
from nlp.util.utils import ps

from nlp.tf_tools.attn1 import Attn1
from nlp.tf_tools.attn2 import Attn2

from nlp.rwa.RWACell import RWACell
# from nlp.rwa.rwa_cell import RWACell

from nlp.rwa.rda_cell import RDACell

# from nlp.ran.ran import RANCell
from nlp.ran.ran_cell import RANCell

from nlp.tensorflow_with_latest_papers import rnn_cell_modern
from nlp.rhn.rhn import RHNCell2 as RHNCell

from nlp.rnn_cells.lru import LRUCell

from nlp.rnn_cells.MultiplicativeLSTM import MultiplicativeLSTMCell

from nlp.hyper.tf_layer_norm import HyperLnLSTMCell
from tensorflow.python.framework.tensor_shape import TensorShape


'''
https://github.com/deepmind/sonnet/blob/master/sonnet/examples/rnn_shakespeare.py
'''
def init_dict(initializer, keys):
    if initializer!=None:
        if U.isnum(initializer):
            initializer = tf.constant_initializer(initializer)
        return {k: initializer for k in keys}
    return None

class WordEmbed(snt.AbstractModule):
    def __init__(self, vocab_size=None, embed_dim=None, initial_matrix=None, trainable=True, name="word_embed"):
        super(WordEmbed, self).__init__(name=name)
        self._vocab_size = vocab_size# word_vocab.size
        self._embed_dim = embed_dim
        self._trainable = trainable
        if initial_matrix:
            self._vocab_size = initial_matrix.shape[0]
            self._embed_dim = initial_matrix.shape[1]
        
        with self._enter_variable_scope():# cuz in init (not build)...
            self._embedding = snt.Embed(vocab_size=self._vocab_size,
                                        embed_dim=self._embed_dim,
                                        trainable=self._trainable,
                                        name="internal_embed")
    
    # inputs shape = [batch_size, ?]
    # inputs = word_idx, output = input_embedded
    def _build(self, inputs):
        return self._embedding(inputs)


class CharEmbed(snt.AbstractModule):
    def __init__(self, vocab_size, embed_dim, max_word_length=None, initializer=None, trainable=True, name="char_embed"):
        super(CharEmbed, self).__init__(name=name)
        self._vocab_size = vocab_size# char_vocab.size
        self._embed_dim = embed_dim
        self._max_word_length = max_word_length
        self._initializers = init_dict(initializer, ['embeddings'])
        self._trainable = trainable
        
        with self._enter_variable_scope():# cuz in init (not build)...
            self._char_embedding = snt.Embed(vocab_size=self._vocab_size,
                                             embed_dim=self._embed_dim,
                                             trainable=self._trainable,
                                             name="internal_embed")
    
    # inputs shape = [batch_size, num_word_steps, max_word_length] (num_unroll_steps)
    # inputs = char ids, output = input_embedded
    ## or ##
    # inputs shape = [batch_size, num_sentence_steps, num_word_steps, max_word_length]
    # inputs = char ids, output = input_embedded
    
    def _build(self, inputs):
        output = self._char_embedding(inputs)
        
        max_word_length = self._max_word_length
        if max_word_length==None:
            max_word_length = tf.shape(inputs)[-1]
            
        output = tf.reshape(output, [-1, max_word_length, self._embed_dim])
        
        self._clear_padding_op = tf.scatter_update(self._char_embedding.embeddings,
                                                   [0],
                                                   tf.constant(0.0, shape=[1, self._embed_dim]))
        return output
    
    @property
    def embeddings(self):
        self._ensure_is_connected()
        return self._char_embedding.embeddings
    
    @property
    def clear_padding_op(self):
        self._ensure_is_connected()
        return self._clear_padding_op
    
    def clear_padding(self, sess):
        sess.run(self.clear_padding_op)
      
    def initialize_to(self, sess, v):
        self._ensure_is_connected()
        sess.run(tf.assign(self.embeddings, v))


''' Time Delay Neural Network'''
class TDNN(snt.AbstractModule):
    def __init__(self, kernels, kernel_features, initializer=None, name="tdnn"):
        super(TDNN, self).__init__(name=name)
        self._kernels = kernels
        self._kernel_features = kernel_features
        self._initializers = init_dict(initializer, ['w','b'])
        assert len(self._kernels) == len(self._kernel_features), 'Kernel and Features must have the same size'
        
    def _build(self, inputs):
        #max_word_length = inputs.get_shape()[1]
        max_word_length = tf.shape(inputs)[1]#xxx1
        embed_size = inputs.get_shape()[-1]
        
        inputs = tf.expand_dims(inputs, 1)
        
        layers = []
        self.conv_layers = []
        for kernel_size, kernel_feature_size in zip(self._kernels, self._kernel_features):
            reduced_length = max_word_length - kernel_size + 1

            ## [batch_size x max_word_length x embed_size x kernel_feature_size]
            conv_fxn = snt.Conv2D(output_channels=kernel_feature_size,
                                  kernel_shape=[1,kernel_size],# ?? [kernel_size,1] ??
                                  initializers=self._initializers,
                                  padding='VALID')
            conv = conv_fxn(inputs)
            self.conv_layers.append(conv_fxn)
            
            ## [batch_size x 1 x 1 x kernel_feature_size]
            
#             pool = tf.nn.max_pool(tf.tanh(conv), 
#                                   ksize= [1,1,reduced_length,1], 
#                                   strides= [1,1,1,1],
#                                   padding= 'VALID')

            # https://stackoverflow.com/questions/43574076/tensorflow-maxpool-with-dynamic-ksize#xxx2
            pool = tf.reduce_max(tf.tanh(conv),
                                 axis=2,
                                 keep_dims=True
                                 )
            
            layers.append(tf.squeeze(pool, [1, 2]))
            
        if len(self._kernels) > 1:
            output = tf.concat(layers, 1)
        else:
            output = layers[0]
        return output
    
    def initialize_conv_layer_to(self, sess, i, w, b):
        self._ensure_is_connected()
        sess.run(tf.assign(self.conv_layers[i].w, w))
        sess.run(tf.assign(self.conv_layers[i].b, b))

class Highway(snt.AbstractModule):
    def __init__(self, output_size, 
                 num_layers=1,
                 bias=-2.0,
                 f=tf.nn.relu,
                 initializer=None,
                 name="highway"):
        super(Highway, self).__init__(name=name)
        self._output_size = output_size
        self._num_layers = num_layers
        self._bias = bias
        self._f = f
        self._initializers = init_dict(initializer, ['w','b'])
        
    def _build(self, inputs):
        self.lin_g, self.lin_t = [],[]
        for idx in range(self._num_layers):
            lin_in_g = snt.Linear(output_size=self._output_size, initializers=self._initializers, name="lin_in_g")
            lin_in_t = snt.Linear(output_size=self._output_size, initializers=self._initializers, name="lin_in_t")
            
            self.lin_g.append(lin_in_g)
            self.lin_t.append(lin_in_t)
            
            g = self._f(lin_in_g(inputs))
            t = tf.sigmoid(lin_in_t(inputs) + self._bias)

            output = t * g + (1. - t) * inputs
            inputs = output
            
        return output
    
    def initialize_lin_layers_to(self, sess, Lg, Lt):
        self._ensure_is_connected()
        i=0
        for g, t in zip(Lg, Lt):
            sess.run(tf.assign(self.lin_g[i].w, g[0]))
            sess.run(tf.assign(self.lin_g[i].b, g[1]))
            sess.run(tf.assign(self.lin_t[i].w, t[0]))
            sess.run(tf.assign(self.lin_t[i].b, t[1]))
            i+=1
        
###############################################################################
''' simple sonnet wrapper for dropout'''
class Dropout(snt.AbstractModule):
    def __init__(self, keep_prob=None, name="dropout"):
        super(Dropout, self).__init__(name=name)
        self._keep_prob = keep_prob
        
        if keep_prob is None:
            with self._enter_variable_scope():
                self._keep_prob = tf.placeholder_with_default(1.0, shape=())
    
    def _build(self, inputs):
        return tf.nn.dropout(inputs, keep_prob=self._keep_prob)
    
    @property
    def keep_prob(self):
        self._ensure_is_connected()
        return self._keep_prob

#########################################################################################

def rnn_unit(args):
    kwargs = {}
    if args.unit=='snt.lstm':
        rnn = snt.LSTM
        kwargs = { 'forget_bias':args.forget_bias }
    elif args.unit=='lstm':
        rnn = tf.nn.rnn_cell.LSTMCell
        kwargs = { 'forget_bias':args.forget_bias, 'reuse':False, 'state_is_tuple':True }
    elif args.unit=='snt.gru':
        rnn = snt.GRU
    elif args.unit=='gru':
        rnn = tf.nn.rnn_cell.GRUCell
        kwargs = { 'reuse':False }
    elif args.unit=='ran':
        rnn = RANCell
    elif args.unit=='ran_ln':
        rnn = RANCell
        kwargs = { 'normalize':True }
    
    elif args.unit=='rwa':
        rnn = RWACell
        
#         decay_rate = [0.0]*args.rnn_size
#         #decay_rate = [0.693/10]*150 + [0.0]*150
#         #decay_rate = [0.693]*75 + [0.693/10]*75 + [0.693/100]*75 + [0.0]*75
#         decay_rate = tf.Variable(tf.constant(decay_rate, dtype=tf.float32), trainable=True, name='decay_rate')
#         
#         #std = 0.001
#         #decay_rate = tf.get_variable('decay_rate', shape=[args.rnn_size], initializer=tf.truncated_normal_initializer(mean=2*std,stddev=std))
#         
#         kwargs = { 'decay_rate':decay_rate }
        
    elif args.unit=='rwa_bn':
        rnn = RWACell
        kwargs = { 'normalize':True }
    elif args.unit=='rda':
        rnn = RDACell
    elif args.unit=='rda_bn':
        rnn = RDACell
        kwargs = { 'normalize':True }
    elif args.unit=='rhn':
        rnn = rnn_cell_modern.HighwayRNNCell
        kwargs = { 'num_highway_layers' : args.FLAGS.rhn_highway_layers,
                   'use_inputs_on_each_layer' : args.FLAGS.rhn_inputs,
                   'use_kronecker_reparameterization' : args.FLAGS.rhn_kronecker }
    elif args.unit=='rhn2':
        rnn = RHNCell
        # num_units, in_size, is_training, depth=3
        kwargs = { 'depth' : args.FLAGS.rhn_highway_layers }
    elif args.unit=='lru':
        rnn = LRUCell
    elif args.unit=='hlstm':
        rnn = HyperLnLSTMCell
        kwargs = {'is_layer_norm':True,
                  'state_is_tuple':False,
                  'hyper_num_units':128,
                  'hyper_embedding_size':32,
                  }
    elif args.unit=='mlstm':
        rnn = MultiplicativeLSTMCell
        kwargs = { 'forget_bias':args.forget_bias }
    return rnn, kwargs

def get_initial_state(cell, args, batch_size=None):
    if batch_size==None:
        batch_size = tf.shape(args.inputs)[0]
    
    if args.train_initial_state:
        if args.unit.startswith('snt'):
            return cell.initial_state(batch_size, tf.float32, trainable=True)
        print('TRAINABLE INITIAL STATE NOT YET IMPLEMENTED FOR: {} !!!'.format(args.unit))
#         else:
#             initializer = r2rt.make_variable_state_initializer()
#             return r2rt.get_initial_cell_state(cell, initializer, args.batch_size, tf.float32)
    return cell.zero_state(batch_size, tf.float32)

def create_rnn_cell(args, scope=None, dropout=True, batch_size=None):
    rnn, kwargs = rnn_unit(args)
    
    if scope!=None:
        with tf.variable_scope(scope):
            cell = rnn(args.rnn_size, **kwargs)
            initial_state = get_initial_state(cell, args, batch_size=batch_size)
    else:
        cell = rnn(args.rnn_size, **kwargs)
        initial_state = get_initial_state(cell, args, batch_size=batch_size)
    
    #cell = tf.contrib.rnn.ResidualWrapper(cell)
    #cell = tf.contrib.rnn.HighwayWrapper(cell)
    #cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=10, attn_size=100)
    
    if dropout and abs(args.dropout)>0:
        if args.dropout<0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=args._keep_prob)
        else:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=args._keep_prob)
        
        ''' variational_recurrent '''
        #variational_recurrent=True
        
    return cell, initial_state

def get_final_state(final_state, unit=None):
        if unit=='snt.lstm':
            return final_state[1]
        elif unit=='lstm':
            return final_state.c
        elif unit=='hlstm':
            return final_state[1]
        elif unit=='rda':
            return final_state.h
        elif unit.startswith('rwa'):
            try:
                return final_state.h
            except AttributeError:
                return final_state[2]
        else:
            return final_state
        
def collapse_final_state_layers(final_state_layers, unit=None):
    states = [get_final_state(s, unit) for s in final_state_layers]
    #return tf.concat(states, axis=1)
    return states[-1]
    
class DeepRNN(snt.AbstractModule):
    def __init__(self, 
                 FLAGS=None,
                 seq_len=None,
                 keep_prob=None,
                 forget_bias=0.,
                 pad='post',
                 name="deep_rnn"):
        super(DeepRNN, self).__init__(name=name)
        self.FLAGS = FLAGS
        self.rnn_size = FLAGS.rnn_dim
        self.num_layers = FLAGS.rnn_layers
        self.batch_size = FLAGS.batch_size
        self.dropout = FLAGS.dropout
        self.train_initial_state = FLAGS.train_initial_state
        self.unit = FLAGS.rnn_unit
        self._seq_len = seq_len
        self._keep_prob = keep_prob
        self.forget_bias = forget_bias
        self.pad = pad
        self.name = name
        
        with self._enter_variable_scope():
            if keep_prob is None:
                self._keep_prob = tf.placeholder_with_default(1.0-abs(self.FLAGS.dropout), shape=())
            if seq_len is None:
                self._seq_len = tf.placeholder(tf.int32, [None])# [self.batch_size]

    def _build(self, inputs):
        self.inputs = inputs#self.batch_size = tf.shape(inputs)[0]
        
        if self.num_layers > 1:
            cells = [create_rnn_cell(self, scope='layer{}'.format(i)) for i in range(self.num_layers)]
            cells, states = zip(*cells)
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            self._initial_rnn_state = tuple(states)
        else:
            cell, self._initial_rnn_state = create_rnn_cell(self)

        if self.dropout<0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self._keep_prob)
        
        sequence_length = self._seq_len if self.pad=='post' else None
        
        output, self._final_rnn_state = tf.nn.dynamic_rnn(cell,
                                                          inputs,
                                                          dtype=tf.float32,
                                                          sequence_length=sequence_length,
                                                          initial_state=self._initial_rnn_state
                                                          )
        
        tf.summary.histogram('{}_output'.format(self.name), output)
        
        return output#, final_rnn_state
    
    @property
    def seq_len(self):
        return self._seq_len
    
    @property
    def keep_prob(self):
        return self._keep_prob
    
    @property
    def initial_rnn_state(self):
        self._ensure_is_connected()
        return self._initial_rnn_state
    
    @property
    def final_rnn_state(self):
        self._ensure_is_connected()
        if self.num_layers > 1:
            return collapse_final_state_layers(self._final_rnn_state, self.unit)
        else:
            return get_final_state(self._final_rnn_state, self.unit)

###############################################################################

def padded_reverse(x, seq_len, batch_dim=0, seq_dim=1, pad='post'):
    if pad=='pre': x = tf.reverse(x,[seq_dim])
    x = tf.reverse_sequence(x, seq_len, batch_dim=batch_dim, seq_dim=seq_dim)
    if pad=='pre': x = tf.reverse(x,[seq_dim])
    return x

class DeepBiRNN(snt.AbstractModule):
    def __init__(self,
                 FLAGS=None,
                 seq_len=None,
                 keep_prob=None,
                 forget_bias=0.,
                 pad='post',
                 name="deep_bi_rnn"):
        super(DeepBiRNN, self).__init__(name=name)
        self.FLAGS = FLAGS
        self.rnn_size = FLAGS.rnn_dim
        self.num_layers = FLAGS.rnn_layers
        self.batch_size = FLAGS.batch_size
        self.dropout = FLAGS.dropout
        self.train_initial_state = FLAGS.train_initial_state
        self.unit = FLAGS.rnn_unit
        self._seq_len = seq_len
        self._keep_prob = keep_prob
        self.forget_bias = forget_bias
        self.pad = pad
        self.name = name
        
        with self._enter_variable_scope():
            if keep_prob is None:
                self._keep_prob = tf.placeholder_with_default(1.0-abs(self.FLAGS.dropout), shape=())
            if seq_len is None:
                self._seq_len = tf.placeholder(tf.int32, [None])# [self.batch_size]

    def _build(self, inputs):
        self.inputs = inputs
        
        with tf.variable_scope('fwd'):
            self.fwd_rnn = DeepRNN(FLAGS=self.FLAGS, seq_len=self._seq_len, keep_prob=self._keep_prob, pad=self.pad)
        with tf.variable_scope('bwd'):
            self.bwd_rnn = DeepRNN(FLAGS=self.FLAGS, seq_len=self._seq_len, keep_prob=self._keep_prob, pad=self.pad)
            
        fwd_outputs = self.fwd_rnn(inputs)
        bwd_outputs = self.bwd_rnn(padded_reverse(inputs, self._seq_len, pad=self.pad))
        
        outputs = tf.concat([fwd_outputs, bwd_outputs], axis=2)
        
        return outputs
    
    @property
    def seq_len(self):
        return self._seq_len
    
    @property
    def keep_prob(self):
        return self._keep_prob
    
    @property
    def final_rnn_state(self):
        self._ensure_is_connected()
#         fwd = collapse_final_state_layers(self.output_state_fw, self.unit)
#         bwd = collapse_final_state_layers(self.output_state_bw, self.unit)
        fwd = self.fwd_rnn.final_rnn_state
        bwd = self.bwd_rnn.final_rnn_state
        return tf.concat([fwd, bwd], axis=1)
    
###############################################################################
    
class DeepBiRNN_v1(snt.AbstractModule):
    def __init__(self,
                 FLAGS=None,
                 seq_len=None,
                 keep_prob=None,
                 forget_bias=0.,
                 pad='post',
                 name="deep_bi_rnn_v1"):
        super(DeepBiRNN_v1, self).__init__(name=name)
        self.FLAGS = FLAGS
        self.rnn_size = FLAGS.rnn_dim
        self.num_layers = FLAGS.rnn_layers
        self.batch_size = FLAGS.batch_size
        self.dropout = FLAGS.dropout
        self.train_initial_state = FLAGS.train_initial_state
        self.unit = FLAGS.rnn_unit
        self._seq_len = seq_len
        self._keep_prob = keep_prob
        self.forget_bias = forget_bias
        self.pad = pad
        self.name = name
        
        with self._enter_variable_scope():
            if keep_prob is None:
                self._keep_prob = tf.placeholder_with_default(1.0-abs(self.FLAGS.dropout), shape=())
            if seq_len is None:
                self._seq_len = tf.placeholder(tf.int32, [self.batch_size])

    def _build(self, inputs):
        self.inputs = inputs
        
        with tf.variable_scope('fwd'):
            cells_fw = [create_rnn_cell(self, scope='layer{}'.format(i)) for i in range(self.num_layers)]
        with tf.variable_scope('bwd'):
            cells_bw = [create_rnn_cell(self, scope='layer{}'.format(i)) for i in range(self.num_layers)]
        
        cells_fw, initial_states_fw = zip(*cells_fw)
        cells_bw, initial_states_bw = zip(*cells_bw)
        
        cells_fw = list(cells_fw)
        cells_bw = list(cells_bw)
        initial_states_fw = list(initial_states_fw)
        initial_states_bw = list(initial_states_bw)
        
        sequence_length = self._seq_len if self.pad=='post' else None
            
        outputs, self.output_state_fw, self.output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
                                                                                                             initial_states_fw=initial_states_fw,
                                                                                                             initial_states_bw=initial_states_bw,
                                                                                                             sequence_length=sequence_length,
                                                                                                             dtype=tf.float32)
        return outputs
    
    @property
    def seq_len(self):
        return self._seq_len
    
    @property
    def keep_prob(self):
        return self._keep_prob
    
    @property
    def final_rnn_state(self):
        self._ensure_is_connected()
        fwd = collapse_final_state_layers(self.output_state_fw, self.unit)
        bwd = collapse_final_state_layers(self.output_state_bw, self.unit)
        return tf.concat([fwd, bwd], axis=1)

''' simple sonnet wrapper for reshaping'''
class Reshape(snt.AbstractModule):
    def __init__(self,
                 batch_size=128,
                 num_unroll_steps=None,
                 name="reshape"):
        super(Reshape, self).__init__(name=name)
        self.batch_size = batch_size
        self.num_unroll_steps = num_unroll_steps
    
    def _build(self, inputs):
        dim = inputs.get_shape().as_list()[1]
        return tf.reshape(inputs, [self.batch_size, self.num_unroll_steps, dim])

def mellowmax(x, axis=-1, omega=0.01, mask=None):
    #n = x.shape[axis]
    #n = x.get_shape().as_list()[axis]
    omega = tf.Variable(tf.constant(omega, dtype=tf.float32), trainable=False)
    
    #ans = (F.logsumexp(omega * x, axis=axis) - np.log(n)) / omega
    #ans = (tf.reduce_logsumexp(omega * x, axis=axis)  - tf.log(n)) / omega
    
    ex = tf.exp(omega * x)
    if mask!=None:
        ex = tf.multiply(ex, mask)
    
    n = tf.reduce_sum(mask, axis=axis, keep_dims=True)
    ans = (tf.log(tf.reduce_sum(ex, axis=axis, keep_dims=True)) - tf.log(n)) / omega
    
    return ans

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
        #output = tf.tensordot(inputs, W, axes=1) + b #
        output = tf.einsum(q, inputs, W) + b
        
        #tf.summary.histogram('{}_output'.format(self.name), output)
        return output

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

class Attention(snt.AbstractModule):
    def __init__(self, FLAGS,
                 final_rnn_state=None,
                 name="attention"):
        super(Attention, self).__init__(name=name)
        self.FLAGS = FLAGS
        self.final_rnn_state = final_rnn_state
    
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
        
        ''' softmax '''
        #T = tf.get_variable('T', shape=[1], initializer=tf.constant_initializer(1.0), dtype=tf.float32, trainable=True)
        alpha = mc.softmask(beta, mask=mask)
        
        ''' attn pooling '''
        ##output = tf.reduce_sum(inputs * tf.expand_dims(alpha, -1), 1)
        w = inputs * tf.expand_dims(alpha, -1)
        #w = v * tf.expand_dims(alpha, -1)
        
        output = tf.reduce_sum(w, 1)
        
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
                
#         ps(inputs, 'inputs')
#         ps(u, 'u')
#         ps(v, 'v')
#         ps(w, 'w')
#         ps(mask, 'mask')
#         ps(beta, 'beta')
#         ps(alpha, 'alpha')
#         ps(output, 'output')
        
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

''' simple sonnet wrapper for aggregation'''
class Aggregation(snt.AbstractModule):
    def __init__(self,
                 FLAGS=None,
                 seq_len=None,
                 final_rnn_state=None,
                 name="aggregation"):
        super(Aggregation, self).__init__(name=name)
        self.FLAGS = FLAGS
        self.seq_len = seq_len
        self.final_rnn_state = final_rnn_state
        self.name = name
    
    def dynamic_mean(self, x, n):
        m = tf.reduce_sum(x, axis=1)
        return m / tf.expand_dims(tf.cast(n, tf.float32), 1)
    
    def _build(self, inputs):
        if self.FLAGS.att_type>-1:
            
            if self.FLAGS.att_type==0:
                self._attn_module = Attention(self.FLAGS, final_rnn_state=self.final_rnn_state)
            elif self.FLAGS.att_type==1:
                self._attn_module = Attn1(self.FLAGS)
            elif self.FLAGS.att_type==2:
                self._attn_module = Attn2(self.FLAGS, final_rnn_state=self.final_rnn_state, seq_len=self.seq_len)
            
            output = self._attn_module(inputs)
            
        elif self.FLAGS.mean_pool: ## use mean pooled rnn states
            if self.seq_len==None:
                #output = tf.reduce_mean(inputs, axis=1)
                output = tf.reduce_mean(inputs, axis=-2)
            else:
                output = self.dynamic_mean(inputs, self.seq_len)
            
        else: ## otherwise just use final rnn state
            output = self.final_rnn_state
#             output = tf.gather_nd(inputs, tf.stack([tf.range(self.FLAGS.batch_size), self.seq_len-1], axis=1))
        
        tf.summary.histogram('{}_output'.format(self.name), output)
        
        return output
    
    @property
    def attn_module(self):
        self._ensure_is_connected()
        try:
            return self._attn_module
        except AttributeError:
            return None
        
    @property
    def z(self):
        self._ensure_is_connected()
        try:
            return self.attn_module.z
        except AttributeError:
            return []

# class Model(snt.AbstractModule):
class FlatModel(snt.AbstractModule):
    def __init__(self,
                 FLAGS=None,
                 embed_word=True,
                 embed_matrix=None,
                 max_word_length=None,
                 char_vocab=None,
                 inputs=None,
                 name="FlatModel"):
        super(FlatModel, self).__init__(name=name)
        self.FLAGS = FLAGS
        self.embed_word = embed_word
        self.embed_matrix = embed_matrix
        self.max_word_length = max_word_length
        self.char_vocab = char_vocab
        self.z = {}
    
    def build(self, inputs):
        
        if self.embed_word:
            word_embed_module = snt.Embed(existing_vocab=self.embed_matrix, trainable=True)
            outputs = word_embed_module(inputs)
        else:
            ## char embed ##
            char_embed_module = CharEmbed(vocab_size=self.char_vocab.size,#char_vocab.size,
                                          embed_dim=self.FLAGS.char_embed_size, 
                                          max_word_length=self.max_word_length,
                                          name='char_embed_b')
            outputs1 = char_embed_module(inputs)
            
            ## tdnn ##
            tdnn_module = TDNN(self.FLAGS.kernel_widths, 
                               self.FLAGS.kernel_features, 
                               initializer=0, 
                               name='TDNN')
            outputs2 = tdnn_module(outputs1)
            
            ## reshape ##
            num_unroll_steps = tf.shape(inputs)[1]
            reshape_module = Reshape(batch_size=self.FLAGS.batch_size,
                                     num_unroll_steps=num_unroll_steps)
            outputs = reshape_module(outputs2)
            
            #dim = outputs2.get_shape().as_list()[1]
            #outputs = tf.reshape(outputs2, [self.FLAGS.batch_size, num_unroll_steps, dim])
            
            self.z['inputs'] = inputs
            self.z['num_unroll_steps'] = num_unroll_steps
            self.z['outputs1'] = outputs1
            self.z['outputs2'] = outputs2
            self.z['outputs'] = outputs
            
            
        ##################################################
        rnn_word = (DeepBiRNN_v1 if self.FLAGS.wpad=='post' else DeepBiRNN) if self.FLAGS.bidirectional else DeepRNN
        self._rnn_module = rnn_word(FLAGS=self.FLAGS, pad=self.FLAGS.wpad)
        outputs = self._rnn_module(outputs)
        
        ##################################################
        
        self._agg_module = Aggregation(self.FLAGS, 
                                       seq_len=self._rnn_module.seq_len, 
                                       final_rnn_state=self._rnn_module.final_rnn_state)
        outputs = self._agg_module(outputs)
        
        ##################################################
#         dim = self.FLAGS.rnn_dim
#         #dim = tf.shape(outputs)[-1]
#         #dim = outputs.get_shape().as_list()[-1]
  
        w_init, b_init = default_initializers(std=self.FLAGS.model_std, bias=self.FLAGS.model_b)
        lin_module = snt.Linear(output_size=1, initializers={ 'w':w_init, 'b':b_init })
        outputs = lin_module(outputs)
        ##################################################
        
        ## tanh
        outputs = tf.nn.tanh(outputs)
        
        return outputs
              
    def _build(self, inputs):
        outputs = self.build(inputs)
        return outputs
    
    @property
    def rnn_module(self):
        self._ensure_is_connected()
        return self._rnn_module
    
    @property
    def seq_len(self):
        return self.rnn_module.seq_len
    
    @property
    def keep_prob(self):
        return self.rnn_module.keep_prob
    
    @property
    def final_rnn_state(self):
        return self.rnn_module.final_rnn_state
    
    @property
    def agg_module(self):
        self._ensure_is_connected()
        return self._agg_module
    
    @property
    def z_attn(self):
        return self.agg_module.z


##############################################################################################
''' https://github.com/davidsvaughn/hierarchical-attention-networks/blob/master/HAN_model.py
'''
import tensorflow.contrib.layers as layers
import nlp.tf_tools.model_components as mc

class Model(snt.AbstractModule):
# class HANModel(snt.AbstractModule):
    def __init__(self,
                 FLAGS=None,
                 embed_word=True,
                 embed_matrix=None,
                 max_word_length=None,
                 char_vocab=None,
                 name="Model"):
        super(Model, self).__init__(name=name)
        self.FLAGS = FLAGS
        self.embed_word = embed_word
        self.embed_matrix = embed_matrix
        self.max_word_length = max_word_length
        self.char_vocab = char_vocab
        self.z_word_attn = []
        self.z_sent_attn = []
        self.z = {}
        #######
        self.rnn_size = FLAGS.rnn_dim
        self.unit = FLAGS.rnn_unit
        self.forget_bias = FLAGS.forget_bias
        self.train_initial_state = FLAGS.train_initial_state
    
    def build(self, inputs):
        CELL = tf.nn.rnn_cell.GRUCell
        #CELL = tf.nn.rnn_cell.BasicLSTMCell
        
        ''' SETUP '''
        self.keep_prob = tf.placeholder_with_default(1.0-abs(self.FLAGS.dropout), shape=())
        #self.seq_len = tf.placeholder(tf.int32, [self.FLAGS.batch_size])
        
        # [document]
        self.sentence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sentence_lengths')
        
        # [document x sentence]
        self.word_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_lengths')
        
        (self.document_size,
        self.sentence_size,
        self.word_size) = tf.unstack(tf.shape(inputs))[:3]
        
        word_bs = self.document_size * self.sentence_size
        char_bs = self.document_size * self.sentence_size * self.word_size
        
        word_level_lengths = tf.reshape(self.word_lengths, [word_bs])
        
        ############################################################
        
        ''' EMBEDDING (word/char) '''
        if self.embed_word:
            word_embed_module = snt.Embed(existing_vocab=self.embed_matrix, trainable=True)
            outputs = word_embed_module(inputs)
            outputs = tf.reshape(outputs, [word_bs, self.word_size, self.FLAGS.embed_dim])
            
        else:
            ## char embed ##
            char_embed_module = CharEmbed(vocab_size=self.char_vocab.size,#char_vocab.size,
                                          embed_dim=self.FLAGS.char_embed_size, 
                                          max_word_length=self.max_word_length,
                                          name='char_embed_b')
            outputs = char_embed_module(inputs)
            
            ######################################################
            ''' gather '''
            bool_idx =  tf.not_equal( tf.reduce_max( tf.reshape(inputs, [-1, self.max_word_length]), axis=-1), tf.constant( 0, dtype=tf.int32 ))
            idx = tf.cast( tf.where(bool_idx), tf.int32 )
            outputs = tf.gather_nd(outputs, idx)
            
            ######################################################
            
            ## tdnn ##
            tdnn_module = TDNN(self.FLAGS.kernel_widths, 
                               self.FLAGS.kernel_features, 
                               initializer=0, 
                               name='TDNN')
            outputs = tdnn_module(outputs)
            dim = outputs.get_shape().as_list()[-1]#dim = tf.shape(outputs)[-1]
            
            ######################################################
            ''' scatter '''
            output_shape = tf.cast( [ char_bs, dim ] , tf.int32 )
            outputs = tf.scatter_nd(indices=idx,
                                    updates=outputs,
                                    shape=output_shape)
            
            ######################################################
            
            ## reshape ##
            outputs = tf.reshape(outputs, [word_bs, self.word_size, dim])
        
        ##########################################################
        
        ##### word_level #####
        word_level_inputs = outputs
        
        if self.FLAGS.sparse_words:
            sps_idx = tf.where(word_level_lengths>0)
            word_level_inputs = tf.gather_nd(word_level_inputs, sps_idx)
            word_level_lengths = tf.gather_nd(word_level_lengths, sps_idx)
    #         self.z['sps_idx'] = sps_idx
    #         self.z['word_level_inputs'] = word_level_inputs
    #         self.z['word_level_lengths'] = word_level_lengths
        
        #### GATHER_ND ##########################
        
        with tf.variable_scope('word') as scope:
            
            # rnn #
            
            if self.FLAGS.rnn_new:                
                #word_cell = CELL(self.FLAGS.rnn_dim)
                word_cell, _ = create_rnn_cell(self, scope=scope, dropout=False, batch_size=tf.shape(word_level_inputs)[0])
                
                word_encoder_output, _ = mc.bidirectional_rnn(
                    word_cell, word_cell,
                    word_level_inputs, 
                    word_level_lengths,
                    pad=self.FLAGS.wpad,
                    scope=scope)
            
            else:
                #rnn_word = (DeepBiRNN_v1 if self.FLAGS.wpad=='post' else DeepBiRNN) if self.FLAGS.bidirectional else DeepRNN
                rnn_word = DeepBiRNN if self.FLAGS.bidirectional else DeepRNN
                rnn_module_word = rnn_word(FLAGS=self.FLAGS,
                                           seq_len=word_level_lengths,
                                           keep_prob=self.keep_prob,
                                           pad=self.FLAGS.wpad)
                word_encoder_output = rnn_module_word(word_level_inputs)
                
            self.z['word_encoder_output_shape'] = tf.shape(word_encoder_output)
            
            # attn #
            with tf.variable_scope('attention') as scope:
                word_level_output = mc.task_specific_attention(word_encoder_output, self.FLAGS.att_size, scope=scope)
                if U.is_sequence(word_level_output): word_level_output, self.z_word_attn = word_level_output
                ## or ##
#                 attn_word = Attention(self.FLAGS); word_level_output = attn_word(word_encoder_output)
            
            # dropout #
            with tf.variable_scope('dropout'):
                word_level_output = layers.dropout(
                    word_level_output, 
                    keep_prob=self.keep_prob)
        
        #### SCATTER_ND #########################
        
        if self.FLAGS.sparse_words:
            output_shape = tf.cast( [ word_bs, tf.shape(word_level_output)[-1]] , tf.int64 )
            word_level_output = tf.scatter_nd(indices=sps_idx,
                                              updates=word_level_output,
                                              shape=output_shape)
#             self.z['output_shape'] = output_shape
#             self.z['word_output'] = word_output
#             self.z['word_level_output'] = word_level_output

            ###########################
            #dim = tf.shape(word_level_output)[-1]
            #dim = word_level_output.get_shape().as_list() [-1]
            dim = self.FLAGS.rnn_dim * (2 if self.FLAGS.bidirectional else 1)
        else:
            dim = word_level_output.get_shape().as_list() [-1]
        
        self.z['dim'] = tf.cast( dim , tf.int32 )
        
        ##### sentence_level ##########################################################
        
        sentence_inputs_shape = tf.cast( [self.document_size, self.sentence_size, dim] , tf.int32 )
        sentence_inputs = tf.reshape( word_level_output, shape=sentence_inputs_shape )
        sentence_level_lengths = self.sentence_lengths
        
        self.z['sentence_inputs_shape'] = sentence_inputs_shape
        
        with tf.variable_scope('sentence') as scope:
            
            # rnn #
            
            if self.FLAGS.rnn_new:
                #sentence_cell = CELL(self.FLAGS.rnn_dim)
                sentence_cell, _ = create_rnn_cell(self, scope=scope, dropout=False, batch_size=tf.shape(sentence_inputs)[0])
                
                sentence_encoder_output, _ = mc.bidirectional_rnn(
                    sentence_cell, sentence_cell,
                    sentence_inputs, 
                    sentence_level_lengths,
                    pad=self.FLAGS.spad,
                    scope=scope)
            
            else:
                #rnn_sent = (DeepBiRNN_v1 if self.FLAGS.spad=='post' else DeepBiRNN) if self.FLAGS.bidirectional else DeepRNN
                rnn_sent = DeepBiRNN if self.FLAGS.bidirectional else DeepRNN
                rnn_module_sent = rnn_sent(FLAGS=self.FLAGS,
                                           seq_len=sentence_level_lengths,
                                           keep_prob=self.keep_prob,
                                           pad=self.FLAGS.spad)
                sentence_encoder_output = rnn_module_sent(sentence_inputs)
            
            # attn #
            with tf.variable_scope('attention') as scope:
                sentence_level_output = mc.task_specific_attention(sentence_encoder_output, self.FLAGS.att_size, scope=scope)
                if U.is_sequence(sentence_level_output): sentence_level_output, self.z_sent_attn = sentence_level_output
                ## or ##
#                 attn_sent = Attention(self.FLAGS); sentence_level_output = attn_sent(sentence_encoder_output)
            
            # dropout #
            with tf.variable_scope('dropout'):
                sentence_level_output = layers.dropout(
                    sentence_level_output, 
                    keep_prob=self.keep_prob)
        
        
        ##################################################
        
        #sentence_level_output.set_shape(tf.TensorShape([self.FLAGS.batch_size, dim]))
        
        w_init, b_init = default_initializers(std=self.FLAGS.model_std, bias=self.FLAGS.model_b)
        lin_module = snt.Linear(output_size=1, initializers={ 'w':w_init, 'b':b_init })
        outputs = lin_module(sentence_level_output)
        
        ##################################################
        
        ## tanh
        outputs = tf.nn.tanh(outputs)
        
        return outputs
              
    def _build(self, inputs):
        outputs = self.build(inputs)
        return outputs
    
#     @property
#     def seq_len(self):
#         return self.rnn_module.seq_len
    
   
##################################################################################################
##################################################################################################
##### EXPERIMENTAL ###############################################################################
# class RHN_Cell(snt.RNNCore):
#     def __init__(self, output_size,
#                  num_layers=5, 
#                  use_inputs_on_each_layer=False,
#                  use_kronecker_reparameterization=False,
#                  initializers=None,
#                  name="rhn_cell"):
#         super(RHN_Cell, self).__init__(name=name)
#         self._output_size = output_size
#         self._num_layers=num_layers
#         self._use_inputs_on_each_layer=use_inputs_on_each_layer
#         self._use_kronecker_reparameterization=use_kronecker_reparameterization
#     
#     @classmethod
#     def get_possible_initializer_keys():
#         return None
#     
#     @property
#     def output_size(self):
#         """Returns a description of the output size, without batch dimension."""
#         return tf.TensorShape([self._output_size])
#         
#     def _build(self, inputs, state):
#         cell = rnn_cell_modern.HighwayRNNCell(self._output_size, 
#                                               num_highway_layers=self._num_layers,
#                                               use_inputs_on_each_layer=self._use_inputs_on_each_layer,
#                                               use_kronecker_reparameterization=self._use_kronecker_reparameterization)
# #         cell = RHNCell(num_units=self._output_size,
# #                        in_size=self._in_size,
# #                        is_training=True,
# #                        depth=self._num_layers)
#         return cell(inputs, state)

# class RWA_Cell(snt.RNNCore):
#     def __init__(self, num_units,
#                  name="rwa_cell"):
#         super(RWA_Cell, self).__init__(name=name)
#         self._num_units = num_units
#     
#     @classmethod
#     def get_possible_initializer_keys():
#         return None
#     
#     @property
#     def output_size(self):
#         return self._num_units
# 
#     @property
#     def state_size(self):
#         return (self._num_units, self._num_units, self._num_units, self._num_units)
#         
#     def _build(self, inputs, state):
#         cell = RWACell(self._num_units)
#         return cell(inputs, state)

##################################################################################################
##################################################################################################
##### OLD STUFF ##################################################################################
    
# class MultiLSTM(snt.AbstractModule):
#     def __init__(self, rnn_size,
#                  num_layers=1,
#                  batch_size=128, 
#                  dropout=0.0,
#                  forget_bias=0.0,
#                  seq_len=None,
#                  use_skip_connections=False,
#                  use_peepholes=False,
#                  train_initial_state=False,
#                  #initializer=None,
#                  name="multi_lstm"):
#         super(MultiLSTM, self).__init__(name=name)
#         self.rnn_size = rnn_size
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#         self.dropout = dropout
#         self.forget_bias = forget_bias
#         self._seq_len = seq_len
#         self.use_skip_connections = use_skip_connections
#         self.use_peepholes = use_peepholes
#         self.train_initial_state = train_initial_state
#         #self.initializer = initializer
#         
#         if use_skip_connections:
#             self.dropout = 0.0
#         
#         with self._enter_variable_scope():
#             self._keep_prob = tf.placeholder_with_default(1.0-self.dropout, shape=())
#             if seq_len is None:
#                 self._seq_len = tf.placeholder(tf.int32, [batch_size])
#             
#             cell = snt.LSTM
#             #cell = lm.RWA_Cell
#             #cell = lm.RHN_Cell
#             
#             self.subcores = [
#                 cell(self.rnn_size, 
#                     forget_bias=self.forget_bias,
#                     use_peepholes=self.use_peepholes,
#                     name="multi_lstm_subcore_{}".format(i+1),
#                      
#                      #initializers=init_dict(self.initializer,
#                      #                       cell.get_possible_initializer_keys()),
#                      #use_batch_norm_h=True, use_batch_norm_x=True, use_batch_norm_c=True, #is_training=self._is_training,
#                      )
#                 for i in range(self.num_layers)
#             ]
# 
#     def _build(self, inputs):
#         
#         ## DROPOUT ##
#         if self.dropout > 0.0:
#             dropouts = [Dropout(keep_prob=self._keep_prob) for i in range(self.num_layers)]
#             self.subcores = interleave(self.subcores, dropouts)
#         
#         if len(self.subcores) > 1:
#             self.core = snt.DeepRNN(self.subcores, name="multi_lstm_core", skip_connections=self.use_skip_connections)
#         else:
#             self.core = self.subcores[0]
# 
#         if self.train_initial_state:
#             self._initial_rnn_state = self.core.initial_state(self.batch_size, tf.float32, trainable=True)
#             #self._initial_rnn_state = snt.TrainableInitialState(self.core.initial_state(self.batch_size, tf.float32))()
#         else:
#             self._initial_rnn_state = self.core.zero_state(self.batch_size, tf.float32)
#         
#         output, final_rnn_state = tf.nn.dynamic_rnn(self.core,
#                                                     inputs,
#                                                     dtype=tf.float32,
#                                                     sequence_length=self._seq_len,
#                                                     initial_state=self._initial_rnn_state
#                                                     )
#     
#         return output, final_rnn_state
#     
#     @property
#     def seq_len(self):
#         return self._seq_len
#     
#     @property
#     def keep_prob(self):
#         return self._keep_prob
#     
#     @property
#     def initial_rnn_state(self):
#         self._ensure_is_connected()
#         return self._initial_rnn_state


###############################################################################


# class BiMultiLSTM(snt.AbstractModule):
#     def __init__(self, rnn_size,
#                  num_layers=1,
#                  batch_size=128, 
#                  dropout=0.0,
#                  forget_bias=0.0,
#                  seq_len=None,
#                  #initializer=None,
#                  name="bi_multi_lstm"):
#         super(BiMultiLSTM, self).__init__(name=name)
#         self.rnn_size = rnn_size
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#         self.dropout = dropout
#         self.forget_bias = forget_bias
#         self._seq_len = seq_len
#         #self.initializer = initializer
#         
#         with self._enter_variable_scope():
#             self._keep_prob = tf.placeholder_with_default(1.0, shape=())
#             if seq_len is None:
#                 self._seq_len = tf.placeholder(tf.int32, [batch_size])
# 
#     def _build(self, inputs):
#         cell = snt.LSTM
#          
#         cells_fw = [cell(self.rnn_size, forget_bias=self.forget_bias, name="bi_multi_lstm_fw_{}".format(i+1))
#                     for i in range(self.num_layers)]
#         cells_bw = [cell(self.rnn_size, forget_bias=self.forget_bias, name="bi_multi_lstm_bw_{}".format(i+1))
#                     for i in range(self.num_layers)]
#          
#         outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
#                                                                                                    sequence_length=self._seq_len,
#                                                                                                    dtype=tf.float32,
#                                                                                                    scope='BiMultiLSTM')
#         return outputs, (output_state_fw, output_state_bw)
#     
#     @property
#     def seq_len(self):
#         return self._seq_len
#     
#     @property
#     def keep_prob(self):
#         return self._keep_prob
    
    
    
    
    ###################################################
    # MultiLSTM skip connections
            
#     ## SKIP CONNECTIONS ## rnn_shakespeare.py
#     if self.use_skip_connections:
#         skips = []
#         embed_dim = inputs.get_shape().as_list()[-1]
#         current_input_shape = embed_dim
#         for lstm in self.subcores:
#             input_shape = tf.TensorShape([current_input_shape])
#             skip = snt.SkipConnectionCore(
#                 lstm,
#                 input_shape=input_shape,
#                 name="skip_{}".format(lstm.module_name))
#             skips.append(skip)
#             # SkipConnectionCore concatenates the input with the output..
#             # ...so the dimensionality increases with depth.
#             current_input_shape += self.rnn_size
#         self.subcores = skips

    ###################################################
    
    # old BiMultiLSTM build() fxn:
#     def _build(self, inputs):
#         output = inputs
#         cell = snt.LSTM
#          
#         for i in range(self.num_layers):
# #             lstm_fw = tf.nn.rnn_cell.LSTMCell(self.rnn_size, forget_bias=self.forget_bias)
# #             lstm_bw = tf.nn.rnn_cell.LSTMCell(self.rnn_size, forget_bias=self.forget_bias)
#             lstm_fw = cell(self.rnn_size, forget_bias=self.forget_bias, name="bi_multi_lstm_fw_{}".format(i+1))
#             lstm_bw = cell(self.rnn_size, forget_bias=self.forget_bias, name="bi_multi_lstm_bw_{}".format(i+1))
#      
#             _initial_state_fw = lstm_fw.zero_state(self.batch_size, tf.float32)
#             _initial_state_bw = lstm_bw.zero_state(self.batch_size, tf.float32)
#      
#             output, _states = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, output, 
#                                                               initial_state_fw=_initial_state_fw,
#                                                               initial_state_bw=_initial_state_bw,
#                                                               sequence_length=self._seq_len,
#                                                               scope='BLSTM_'+str(i+1))
#             output = tf.concat(output, 2)
#         return output, _states