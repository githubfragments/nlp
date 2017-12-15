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

from nlp.tf_tools.attn1 import Attn1
from nlp.tf_tools.attn2 import Attn2

# from nlp.rwa.RWACell import RWACell         # <-- jostmey
from nlp.rwa.rwa_cell import RWACell

from nlp.rwa.rda_cell import RDACell

# from nlp.ran.ran import RANCell
from nlp.ran.ran_cell import RANCell

from nlp.tensorflow_with_latest_papers import rnn_cell_modern
# from nlp.recurrent_highway_networks.rhn import RHNCell

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
    
    # inputs shape = [batch_size, num_unroll_steps, max_word_length]
    # inputs = char ids, output = input_embedded
    def _build(self, inputs):
        output = self._char_embedding(inputs)
        
        max_word_length = self._max_word_length
        if max_word_length==None:
            max_word_length = tf.shape(inputs)[2]
            
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
    elif args.unit=='rwa_bn':
        rnn = RWACell
        kwargs = { 'normalize':True }
    elif args.unit=='rda':
        rnn = RDACell
    elif args.unit=='rda_bn':
        rnn = RDACell
        kwargs = { 'normalize':True }
    elif args.unit=='rhn':
        #rnn = RHN_Cell
        rnn = rnn_cell_modern.HighwayRNNCell
        kwargs = { 'num_highway_layers' : args.FLAGS.rhn_highway_layers,
                   'use_inputs_on_each_layer' : args.FLAGS.rhn_inputs,
                   'use_kronecker_reparameterization' : args.FLAGS.rhn_kronecker }
    return rnn, kwargs

def get_initial_state(cell, args):
    if args.train_initial_state:
        if args.unit.startswith('snt'):
            return cell.initial_state(args.batch_size, tf.float32, trainable=True)
        print('TRAINABLE INITIAL STATE NOT YET IMPLEMENTED FOR: {} !!!'.format(args.unit))
#         else:
#             initializer = r2rt.make_variable_state_initializer()
#             return r2rt.get_initial_cell_state(cell, initializer, args.batch_size, tf.float32)
    return cell.zero_state(args.batch_size, tf.float32)

def create_rnn_cell(args):
    rnn, kwargs = rnn_unit(args)
    cell = rnn(args.rnn_size, **kwargs)
    
    initial_state = get_initial_state(cell, args)
    
    #cell = tf.contrib.rnn.ResidualWrapper(cell)
    #cell = tf.contrib.rnn.HighwayWrapper(cell)
    #cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=10, attn_size=100)
    
    if abs(args.dropout)>0:
        if args.dropout<0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=args._keep_prob)
        else:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=args._keep_prob)

        #variational_recurrent=True
        
    return cell, initial_state

def get_final_state(final_state, unit=None):
        if unit=='snt.lstm':
            return final_state[1]
        elif unit=='lstm':
            return final_state.c
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
                 forget_bias=0.,
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
        self.forget_bias = forget_bias
        self.name = name
        
        with self._enter_variable_scope():
            self._keep_prob = tf.placeholder_with_default(1.0-abs(self.dropout), shape=())
            if seq_len is None:
                self._seq_len = tf.placeholder(tf.int32, [self.batch_size])

    def _build(self, inputs):
        if self.num_layers > 1:
            cells = [create_rnn_cell(self) for _ in range(self.num_layers)]
            cells, states = zip(*cells)
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            self._initial_rnn_state = tuple(states)
        else:
            cell, self._initial_rnn_state = create_rnn_cell(self)

        if self.dropout<0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self._keep_prob)
        
        output, self._final_rnn_state = tf.nn.dynamic_rnn(cell,
                                                          inputs,
                                                          dtype=tf.float32,
                                                          sequence_length=self._seq_len,
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

class DeepBiRNN(snt.AbstractModule):
    def __init__(self,
                 FLAGS=None,
                 seq_len=None,
                 forget_bias=0.,
                 name="deep_bi_rnn"):
        super(DeepBiRNN, self).__init__(name=name)
#         self.rnn_size = rnn_size
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#         self.dropout = abs(dropout)
#         self.forget_bias = forget_bias
#         self._seq_len = seq_len
#         self.train_initial_state = train_initial_state
#         self.rhn_highway_layers = rhn_highway_layers
#         self.unit = unit
#         self.name = name
        self.FLAGS = FLAGS
        self.rnn_size = FLAGS.rnn_dim
        self.num_layers = FLAGS.rnn_layers
        self.batch_size = FLAGS.batch_size
        self.dropout = FLAGS.dropout
        self.train_initial_state = FLAGS.train_initial_state
        self.unit = FLAGS.rnn_unit
        self._seq_len = seq_len
        self.forget_bias = forget_bias
        self.name = name
        
        
        with self._enter_variable_scope():
            self._keep_prob = tf.placeholder_with_default(1.0, shape=())
            if seq_len is None:
                self._seq_len = tf.placeholder(tf.int32, [self.batch_size])

    def _build(self, inputs):
        cells_fw = [create_rnn_cell(self) for i in range(self.num_layers)]
        cells_bw = [create_rnn_cell(self) for i in range(self.num_layers)]
        
        cells_fw, initial_states_fw = zip(*cells_fw)
        cells_bw, initial_states_bw = zip(*cells_bw)
        
        cells_fw = list(cells_fw)
        cells_bw = list(cells_bw)
        initial_states_fw = list(initial_states_fw)
        initial_states_bw = list(initial_states_bw)
         
        outputs, self.output_state_fw, self.output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
                                                                                                             initial_states_fw=initial_states_fw,
                                                                                                             initial_states_bw=initial_states_bw,
                                                                                                             sequence_length=self._seq_len,
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
        #return (self.output_state_fw, self.output_state_bw)
        if self.num_layers > 1:
            fwd = collapse_final_state_layers(self.output_state_fw, self.unit)
            bwd = collapse_final_state_layers(self.output_state_bw, self.unit)
        else:
            fwd = get_final_state(self.output_state_fw, self.unit)
            bwd = get_final_state(self.output_state_bw, self.unit)
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

def softmask(x, axis=-1, mask=None):
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    x = x - x_max
    ex = tf.exp(x)
    
    if mask!=None:
        ex = tf.multiply(ex, mask)
        
    es = tf.reduce_sum(ex, axis=axis, keep_dims=True)
    return ex/es

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
    
    def build2(self, inputs):
        A = self.FLAGS.att_size
        D = inputs.shape[-1].value # D value - hidden size of the RNN layer
        
        mask = tf.cast(tf.abs(tf.reduce_sum(inputs, axis=2))>0, tf.float32)
        
        '''# Linear Layer '''
        w_init = tf.truncated_normal_initializer(stddev=self.FLAGS.attn_std)
        if FLAGS.attn_b != None:
            b_init = tf.constant_initializer(FLAGS.attn_b)
        else:
            b_init = tf.truncated_normal_initializer(stddev=self.FLAGS.attn_std)
        
#         ## Linear 1
#         W = tf.get_variable('W', shape=[D, A], initializer=w_init)
#         b = tf.get_variable('b', shape=[A], initializer=b_init)
#         z = tf.tensordot(inputs, W, axes=1) + b #
        
        ## Linear 2
        lin_module = snt.Linear(output_size=A, initializers={'w':w_init, 'b':b_init})
        z = lin_module(inputs)
        
        
        v = tf.tanh(z)
        
        #####
        u = tf.get_variable('u', shape=[A], initializer=tf.truncated_normal_initializer(stddev=self.FLAGS.attn_std))
        
        ''' ??? '''
        M = tf.get_variable('M', shape=[D, A], initializer=tf.truncated_normal_initializer(stddev=self.FLAGS.attn_std))
        m = tf.get_variable('m', shape=[A], initializer=tf.truncated_normal_initializer(stddev=self.FLAGS.attn_std))
        p = tf.get_variable('p', shape=[A], initializer=tf.truncated_normal_initializer(stddev=self.FLAGS.attn_std))
        q = tf.tanh(tf.tensordot(self.final_rnn_state, M, axes=1) + m)
        s = tf.get_variable('s', shape=[A], initializer=tf.truncated_normal_initializer(stddev=self.FLAGS.attn_std))
        p = tf.tensordot(q, s, axes=1)
        
        ''' softmax '''
        beta = tf.tensordot(v, u, axes=1)
        
        beta = beta * tf.expand_dims(p, 1)
        
        alpha = softmask(beta, mask=mask)
        
        ''' attn pooling '''
        output = tf.reduce_sum(inputs * tf.expand_dims(alpha, -1), 1)
        
        return output
    
    def build1(self, inputs):
        A = self.FLAGS.att_size
        D = inputs.shape[-1].value # D value - hidden size of the RNN layer
        mask = tf.cast(tf.abs(tf.reduce_sum(inputs, axis=2))>0, tf.float32)
 
        w_init, b_init = default_initializers(std=self.FLAGS.attn_std, bias=self.FLAGS.attn_b)
            
        ''' Linear Layer '''
        lin_module = Linear(output_dim=A, w_init=w_init, b_init=b_init)
        z = lin_module(inputs)
        
        ''' softmax '''
        v = tf.tanh(z)
        u = tf.get_variable('u', shape=[A], initializer=w_init)
         
        #beta = tf.tensordot(v, u, axes=1)
        beta = tf.einsum('ijk,k->ij', v, u)
        
        alpha = softmask(beta, mask=mask)
        
        ''' attn pooling '''
        output = tf.reduce_sum(inputs * tf.expand_dims(alpha, -1), 1)
        
        #######
        ps(inputs, 'inputs')
        ps(z, 'z')
        ps(v, 'v')
        ps(beta, 'beta')
        ps(alpha, 'alpha')
        ps(output, 'output')
        
        return output
    
    def _build(self, inputs):
        return self.build1(inputs)
        #return self.build2(inputs)
    

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
            output = self.dynamic_mean(inputs, self.seq_len)
#             output = tf.reduce_mean(inputs, axis=1)
            
        else: ## otherwise just use final rnn state
            #output = tf.gather_nd(inputs, tf.stack([tf.range(self.FLAGS.batch_size), self.seq_len-1], axis=1))
            output = self.final_rnn_state
        
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
    def outputs(self):
        self._ensure_is_connected()
        try:
            return self.attn_module.outputs
        except AttributeError:
            return []
        
    @property
    def N(self):
        self._ensure_is_connected()
        try:
            return self.attn_module.outputs[-2]
        except AttributeError:
            return []


class Model(snt.AbstractModule):
    def __init__(self,
                 FLAGS=None,
                 embed_word=True,
                 embed_matrix=None,
                 max_word_length=None,
                 char_vocab=None,
                 inputs=None,
                 name="Model"):
        super(Model, self).__init__(name=name)
        self.FLAGS = FLAGS
        self.embed_word = embed_word
        self.embed_matrix = embed_matrix
        self.max_word_length = max_word_length
        self.char_vocab = char_vocab
    
    def build_1(self, inputs, layers):
        model = snt.Sequential(layers)
        output = model(inputs)
        return output
    
    def build_2(self, inputs, layers):
        outputs = []
        net = inputs
        for layer in layers:
            if isinstance(net, tuple):
                net = layer(*net)
            else:
                net = layer(net)
            outputs.append(net)
        return tuple(outputs)
    
    def build(self, inputs):
        outputs = inputs
        
        if self.embed_word:
            word_embed_module = snt.Embed(existing_vocab=self.embed_matrix, trainable=True)
            outputs = word_embed_module(outputs)
        else:
            ## char embed ##
            char_embed_module = CharEmbed(vocab_size=self.char_vocab.size,#char_vocab.size,
                                          embed_dim=self.FLAGS.char_embed_size, 
                                          max_word_length=self.max_word_length,
                                          name='char_embed_b')
            outputs = char_embed_module(outputs)
            
            ## tdnn ##
            tdnn_module = TDNN(self.FLAGS.kernel_widths, 
                               self.FLAGS.kernel_features, 
                               initializer=0, 
                               name='TDNN')
            outputs = tdnn_module(outputs)
            
            ## reshape ##
            num_unroll_steps = tf.shape(inputs)[1]
            reshape_module = Reshape(batch_size=self.FLAGS.batch_size,
                                     num_unroll_steps=num_unroll_steps)
            outputs = reshape_module(outputs)
            
        ##################################################
        RNN = DeepRNN
        if self.FLAGS.bidirectional:
            RNN = DeepBiRNN
            
        self._rnn_module = RNN(FLAGS=self.FLAGS)
        outputs = self._rnn_module(outputs)
        
        ##################################################
        
        self._agg_module = Aggregation(self.FLAGS, 
                                       seq_len=self._rnn_module.seq_len, 
                                       final_rnn_state=self._rnn_module.final_rnn_state)
        outputs = self._agg_module(outputs)
        
        ##################################################

#         ## linear
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
        #outputs = self.build_1(inputs, self.layers)
        #outputs = self.build_2(inputs, self.layers)
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
    def attn_outputs(self):
        return self.agg_module.outputs
        
    @property
    def attn_N(self):
        return self.agg_module.N
    
#---------------------------------------------------------------------------------
''' EXPERIMENTAL.....'''
#---------------------------------------------------------------------------------
class RHN_Cell(snt.RNNCore):
    def __init__(self, output_size,
                 num_layers=5, 
                 use_inputs_on_each_layer=False,
                 use_kronecker_reparameterization=False,
                 initializers=None,
                 name="rhn_cell"):
        super(RHN_Cell, self).__init__(name=name)
        self._output_size = output_size
        self._num_layers=num_layers
        self._use_inputs_on_each_layer=use_inputs_on_each_layer
        self._use_kronecker_reparameterization=use_kronecker_reparameterization
    
    @classmethod
    def get_possible_initializer_keys():
        return None
    
    @property
    def output_size(self):
        """Returns a description of the output size, without batch dimension."""
        return tf.TensorShape([self._output_size])
        
    def _build(self, inputs, state):
        cell = rnn_cell_modern.HighwayRNNCell(self._output_size, 
                                              num_highway_layers=self._num_layers,
                                              use_inputs_on_each_layer=self._use_inputs_on_each_layer,
                                              use_kronecker_reparameterization=self._use_kronecker_reparameterization)
#         cell = RHNCell(num_units=self._output_size,
#                        in_size=self._in_size,
#                        is_training=True,
#                        depth=self._num_layers)
        return cell(inputs, state)

class RWA_Cell(snt.RNNCore):
    def __init__(self, num_units,
                 name="rwa_cell"):
        super(RWA_Cell, self).__init__(name=name)
        self._num_units = num_units
    
    @classmethod
    def get_possible_initializer_keys():
        return None
    
    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return (self._num_units, self._num_units, self._num_units, self._num_units)
        
    def _build(self, inputs, state):
        cell = RWACell(self._num_units)
        return cell(inputs, state)

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