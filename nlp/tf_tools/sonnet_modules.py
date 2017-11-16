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

from nlp.rwa.RWACell import RWACell
from nlp.tensorflow_with_latest_papers import rnn_cell_modern
# from nlp.recurrent_highway_networks.rhn import RHNCell

#import LM_sonnet_modules as lm
#from nlp.util.utils import interleave

'''
https://github.com/deepmind/sonnet/blob/master/sonnet/examples/rnn_shakespeare.py
'''

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

###############################################################################

def rnn_unit(args):
    kwargs = {}
    if args.unit=='lstm':
        rnn = tf.nn.rnn_cell.LSTMCell
        kwargs = { 'reuse':False, 'forget_bias':args.forget_bias, 'state_is_tuple':True }
    elif args.unit=='gru':
        rnn = tf.nn.rnn_cell.GRUCell
        kwargs = { 'reuse':False }
    elif args.unit=='rwa':
        rnn = RWA_Cell
    elif args.unit=='rhn':
        rnn = RHN_Cell
    return rnn, kwargs

def create_rnn_cell(args):
    rnn, kwargs = rnn_unit(args)
    cell = rnn(args.rnn_size, **kwargs)
    #cell = tf.contrib.rnn.ResidualWrapper(cell)
    #cell = tf.contrib.rnn.HighwayWrapper(cell)
    #cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=10, attn_size=100)
    if args.dropout>0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=args._keep_prob)#variational_recurrent=True
    return cell

class DeepRNN(snt.AbstractModule):
    def __init__(self, rnn_size,
                 num_layers=1,
                 batch_size=128, 
                 dropout=0.0,
                 forget_bias=0.0,
                 seq_len=None,
                 train_initial_state=False,
                 unit='lstm',
                 name="deep_rnn"):
        super(DeepRNN, self).__init__(name=name)
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.forget_bias = forget_bias
        self._seq_len = seq_len
        self.train_initial_state = train_initial_state
        self.unit = unit
        
        with self._enter_variable_scope():
            self._keep_prob = tf.placeholder_with_default(1.0-self.dropout, shape=())
            if seq_len is None:
                self._seq_len = tf.placeholder(tf.int32, [batch_size])

    def _build(self, inputs):
        if self.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([create_rnn_cell(self) for _ in range(self.num_layers)], state_is_tuple=True)
        else:
            cell = create_rnn_cell(self)

        if self.train_initial_state:
            self._initial_rnn_state = cell.initial_state(self.batch_size, tf.float32, trainable=True)
        else:
            self._initial_rnn_state = cell.zero_state(self.batch_size, tf.float32)
        
        output, final_rnn_state = tf.nn.dynamic_rnn(cell,
                                                    inputs,
                                                    dtype=tf.float32,
                                                    sequence_length=self._seq_len,
                                                    initial_state=self._initial_rnn_state)
        return output, final_rnn_state
    
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

###############################################################################

class DeepBiRNN(snt.AbstractModule):
    def __init__(self, rnn_size,
                 num_layers=1,
                 batch_size=128, 
                 dropout=0.0,
                 forget_bias=0.0,
                 seq_len=None,
                 train_initial_state=False,
                 unit='lstm',
                 name="deep_bi_rnn"):
        super(DeepBiRNN, self).__init__(name=name)
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout
        self.forget_bias = forget_bias
        self._seq_len = seq_len
        self.train_initial_state = train_initial_state
        self.unit = unit
        
        with self._enter_variable_scope():
            self._keep_prob = tf.placeholder_with_default(1.0, shape=())
            if seq_len is None:
                self._seq_len = tf.placeholder(tf.int32, [batch_size])

    def _build(self, inputs):
        cells_fw = [create_rnn_cell(self) for i in range(self.num_layers)]
        cells_bw = [create_rnn_cell(self) for i in range(self.num_layers)]
         
        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
                                                                                                   sequence_length=self._seq_len,
                                                                                                   dtype=tf.float32,
                                                                                                   scope='BiMultiLSTM')
        return outputs, (output_state_fw, output_state_bw)
    
    @property
    def seq_len(self):
        return self._seq_len
    
    @property
    def keep_prob(self):
        return self._keep_prob
    
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