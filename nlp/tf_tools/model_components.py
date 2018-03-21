import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
from nlp.util import utils as U
#from pandas.util.doctools import idx

try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple

''' https://github.com/davidsvaughn/hierarchical-attention-networks/blob/master/model_components.py
'''
    
def bidirectional_rnn(cell_fw, cell_bw, 
                      inputs_embedded, 
                      input_lengths=None,
                      pad='post',
                      scope=None):
    """Bidirecional RNN with concatenated outputs and states"""
    with tf.variable_scope(scope or "birnn") as scope:
        ((fw_outputs,
          bw_outputs),
         (fw_state,
          bw_state)) = (
            #tf.nn.bidirectional_dynamic_rnn(
            bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=inputs_embedded,
                sequence_length=input_lengths,
                pad=pad,
                dtype=tf.float32,
                swap_memory=True,
                scope=scope))
          
        outputs = tf.concat((fw_outputs, bw_outputs), 2)

        def concatenate_state(fw_state, bw_state):
            if isinstance(fw_state, LSTMStateTuple):
                state_c = tf.concat(
                    (fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
                state_h = tf.concat(
                    (fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
                state = LSTMStateTuple(c=state_c, h=state_h)
                return state
            elif isinstance(fw_state, tf.Tensor):
                state = tf.concat((fw_state, bw_state), 1,
                                  name='bidirectional_concat')
                return state
            elif (isinstance(fw_state, tuple) and
                    isinstance(bw_state, tuple) and
                    len(fw_state) == len(bw_state)):
                # multilayer
                state = tuple(concatenate_state(fw, bw)
                              for fw, bw in zip(fw_state, bw_state))
                return state

            else:
                raise ValueError(
                    'unknown state type: {}'.format((fw_state, bw_state)))


        state = concatenate_state(fw_state, bw_state)
        return outputs, state

###############################################################################

from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs

# pylint: disable=protected-access
_like_rnncell = rnn_cell_impl._like_rnncell
# pylint: enable=protected-access

def padded_reverse(x, seq_len, batch_dim=0, seq_dim=1, pad='post'):
    if pad=='pre': x = tf.reverse(x, [seq_dim])
    x = tf.reverse_sequence(x, seq_len, batch_dim=batch_dim, seq_dim=seq_dim)
    if pad=='pre': x = tf.reverse(x, [seq_dim])
    return x

def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              pad='post',
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
    """Creates a dynamic version of bidirectional recurrent neural network.
    
    Takes input and builds independent forward and backward RNNs. The input_size
    of forward and backward cell must match. The initial state for both directions
    is zero by default (but can be set optionally) and no intermediate states are
    ever returned -- the network is fully unrolled for the given (passed in)
    length(s) of the sequence(s) or completely unrolled if length(s) is not
    given.
    
    Args:
      cell_fw: An instance of RNNCell, to be used for forward direction.
      cell_bw: An instance of RNNCell, to be used for backward direction.
      inputs: The RNN inputs.
        If time_major == False (default), this must be a tensor of shape:
          `[batch_size, max_time, ...]`, or a nested tuple of such elements.
        If time_major == True, this must be a tensor of shape:
          `[max_time, batch_size, ...]`, or a nested tuple of such elements.
      sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
        containing the actual lengths for each of the sequences in the batch.
        If not provided, all batch entries are assumed to be full sequences; and
        time reversal is applied from time `0` to `max_time` for each sequence.
      initial_state_fw: (optional) An initial state for the forward RNN.
        This must be a tensor of appropriate type and shape
        `[batch_size, cell_fw.state_size]`.
        If `cell_fw.state_size` is a tuple, this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
      initial_state_bw: (optional) Same as for `initial_state_fw`, but using
        the corresponding properties of `cell_bw`.
      dtype: (optional) The data type for the initial states and expected output.
        Required if initial_states are not provided or RNN states have a
        heterogeneous dtype.
      parallel_iterations: (Default: 32).  The number of iterations to run in
        parallel.  Those operations which do not have any temporal dependency
        and can be run in parallel, will be.  This parameter trades off
        time for space.  Values >> 1 use more memory but take less time,
        while smaller values use less memory but computations take longer.
      swap_memory: Transparently swap the tensors produced in forward inference
        but needed for back prop from GPU to CPU.  This allows training RNNs
        which would typically not fit on a single GPU, with very minimal (or no)
        performance penalty.
      time_major: The shape format of the `inputs` and `outputs` Tensors.
        If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
        If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
        Using `time_major = True` is a bit more efficient because it avoids
        transposes at the beginning and end of the RNN calculation.  However,
        most TensorFlow data is batch-major, so by default this function
        accepts input and emits output in batch-major form.
      scope: VariableScope for the created subgraph; defaults to
        "bidirectional_rnn"
    
    Returns:
      A tuple (outputs, output_states) where:
        outputs: A tuple (output_fw, output_bw) containing the forward and
          the backward rnn output `Tensor`.
          If time_major == False (default),
            output_fw will be a `Tensor` shaped:
            `[batch_size, max_time, cell_fw.output_size]`
            and output_bw will be a `Tensor` shaped:
            `[batch_size, max_time, cell_bw.output_size]`.
          If time_major == True,
            output_fw will be a `Tensor` shaped:
            `[max_time, batch_size, cell_fw.output_size]`
            and output_bw will be a `Tensor` shaped:
            `[max_time, batch_size, cell_bw.output_size]`.
          It returns a tuple instead of a single concatenated `Tensor`, unlike
          in the `bidirectional_rnn`. If the concatenated one is preferred,
          the forward and backward outputs can be concatenated as
          `tf.concat(outputs, 2)`.
        output_states: A tuple (output_state_fw, output_state_bw) containing
          the forward and the backward final states of bidirectional rnn.
    
    Raises:
      TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
    """

    if not _like_rnncell(cell_fw):
        raise TypeError("cell_fw must be an instance of RNNCell")
    if not _like_rnncell(cell_bw):
        raise TypeError("cell_bw must be an instance of RNNCell")
    
    with vs.variable_scope(scope or "bidirectional_rnn"):
        
        rnn_seq_length = sequence_length if pad=='post' else None
        
        # Forward direction
        with vs.variable_scope("fw") as fw_scope:
            output_fw, output_state_fw = dynamic_rnn(
                cell=cell_fw, inputs=inputs, sequence_length=rnn_seq_length,
                initial_state=initial_state_fw, dtype=dtype,
                parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                time_major=time_major, scope=fw_scope)

        # Backward direction
        if not time_major:
            time_dim = 1
            batch_dim = 0
        else:
            time_dim = 0
            batch_dim = 1

        def _reverse(input_, seq_lengths, seq_dim, batch_dim):
            if seq_lengths is not None:
                #return array_ops.reverse_sequence(input=input_, seq_lengths=seq_lengths, seq_dim=seq_dim, batch_dim=batch_dim)
                return padded_reverse(input_, seq_lengths, batch_dim=batch_dim, seq_dim=seq_dim, pad=pad)
            else:
                return array_ops.reverse(input_, axis=[seq_dim])

        with vs.variable_scope("bw") as bw_scope:
            inputs_reverse = _reverse(
                inputs, seq_lengths=sequence_length,
                seq_dim=time_dim, batch_dim=batch_dim)
            tmp, output_state_bw = dynamic_rnn(
                cell=cell_bw, inputs=inputs_reverse, sequence_length=rnn_seq_length,
                initial_state=initial_state_bw, dtype=dtype,
                parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                time_major=time_major, scope=bw_scope)

    output_bw = _reverse(
        tmp, seq_lengths=sequence_length,
        seq_dim=time_dim, batch_dim=batch_dim)

    outputs = (output_fw, output_bw)
    output_states = (output_state_fw, output_state_bw)
    
    return (outputs, output_states)


def softmask(x, axis=-1, mask=None, T=None):
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    x = x - x_max
    
    if T!=None:
#         if not tf.is_numeric_tensor(T):
#             T = tf.get_variable('T', shape=[1], initializer=tf.constant_initializer(T), dtype=tf.float32, trainable=True)
        x = x/T
    
    ex = tf.exp(x)
    
    if mask!=None:
        ex = tf.multiply(ex, mask)
        
    es = tf.reduce_sum(ex, axis=axis, keep_dims=True)
    
#     if mask!=None:
#         ez = tf.cast(tf.reduce_sum(mask, axis=-1, keep_dims=True)==0, tf.float32)
#         es = es + ez

#     ez = tf.cast( tf.equal( es, tf.constant( 0, dtype=tf.float32 ) ), tf.float32)
#     es = es + ez
    
    ret = ex/es
    
    return ret#, ex, es, ez

def softmask2(x, mask=None, axis=-1):
    z = {}
    
    #output_shape = tf.cast( [ word_bs, tf.shape(word_level_output)[-1]] , tf.int64 )
    output_shape = tf.cast( tf.shape(x), tf.int64 )
    
    idx = tf.where(mask>0)
    x_sps = tf.gather_nd(x, idx)
    
    y_sps = tf.nn.softmax(x_sps, dim=1)
    
    y = tf.scatter_nd(indices=idx,
                      updates=y_sps,
                      shape=output_shape)
    
    z['idx'] = idx
    z['x_sps'] = x_sps
    
    return y, z

def task_specific_attention_test1(inputs, output_size,
                            initializer=layers.xavier_initializer(),
                            activation_fn=tf.tanh, scope=None):
    """
    Performs task-specific attention reduction, using learned
    attention context vector (constant within task of interest).

    Args:
        inputs: Tensor of shape [batch_size, units, input_size]
            `input_size` must be static (known)
            `units` axis will be attended over (reduced from output)
            `batch_size` will be preserved
        output_size: Size of output's inner (feature) dimension

    Returns:
        outputs: Tensor of shape [batch_size, output_dim].
    """
    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

    z = {}
    
    with tf.variable_scope(scope or 'attention') as scope:
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[output_size],
                                                   initializer=initializer,
                                                   dtype=tf.float32)
        input_projection = layers.fully_connected(inputs, output_size,
                                                  activation_fn=activation_fn,
                                                  scope=scope)

        ''' softmax '''
        ## original (softmax) ##
#         vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
#         attention_weights = tf.nn.softmax(vector_attn, dim=1)

        ## softmask ##
        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2)
        mask = tf.cast(tf.abs(tf.reduce_sum(input_projection, axis=2))>0, tf.float32)
        
        attention_weights = softmask(vector_attn, mask=mask)
        attention_weights = tf.expand_dims(attention_weights, -1)
        #if U.is_sequence(attention_weights):
        #attention_weights, ex, es, ez = attention_weights
        #z['ex']=ex; z['es']=es; z['ez']=ez 
        
        ''' weighted mean '''
        #outputs = tf.reduce_sum(tf.multiply(inputs, attention_weights), axis=1)
        
        
        ##### TESTING ###########################
        #idx = tf.cast( tf.not_equal( tf.reduce_sum(mask, axis=-1), tf.constant( 0, dtype=tf.float32 ) ), tf.float32)
        bool_idx =  tf.not_equal( tf.reduce_sum(mask, axis=-1), tf.constant( 0, dtype=tf.float32 ))
        sps_idx = tf.cast( tf.where(bool_idx), tf.int32 )
        
        #input_sps = tf.boolean_mask(inputs, bool_idx)
        input_sps = tf.gather_nd(inputs, sps_idx)
        
        attn_sps = tf.boolean_mask(attention_weights, bool_idx)
        output_sps = tf.reduce_sum(tf.multiply(input_sps, attn_sps), axis=1)
        
        
        z['bool_idx'] = bool_idx
        z['sps_idx'] = sps_idx
        z['input_sps'] = input_sps
        z['attn_sps'] = attn_sps
        z['output_sps'] = output_sps
        
        #output_shape = [tf.shape(inputs)[0], tf.shape(inputs)[-1]]
        #output_shape = [inputs.get_shape()[0], inputs.get_shape()[-1]]
        
        #input_shape = inputs.get_shape()
        input_shape = tf.shape(inputs)
        output_shape = tf.boolean_mask(input_shape, np.array([True, False, True]))
        
        z['input_shape'] = input_shape
        z['output_shape'] = output_shape
        
        outputs = tf.scatter_nd(indices=sps_idx,
                                updates=output_sps,
                                shape=output_shape)

        ### TEST #########################
        #output_shape = [tf.shape(inputs)[0], tf.shape(inputs)[-1]]
        #output_shape = [inputs.get_shape()[0], inputs.get_shape()[-1]]
        
        #init = tf.placeholder(tf.float32, shape=(2,))
        #out_var = tf.Variable(init, validate_shape=False)
        
        #zeros = tf.fill(dims=output_shape, value=0.0)
        #out_var = tf.Variable(zeros, validate_shape=False)
        
        #zeros = tf.fill(dims=output_shape, value=0.0)
        #out_var.assign(zeros)
        #op = tf.assign(out_var, zeros, validate_shape=False)
        
        #out_var = tf.get_variable('out_var', shape=output_shape, dtype=tf.float32, trainable=False)
        
        ## DEBUG ###############################################
        #z['out_var'] = out_var
        #z['zeros'] = zeros
        #z['op'] = op
        
        z['mask'] = mask
        z['inputs'] = inputs
        z['attention_context_vector'] = attention_context_vector
        z['input_projection'] = input_projection
        z['vector_attn'] = vector_attn
        z['attention_weights'] = attention_weights
        z['outputs'] = outputs
        
        ##
        return outputs, z

def task_specific_attention(inputs, output_size,
                            initializer=layers.xavier_initializer(),
                            activation_fn=tf.tanh, scope=None):

    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

    z = {}
    with tf.variable_scope(scope or 'attention') as scope:
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[output_size],
                                                   initializer=initializer,
                                                   dtype=tf.float32)
        input_projection = layers.fully_connected(inputs, output_size,
                                                  activation_fn=activation_fn,
                                                  scope=scope)
        
        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2)
        mask = tf.cast(tf.abs(tf.reduce_sum(input_projection, axis=2))>0, tf.float32)
        
        #################################################################
        attention_weights = softmask(vector_attn, mask=mask)
        attention_weights = tf.expand_dims(attention_weights, -1)

#         attention_weights = softmask2(vector_attn, mask=mask)
#         if U.is_sequence(attention_weights):
#             attention_weights, z = attention_weights
        #################################################################
        
        outputs = tf.reduce_sum(tf.multiply(inputs, attention_weights), axis=1)
        
        return outputs#, z


def task_specific_attention_ORIGINAL(inputs, output_size,
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

        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
        attention_weights = tf.nn.softmax(vector_attn, dim=1)
        
        weighted_projection = tf.multiply(input_projection, attention_weights)
        outputs = tf.reduce_sum(weighted_projection, axis=1)

        return outputs
