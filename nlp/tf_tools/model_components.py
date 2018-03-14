import tensorflow as tf
import tensorflow.contrib.layers as layers
from nlp.util import utils as U

try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


def bidirectional_rnn(cell_fw, cell_bw, inputs_embedded, 
                      input_lengths=None,
                      scope=None):
    """Bidirecional RNN with concatenated outputs and states"""
    with tf.variable_scope(scope or "birnn") as scope:
        ((fw_outputs,
          bw_outputs),
         (fw_state,
          bw_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                            cell_bw=cell_bw,
                                            inputs=inputs_embedded,
                                            sequence_length=input_lengths,
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
    ez = tf.cast( tf.equal( es, tf.constant( 0, dtype=tf.float32 ) ), tf.float32)
    es = es + ez
    
    ret = ex/es
    
    return ret#, ex, es, ez

def task_specific_attention(inputs, output_size,
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

        ''' orig '''
#         vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
#         attention_weights = tf.nn.softmax(vector_attn, dim=1)

        ''' softmask '''
        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2)
        mask = tf.cast(tf.abs(tf.reduce_sum(inputs, axis=2))>0, tf.float32); z['mask'] = mask
        attention_weights = softmask(vector_attn, mask=mask)
        #if U.is_sequence(attention_weights):
        #attention_weights, ex, es, ez = attention_weights
        #z['ex']=ex; z['es']=es; z['ez']=ez 
        attention_weights = tf.expand_dims(attention_weights, -1)
        
        ''' ???? '''
        ## original
        #weighted_projection = tf.multiply(input_projection, attention_weights); z['weighted_projection'] = weighted_projection
        #outputs = tf.reduce_sum(tf.multiply(input_projection, attention_weights), axis=1)
        ## dsv
        outputs = tf.reduce_sum(tf.multiply(inputs, attention_weights), axis=1)
        
        z['inputs'] = inputs
        z['attention_context_vector'] = attention_context_vector
        z['input_projection'] = input_projection
        z['vector_attn'] = vector_attn
        z['attention_weights'] = attention_weights
        z['outputs'] = outputs

        return outputs, z
