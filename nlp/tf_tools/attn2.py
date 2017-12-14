import tensorflow as tf
import sonnet as snt

def my_softmax(x, axis=-1, z=0., n=0.):
#     x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
#     x = x - x_max
    x_max = tf.reduce_max(x, axis=axis, keep_dims=False)
    x = x - tf.expand_dims(x_max, -1)
    ex = tf.exp(x)
    es = tf.reduce_sum(ex, axis=axis, keep_dims=True)
    
    z = tf.expand_dims(z, -1) - x_max
    ez = tf.exp(n) * tf.exp(z)
    
    #???????????????????????????
    #ex += tf.expand_dims(ez, -1)
    #???????????????????????????
    
    es += tf.expand_dims(ez, -1)
    v = ex / es
    return v, z, ez

def softmask(x, axis=-1, mask=None):
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    x = x - x_max
    
    ex = tf.exp(x)
    if mask!=None:
        ex = tf.multiply(ex, mask)
        
    es = tf.reduce_sum(ex, axis=axis, keep_dims=True)
    
    return ex/es

def ps(x, s):
    print('{} : {}'.format(s, x.shape))

def attention(inputs, attention_size, cT, seq_len=None):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
    mask = tf.cast(tf.abs(tf.reduce_sum(inputs, axis=2))>0, tf.float32)
    
    outputs = []
    
    outputs.append(inputs)      # 0 = inputs
    outputs.append(cT)          # 1 = cT
    
    ps(inputs, 'inputs')
    ps(cT, 'cT')
    
    ct = tf.expand_dims(cT,-1)
    
    ct = tf.tanh(ct)# ?????#dsv
    
    out = z = tf.squeeze(tf.matmul(inputs, ct))
    outputs.append(out)         # 2 = z
    ps(z, 'z')
    
    #out = alphas = tf.nn.softmax(z)
    out = alphas = softmask(z, mask=mask)
    outputs.append(out)         # 3 = alphas
    ps(alphas, 'alphas')
    
    #alphas = tf.expand_dims(alphas, -1)
    out = c = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    outputs.append(out)         # 4 = c
    ps(c, 'c')
    
    out = cc = tf.concat([c, cT], axis=1)
    outputs.append(out)         # 5 = cc
    ps(cc, 'cc')
    
    Wc = tf.get_variable('W', shape=[hidden_size*2, attention_size], initializer=tf.random_normal_initializer(stddev=0.1))
    
    out = h = tf.tanh(tf.matmul(cc, Wc))
    outputs.append(out)         # 6 = h
    ps(h, 'h')
    
    return outputs
    

class Attn2(snt.AbstractModule):
    def __init__(self, FLAGS,
                 final_rnn_state=None,
                 seq_len=None,
                 name="attn2"):
        super(Attn2, self).__init__(name=name)
        self.FLAGS = FLAGS
        self.final_rnn_state = final_rnn_state
        self.seq_len = seq_len
    
    def _build(self, inputs):
        self._outputs = attention(inputs, self.FLAGS.att_size, cT=self.final_rnn_state, seq_len=self.seq_len)
        
        if type(self._outputs) is list:
            return self._outputs[-1]
        return self._outputs
    
    @property
    def outputs(self):
        self._ensure_is_connected()
        if type(self._outputs) is list:
            return self._outputs
        return []