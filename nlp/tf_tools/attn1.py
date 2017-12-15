import tensorflow as tf
import sonnet as snt

def softmask(x, axis=-1, mask=None):
    if mask is None:
        mask = tf.cast(tf.abs(x)>0, tf.float32)
    
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    x = x - x_max
    
    ex = tf.multiply(tf.exp(x), mask)
    es = tf.reduce_sum(ex, axis=axis, keep_dims=True)
    alpha = ex/es
    
    #return alpha
    return [x_max, x, ex, es, alpha]

def attention_orig(inputs, attention_size, std=0.1):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.

    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article
    
    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """
    info = []
    
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    W_omega = tf.get_variable('W_omega', shape=[hidden_size, attention_size], initializer=tf.random_normal_initializer(stddev=std))
    b_omega = tf.get_variable('b_omega', shape=[attention_size], initializer=tf.random_normal_initializer(stddev=std))
    u_omega = tf.get_variable('u_omega', shape=[attention_size], initializer=tf.random_normal_initializer(stddev=std))

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    v = tf.tanh(tf.tensordot(inputs, W_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1)   # (B,T) shape
    
    mask = tf.cast(tf.abs(tf.reduce_sum(inputs, axis=2))>0, tf.float32)
    
    info.append(inputs)
    info.append(v)
    info.append(vu)
    info.append(mask)
    
    #alphas = tf.nn.softmax(vu)              # (B,T) shape also
    alphas = softmask(vu, mask=mask)
    
    if type(alphas) is list:
        info.extend(alphas)
        alphas = alphas[-1]

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    
    info.append(output)
    return info
    #return output

def ps(x, s):
    print('{} : {}'.format(s, x.shape))

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

def attention(inputs, attention_size, std=0.1, n_init=7.0):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer
    
    W_omega = tf.get_variable('W_omega', shape=[hidden_size, attention_size], initializer=tf.random_normal_initializer(stddev=std))
    b_omega = tf.get_variable('b_omega', shape=[attention_size], initializer=tf.random_normal_initializer(stddev=std))
    u_omega = tf.get_variable('u_omega', shape=[attention_size], initializer=tf.random_normal_initializer(stddev=std))

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
#     v = tf.tanh(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
    w = tf.tensordot(inputs, W_omega, axes=1)
    v = tf.tanh(w + b_omega)
    
    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1)   # (B,T) shape
    
    ########################################
    
    ''' masking? '''
    mask0 = tf.reduce_sum(inputs, axis=2) 
    mask = tf.cast(tf.abs(mask0)>0, tf.float32)
    ## HERE! ##
    #vu = tf.multiply(vu, mask)
    
    n = tf.get_variable('n', shape=[1], trainable=True, initializer=tf.constant_initializer(n_init, dtype=tf.float32))
    
    z = tf.tensordot(u_omega, tf.tanh(b_omega), axes=1)
    #z = tf.get_variable('z', shape=[1,1], trainable=False, initializer=tf.constant_initializer(1.0, dtype=tf.float32))
    
    ########################################
    
    ## HERE! ##
    #alphas = tf.nn.softmax(vu)              # (B,T) shape also
    ## -OR- ##
    #alphas, z, ez = my_softmax(vu, z=z, n=n)
    # or
    alphas = softmask(vu)#, mask=mask)
    
    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    #output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    ia = inputs * tf.expand_dims(alphas, -1)
    output = tf.reduce_sum(ia, 1)
    
    ps(inputs, 'inputs')
    ps(alphas, 'alphas')
    ps(output, 'output')

    return [w, v, vu, alphas, ia, mask0, mask, n, output]

    
class Attn1(snt.AbstractModule):
    def __init__(self, FLAGS,
                 name="attn1"):
        super(Attn1, self).__init__(name=name)
        self.FLAGS = FLAGS
    
    def _build(self, inputs):
        self._outputs = attention_orig(inputs, self.FLAGS.att_size, std=self.FLAGS.attn_std)
        #self._outputs = attention(inputs, self.FLAGS.att_size, std=self.FLAGS.attn_std, n_init=self.FLAGS.att_n)
        
        if type(self._outputs) is list:
            return self._outputs[-1]
        return self._outputs
    
    @property
    def outputs(self):
        self._ensure_is_connected()
        if type(self._outputs) is list:
            return self._outputs
        return []