import numpy as np
import tensorflow as tf

a = [[0,0,0,0,1,2,3,4],[0,0,1,2,3,4,5,6]]
b = [4,6]

x = np.array(a, dtype=np.float32)
#xt = tf.placeholder(tf.float32, shape=[2, None])
x0 = tf.Variable(tf.constant(a, dtype=tf.float32))
z = tf.Variable(tf.constant(b, dtype=tf.int32))

x1 = tf.reverse(x0,[1])
x2 = tf.reverse_sequence(x1, z, batch_dim=0, seq_dim=1)
x3 = tf.reverse(x2,[1])

def padded_reverse(x, seq_len, batch_dim=0, seq_dim=1):
    x = tf.reverse(x,[seq_dim])
    x = tf.reverse_sequence(x, seq_len, batch_dim=batch_dim, seq_dim=seq_dim)
    x = tf.reverse(x,[seq_dim])
    return x
    

sess = tf.Session()
sess.run(tf.initialize_all_variables())

#y1,y2,y3 = sess.run([x1,x2,x3])

y = sess.run(padded_reverse(x))