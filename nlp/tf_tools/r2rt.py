''' 
https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
'''

from tensorflow.python.util import nest

#from tensorflow.python.ops import rnn_cell_impl as rnn_cell_impl
#_state_size_with_prefix = rnn_cell_impl._state_size_with_prefix
def _state_size_with_prefix(size, prefix=None):
    if nest.is_sequence(size):
        size = list(size)
    else:
        size = [size]
    if prefix:
        size.insert(0, prefix[0])
    return tuple(size)
    
# def zero_state(cell, batch_size, dtype):
#     """Return zero-filled state tensor(s).
#     Args:
#       cell: RNNCell.
#       batch_size: int, float, or unit Tensor representing the batch size.
#       dtype: the data type to use for the state.
#     Returns:
#       If `state_size` is an int or TensorShape, then the return value is a
#       `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.
#       If `state_size` is a nested list or tuple, then the return value is
#       a nested list or tuple (of the same structure) of `2-D` tensors with
#     the shapes `[batch_size x s]` for each s in `state_size`.
#     """
#     state_size = cell.state_size
#     if nest.is_sequence(state_size):
#         state_size_flat = nest.flatten(state_size)
#         zeros_flat = [
#             tf.zeros(
#                 tf.stack(_state_size_with_prefix(s, prefix=[batch_size])),
#                 dtype=dtype)
#             for s in state_size_flat]
#         for s, z in zip(state_size_flat, zeros_flat):
#             z.set_shape(_state_size_with_prefix(s, prefix=[None]))
#         zeros = nest.pack_sequence_as(structure=state_size,
#                                       flat_sequence=zeros_flat)
#     else:
#         zeros_size = _state_size_with_prefix(state_size, prefix=[batch_size])
#         zeros = tf.zeros(tf.stack(zeros_size), dtype=dtype)
#         zeros.set_shape(_state_size_with_prefix(state_size, prefix=[None]))
# 
#     return zeros
# 
# # get_initial_cell_state(cell, zero_state_initializer, batch_size, tf.float32)
# def zero_state_initializer(shape, batch_size, dtype, index):
#     z = tf.zeros(tf.stack(_state_size_with_prefix(shape, [batch_size])), dtype)
#     z.set_shape(_state_size_with_prefix(shape, prefix=[None]))
#     return z

def get_initial_cell_state(cell, initializer, batch_size, dtype):
    """Return state tensor(s), initialized with initializer.
    Args:
      cell: RNNCell.
      batch_size: int, float, or unit Tensor representing the batch size.
      initializer: function with two arguments, shape and dtype, that
          determines how the state is initialized.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` initialized
      according to the initializer.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
    the shapes `[batch_size x s]` for each s in `state_size`.
    """
    state_size = cell.state_size
    if nest.is_sequence(state_size):
        state_size_flat = nest.flatten(state_size)
        init_state_flat = [
            initializer(_state_size_with_prefix(s), batch_size, dtype, i)
                for i, s in enumerate(state_size_flat)]
        init_state = nest.pack_sequence_as(structure=state_size,
                                    flat_sequence=init_state_flat)
    else:
        init_state_size = _state_size_with_prefix(state_size)
        init_state = initializer(init_state_size, batch_size, dtype, None)

    return init_state

# get_initial_cell_state(cell, make_variable_initializer(), batch_size, tf.float32)
def make_variable_state_initializer(**kwargs):
    def variable_state_initializer(shape, batch_size, dtype, index):
        args = kwargs.copy()

        if args.get('name'):
            args['name'] = args['name'] + '_' + str(index)
        else:
            args['name'] = 'init_state_' + str(index)

        args['shape'] = shape
        args['dtype'] = dtype

        var = tf.get_variable(**args)
        var = tf.expand_dims(var, 0)
        var = tf.tile(var, tf.stack([batch_size] + [1] * len(shape)))
        var.set_shape(_state_size_with_prefix(shape, prefix=[None]))
        return var

    return variable_state_initializer

# # gaussian_state_initializer = make_gaussian_state_initializer(zero_state_initializer, stddev=0.01)
# def make_gaussian_state_initializer(initializer, deterministic_tensor=None, stddev=0.3):
#     def gaussian_state_initializer(shape, batch_size, dtype, index):
#         init_state = initializer(shape, batch_size, dtype, index)
#         if deterministic_tensor is not None:
#             return tf.cond(deterministic_tensor,
#                 lambda: init_state,
#                 lambda: init_state + tf.random_normal(tf.shape(init_state), stddev=stddev))
#         else:
#             return init_state + tf.random_normal(tf.shape(init_state), stddev=stddev)
#     return gaussian_state_initializer

# class StateInitializer(Enum):
#     ZERO_STATE = 1
#     VARIABLE_STATE = 2
#     NOISY_ZERO_STATE = 3
#     NOISY_VARIABLE_STATE = 4