import tensorflow as tf
def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

init_op = tf.global_variables_initializer()
saver = tf.train.Saver({"W_conv1": W_conv1,
                        "b_conv1": b_conv1,
                        "W_conv2": W_conv2,
                        "b_conv2": b_conv2,
                        "W_fc1": W_fc1,
                        "b_fc1": b_fc1,
                        "W_fc2": W_fc2,
                        "b_fc2": b_fc2})
with tf.Session() as sess:
        sess.run(init_op)
        saver.save(sess, "savemodel/initialize_wb.ckpt")
