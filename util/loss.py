import tensorflow as tf

def hinge_loss(input_1,input_2,delta):
    with tf.variable_scope('hinge_loss'):
        margin = tf.reduce_sum(tf.squared_difference(input_1, input_2), 0)
        loss = tf.maximum(0.0, delta - margin)
        tf.summary.scalar('loss', loss)
    return loss