# MIT License
#
# Copyright (c) 2017 BingZhang Hu
#
# Permission is hereby granted, free of charge, to any person obtaiinput_dimg a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NOinput_dimFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers import *

def conv(input_tensor, input_dim, output_dim, kernel_H, kernel_W, stride_H, stride_W, padType, name, phase_train=True,
         use_batch_norm=True, weight_decay=0.0):
    with tf.variable_scope(name):
        l2_regularizer = lambda t: l2_loss(t, weight=weight_decay)
        kernel = tf.get_variable("weights", [kernel_H, kernel_W, input_dim, output_dim],
                                 initializer=tf.truncated_normal_initializer(stddev=1e-1),
                                 regularizer=l2_regularizer, dtype=input_tensor.dtype)
        cnv = tf.nn.conv2d(input_tensor, kernel, [1, stride_H, stride_W, 1], padding=padType)

        if use_batch_norm:
            conv_bn = batch_norm(cnv, phase_train)
        else:
            conv_bn = cnv
        biases = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(),
                                 dtype=input_tensor.dtype)
        bias = tf.nn.bias_add(conv_bn, biases)
        conv1 = tf.nn.relu(bias)
        variable_summaries(kernel,name)
        variable_summaries(biases,name)
    return conv1


def affine(input_tensor, input_dim, output_dim, name, weight_decay=0.0):
    with tf.variable_scope(name):
        l2_regularizer = lambda t: l2_loss(t, weight=weight_decay)
        weights = tf.get_variable("weights", [input_dim, output_dim],
                                  initializer=tf.truncated_normal_initializer(stddev=1e-1),
                                  regularizer=l2_regularizer, dtype=input_tensor.dtype)
        biases = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(),
                                 dtype=input_tensor.dtype)
        affine1 = tf.nn.relu_layer(input_tensor, weights, biases)
    return affine1


def l2_loss(tensor, weight=1.0, scope=None):
    """Define a L2Loss, useful for regularize, i.e. weight decay.
    Args:
      tensor: tensor to regularize.
      weight: an optional weight to modulate the loss.
      scope: Optional scope for op_scope.
    Returns:
      the L2 loss op.
    """
    with tf.name_scope(scope):
        weight = tf.convert_to_tensor(weight,
                                      dtype=tensor.dtype.base_dtype,
                                      name='loss_weight')
        loss = tf.multiply(weight, tf.nn.l2_loss(tensor), name='value')
    return loss


def lppool(input_tensor, pnorm, kernel_H, kernel_W, stride_H, stride_W, padding, name):
    with tf.variable_scope(name):
        if pnorm == 2:
            pwr = tf.square(input_tensor)
        else:
            pwr = tf.pow(input_tensor, pnorm)

        subsamp = tf.nn.avg_pool(pwr,
                                 ksize=[1, kernel_H, kernel_W, 1],
                                 strides=[1, stride_H, stride_W, 1],
                                 padding=padding)
        subsamp_sum = tf.multiply(subsamp, kernel_H * kernel_W)

        if pnorm == 2:
            out = tf.sqrt(subsamp_sum)
        else:
            out = tf.pow(subsamp_sum, 1 / pnorm)

    return out


def mpool(input_tensor, kernel_H, kernel_W, stride_H, stride_W, padding, name):
    with tf.variable_scope(name):
        maxpool = tf.nn.max_pool(input_tensor,
                                 ksize=[1, kernel_H, kernel_W, 1],
                                 strides=[1, stride_H, stride_W, 1],
                                 padding=padding)
    return maxpool


def apool(input_tensor, kernel_H, kernel_W, stride_H, stride_W, padding, name):
    with tf.variable_scope(name):
        avgpool = tf.nn.avg_pool(input_tensor,
                                 ksize=[1, kernel_H, kernel_W, 1],
                                 strides=[1, stride_H, stride_W, 1],
                                 padding=padding)
    return avgpool


def batch_norm(x, phase_train):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Variable, true indicates traiinput_dimg phase
        scope:       string, variable scope
        affn:      whether to affn-transform outputs
    Return:
        normed:      batch-normalized maps
    Ref: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow/33950177
    """
    name = 'batch_norm'
    with tf.variable_scope(name):
        phase_train = tf.convert_to_tensor(phase_train, dtype=tf.bool)
        n_out = int(x.get_shape()[3])
        beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=x.dtype),
                           name=name + '/beta', trainable=True, dtype=x.dtype)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=x.dtype),
                            name=name + '/gamma', trainable=True, dtype=x.dtype)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = control_flow_ops.cond(phase_train,
                                          mean_var_with_update,
                                          lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def inception(input_tensor, input_dim, stride_size, conv1_output_dim, conv2a_output_dim, conv2_output_dim,
              conv3a_output_dim, conv3_output_dim, pool_kernel_size, conv4_output_dim, pool_stride_size, poolType, name,
              phase_train=True, use_batch_norm=True, weight_decay=0.0):
    print('name = ', name)
    print('inputSize = ', input_dim)
    print('kernelSize = {3,5}')
    print('kernelStride = {%d,%d}' % (stride_size, stride_size))
    print('outputSize = {%d,%d}' % (conv2_output_dim, conv3_output_dim))
    print('reduceSize = {%d,%d,%d,%d}' % (conv2a_output_dim, conv3a_output_dim, conv4_output_dim, conv1_output_dim))
    print('pooling = {%s, %d, %d, %d, %d}' % (
    poolType, pool_kernel_size, pool_kernel_size, pool_stride_size, pool_stride_size))
    if (conv4_output_dim > 0):
        o4 = conv4_output_dim
    else:
        o4 = input_dim
    print('outputSize = ', conv1_output_dim + conv2_output_dim + conv3_output_dim + o4)
    print('\n\n')

    net = []

    with tf.variable_scope(name):
        with tf.variable_scope('branch1_1x1'):
            if conv1_output_dim > 0:
                conv1 = conv(input_tensor, input_dim, conv1_output_dim, 1, 1, 1, 1, 'SAME', 'conv1x1',
                             phase_train=phase_train,
                             use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                net.append(conv1)

        with tf.variable_scope('branch2_3x3'):
            if conv2a_output_dim > 0:
                conv2a = conv(input_tensor, input_dim, conv2a_output_dim, 1, 1, 1, 1, 'SAME', 'conv1x1',
                              phase_train=phase_train,
                              use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                conv2 = conv(conv2a, conv2a_output_dim, conv2_output_dim, 3, 3, stride_size, stride_size, 'SAME',
                             'conv3x3', phase_train=phase_train,
                             use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                net.append(conv2)

        with tf.variable_scope('branch3_5x5'):
            if conv3a_output_dim > 0:
                conv3a = conv(input_tensor, input_dim, conv3a_output_dim, 1, 1, 1, 1, 'SAME', 'conv1x1',
                              phase_train=phase_train,
                              use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                conv3 = conv(conv3a, conv3a_output_dim, conv3_output_dim, 5, 5, stride_size, stride_size, 'SAME',
                             'conv5x5', phase_train=phase_train,
                             use_batch_norm=use_batch_norm, weight_decay=weight_decay)
                net.append(conv3)

        with tf.variable_scope('branch4_pool'):
            if poolType == 'MAX':
                pool = mpool(input_tensor, pool_kernel_size, pool_kernel_size, pool_stride_size, pool_stride_size,
                             'SAME', 'pool')
            elif poolType == 'L2':
                pool = lppool(input_tensor, 2, pool_kernel_size, pool_kernel_size, pool_stride_size, pool_stride_size,
                              'SAME', 'pool')
            else:
                raise ValueError('Invalid pooling type "%s"' % poolType)

            if conv4_output_dim > 0:
                pool_conv = conv(pool, input_dim, conv4_output_dim, 1, 1, 1, 1, 'SAME', 'conv1x1',
                                 phase_train=phase_train,
                                 use_batch_norm=use_batch_norm, weight_decay=weight_decay)
            else:
                pool_conv = pool
            net.append(pool_conv)

        concatenated = array_ops.concat(net, 3, name=name)
    return concatenated


def variable_summaries(var,name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name+'/summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn1_forward_propagation(images, phase_train=True, weight_decay=0.0):
    endpoints = {}
    weight_decay = 0.0
    phase_train = True
    net = conv(images, 3, 64, 7, 7, 1, 1, 'SAME', 'conv1_7x7', phase_train=phase_train,
               use_batch_norm=True,
               weight_decay=weight_decay)
    endpoints['conv1'] = net
    net = mpool(net, 3, 3, 2, 2, 'SAME', 'pool1')
    endpoints['pool1'] = net
    net = conv(net, 64, 64, 1, 1, 1, 1, 'SAME', 'conv2_1x1', phase_train=phase_train, use_batch_norm=True,
               weight_decay=weight_decay)
    endpoints['conv2_1x1'] = net
    net = conv(net, 64, 192, 3, 3, 1, 1, 'SAME', 'conv3_3x3', phase_train=phase_train, use_batch_norm=True,
               weight_decay=weight_decay)
    endpoints['conv3_3x3'] = net
    net = mpool(net, 3, 3, 2, 2, 'SAME', 'pool3')
    endpoints['pool3'] = net
    net = inception(net, 192, 1, 64, 96, 128, 16, 32, 3, 32, 1, 'MAX', 'incept3a', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept3a'] = net
    net = inception(net, 256, 1, 64, 96, 128, 32, 64, 3, 64, 1, 'MAX', 'incept3b', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept3b'] = net
    net = inception(net, 320, 2, 0, 128, 256, 32, 64, 3, 0, 2, 'MAX', 'incept3c', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept3c'] = net
    net = inception(net, 640, 1, 256, 96, 192, 32, 64, 3, 128, 1, 'MAX', 'incept4a', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept4a'] = net
    net = inception(net, 640, 1, 224, 112, 224, 32, 64, 3, 128, 1, 'MAX', 'incept4b', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept4b'] = net
    net = inception(net, 640, 1, 192, 128, 256, 32, 64, 3, 128, 1, 'MAX', 'incept4c', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept4c'] = net
    net = inception(net, 640, 1, 160, 144, 288, 32, 64, 3, 128, 1, 'MAX', 'incept4d', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept4d'] = net
    net = inception(net, 640, 2, 0, 160, 256, 64, 128, 3, 0, 2, 'MAX', 'incept4e', phase_train=phase_train,
                    use_batch_norm=True)
    endpoints['incept4e'] = net
    net = inception(net, 1024, 1, 384, 192, 384, 48, 128, 3, 128, 1, 'MAX', 'incept5a', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept5a'] = net
    net = inception(net, 1024, 1, 384, 192, 384, 48, 128, 3, 128, 1, 'MAX', 'incept5b', phase_train=phase_train,
                    use_batch_norm=True, weight_decay=weight_decay)
    endpoints['incept5b'] = net
    net = apool(net, 4, 4, 2, 2, 'VALID', 'pool6')
    endpoints['pool6'] = net
    net = tf.reshape(net, [-1, 1024])
    endpoints['prelogits'] = net
    net = tf.nn.dropout(net, 1)
    endpoints['dropout'] = net
    return net, endpoints


# Adience age and gender recognition net
def nn2_forward_propagation(images, phase_train=True, weight_decay=0.0):
    endpoints = {}
    weight_decay = 0.0
    phase_train = True
    net = conv(images, 3, 96, 7, 7, 4, 4, 'SAME', 'conv1_7x7', phase_train=phase_train,
               use_batch_norm=True,
               weight_decay=weight_decay)
    endpoints['conv1'] = net

    net = mpool(net, 3, 3, 2, 2, 'SAME', 'pool1')
    endpoints['pool1'] = net

    net = tf.nn.local_response_normalization(net,5,alpha=0.0001,beta=0.75,name='norm1')
    endpoints['norm1'] = net

    net = conv(net, 96, 256, 5, 5, 1, 1, 'SAME', 'conv2_5x5', phase_train=phase_train, use_batch_norm=True,
               weight_decay=weight_decay)
    endpoints['conv2_1x1'] = net

    net = mpool(net,3,3,2,2,'SAME','pool2')
    endpoints['pool2'] = net


    net = conv(net, 256, 384, 3, 3, 1, 1, 'SAME', 'conv3_3x3', phase_train=phase_train, use_batch_norm=True,
               weight_decay=weight_decay)
    endpoints['conv3_3x3'] = net

    net = mpool(net, 3, 3, 2, 2, 'SAME', 'pool3')
    endpoints['pool3'] = net

    net = tf.reshape(net, [-1, 8*8*384])
    endpoints['flat'] = net

    net = fully_connected(net,512,scope='fc1')
    endpoints['fc1']=net

    net = tf.nn.dropout(net,keep_prob=1,name='drop1')
    endpoints['drop1']=net

    net = fully_connected(net, 512, scope='fc2')
    endpoints['fc2'] = net

    net = tf.nn.dropout(net, keep_prob=1, name='drop2')
    endpoints['drop2'] = net

    return net, endpoints