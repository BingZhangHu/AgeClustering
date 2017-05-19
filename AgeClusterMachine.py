# MIT License
#
# Copyright (c) 2017 BingZhang Hu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import tensorflow as tf
import util.net_builder as nb
import util.data_reader as dr
import util.loss as lss
import os
from datetime import datetime
import time
import numpy as np

class AgeClusterMachine():
    """Summary of class here.

    Longer class information....
    Longer class information....

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self, sess=tf.Session()):
        self.sess = sess
        self.root_dir = os.getcwd()
        self.subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        self.log_dir = os.path.join(os.path.expanduser('logs'), self.subdir)
        self.model_dir = os.path.join(os.path.expanduser('models'), self.subdir)
        self.learning_rate = 0.0001
        self.batch_size = 40
        self.embedding_bits = 128
        self.max_epoch = 20
        self.data_dir = '/home/bingzhang/Documents/Dataset/MORPH/data_with_dense_label'
        self.image_in = tf.placeholder(tf.float32, [self.batch_size, 64, 64, 3])
        self.label_in = tf.placeholder(tf.float32, [self.batch_size, 1])
        self.net = self._build_net()
        self.loss = self._build_loss()
        self.accuracy = self._build_accuracy()
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _build_net(self):
        # convolution layers
        net, _ = nb.nn1_forward_propagation(images=self.image_in, phase_train=True, weight_decay=0.0)
        # embedding dimention is 128
        with tf.variable_scope('output') as scope:
            weights = tf.get_variable('weights', [1024, self.embedding_bits], dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=1e-2))
            biases = tf.get_variable('biases', [self.embedding_bits], dtype=tf.float32,
                                     initializer=tf.constant_initializer())
            prelogits = tf.add(tf.matmul(net, weights), biases, name=scope.name)
            nb.variable_summaries(weights, 'weights')
            nb.variable_summaries(biases, 'biases')
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        return embeddings

    def _build_loss(self):
        embeddings = tf.reshape(self.net,[self.batch_size,self.embedding_bits])
        loss=0.0
        for i in range(self.batch_size):
            for j in range(self.batch_size):
                loss_piece = lss.hinge_loss(embeddings[i][:],embeddings[j][:],abs(self.label_in[i][0]-self.label_in[j][0]))
                loss = loss+loss_piece
        tf.summary.scalar('loss', loss)

        return loss

    def _build_accuracy(self):
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.net, 1), tf.argmax(self.label_in, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def train(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        saver = tf.train.Saver()
        data_reader = dr.DataReader(self.data_dir, 55606, self.batch_size, 0.8, reproducible=True)
        tf.summary.image('image', self.image_in, 10)
        summary_op = tf.summary.merge_all()
        writer_train = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        writer_test = tf.summary.FileWriter(self.log_dir + '/test', self.sess.graph)
        step = 1
        while data_reader.epoch < self.max_epoch:
            if step % 10 == 0:
                images, label = data_reader.next_batch(phase_train=False)
                reshaped_image = np.reshape(images, [self.batch_size, 64, 64, 3])
                feed_dict = {self.image_in: reshaped_image, self.label_in: label}
                start_time = time.time()
                err, acc, sum = self.sess.run([self.loss, self.accuracy, summary_op], feed_dict=feed_dict)
                duration = time.time() - start_time
                print('Epoch:%d/%d\tTime:%.3f\tLoss:%2.4f\tAcc:%2.4f\t@[TEST]' % (
                    data_reader.current_test_batch_index, data_reader.epoch, duration, err, acc))
                writer_test.add_summary(sum, step)
            else:
                images, label = data_reader.next_batch(phase_train=True)
                reshaped_image = np.reshape(images, [self.batch_size, 64, 64, 3])
                feed_dict = {self.image_in: reshaped_image, self.label_in: label}
                start_time = time.time()
                err, acc, sum, _ = self.sess.run([self.loss, self.accuracy, summary_op, self.opt], feed_dict=feed_dict)
                duration = time.time() - start_time
                print('Epoch:%d/%d\tTime:%.3f\tLoss:%2.4f\tAcc:%2.4f\t' % (
                    data_reader.current_train_batch_index, data_reader.epoch, duration, err, acc))
                writer_train.add_summary(sum, step)
            if step % 2224 == 0:
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                saver.save(self.sess, self.model_dir, step)

            step += 1;


if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    config.gpu_options.allow_growth = True
    this_session = tf.Session(config=config)
    sess = tf.Session()
    model = AgeClusterMachine(sess=this_session)
    model.train()