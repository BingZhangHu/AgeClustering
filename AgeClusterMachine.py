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
from util.inception_resnet_v1 import *
from util.file_reader import *
from util.progress import *
import os
from datetime import datetime
import time
import numpy as np
from tensorflow.python.ops import data_flow_ops


class AgeClusterMachine():
    """
    set different hinge for different age gap
    """

    def __init__(self, sess=tf.Session()):

        self.data_dir = '/scratch/BingZhang/dataset/CACD2000_Cropped'
        self.data_info = '/scratch/BingZhang/dataset/CACD2000/celenew2.mat'

        # image size
        self.image_height = 250
        self.image_width = 250
        self.image_channel = 3

        # net parameters
        self.step = 0
        self.learning_rate = 0.006
        self.batch_size = 30
        self.embedding_bits = 128
        self.max_epoch = 1000

        self.nof_sampled_age = 20
        self.nof_images_per_age = 45
        self.age_sampled_examples = self.nof_images_per_age * self.nof_images_per_age

        # age affinity matrix, add to summary to be monitored
        self.age_affinity = tf.placeholder(tf.float32, [None, self.age_sampled_examples, self.age_sampled_examples, 1],
                                           name='age_affinity')
        self.age_affinity_binarized = tf.placeholder(tf.float32,
                                                     [None, self.age_sampled_examples, self.age_sampled_examples, 1],
                                                     name='age_affinity_binarized')
        self.nof_selected_age_triplets = tf.placeholder(tf.int32, name='nof_triplet')

        ''' input pipeline '''
        # placeholders
        self.path_placeholder = tf.placeholder(tf.string, [None, 3], name='paths')
        self.index_placeholder = tf.placeholder(tf.int64, [None, 3], name='indices')
        self.batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        # input queue (FIFO queue)
        self.input_queue = data_flow_ops.FIFOQueue(capacity=10000, dtypes=[tf.string, tf.int64], shapes=[(3,), (3,)])
        self.enqueue_op = self.input_queue.enqueue_many([self.path_placeholder, self.index_placeholder])

        # de-queue an element from input_queue
        nof_process_threads = 4
        images_and_labels = []
        for _ in range(nof_process_threads):
            file_paths, labels = self.input_queue.dequeue()
            images = []
            for file_path in file_paths:
                file_content = tf.read_file(file_path)
                image = tf.image.decode_png(file_content)
                image.set_shape((self.image_width, self.image_height, self.image_channel))
                images.append(image)
            images_and_labels.append([images, labels])
        # generate batch
        self.image_batch, self.index_batch = tf.train.batch_join(images_and_labels,
                                                                 batch_size=self.batch_size_placeholder,
                                                                 enqueue_many=True,
                                                                 capacity=nof_process_threads * self.batch_size_placeholder,
                                                                 shapes=[(self.image_width, self.image_height,
                                                                          self.image_channel), ()],
                                                                 allow_smaller_final_batch=True)
        ''' end of input pipeline '''

        # ops and tensors in graph

    def net_forward(self, image_batch):
        # convolution layers
        net, _ = inference(image_batch, keep_probability=1.0, bottleneck_layer_size=128, phase_train=True,
                           weight_decay=0.0, reuse=None)
        embeddings = slim.fully_connected(net, self.embedding_bits, activation_fn=None,
                                          weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                          weights_regularizer=slim.l2_regularizer(0.0))
        embeddings = tf.nn.l2_normalize(embeddings, dim=1, epsilon=1e-12, name='embeddings')
        return embeddings

    def get_triplet_loss(self,embeddings,deltas):
        anchor = embeddings[0:self.batch_size:3][:]
        positive = embeddings[1:self.batch_size:3][:]
        negative = embeddings[2:self.batch_size:3][:]

        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),1)
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),deltas)
        loss = tf.reduce_sum(tf.maximum(basic_loss,0.0),0)
        return loss

    def get_triplet_loss_facenet(self,anchor,positive,negative,delta):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), delta)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        return loss

    # def train(self):
    #     init_op = tf.global_variables_initializer()
    #     self.sess.run(init_op)
    #     saver = tf.train.Saver()
    #     data_reader = dr.DataReader(self.data_dir, 55606, self.batch_size, 0.8, reproducible=True)
    #     tf.summary.image('image', self.image_in, 10)
    #     summary_op = tf.summary.merge_all()
    #     writer_train = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
    #     writer_test = tf.summary.FileWriter(self.log_dir + '/test', self.sess.graph)
    #     step = 1
    #     while data_reader.epoch < self.max_epoch:
    #         if step % 10 == 0:
    #             images, label = data_reader.next_batch(phase_train=False)
    #             reshaped_image = np.reshape(images, [self.batch_size, 64, 64, 3])
    #             feed_dict = {self.image_in: reshaped_image, self.label_in: label}
    #             start_time = time.time()
    #             err, acc, sum = self.sess.run([self.loss, self.accuracy, summary_op], feed_dict=feed_dict)
    #             duration = time.time() - start_time
    #             print('[%s]\tEpoch:%d/%d\tTime:%.3f\tLoss:%2.4f\tAcc:%2.4f\t@[TEST]' % (datetime.now().isoformat(),
    #                                                                                     data_reader.current_test_batch_index,
    #                                                                                     data_reader.epoch, duration,
    #                                                                                     err, acc))
    #             writer_test.add_summary(sum, step)
    #         else:
    #             images, label = data_reader.next_batch(phase_train=True)
    #             reshaped_image = np.reshape(images, [self.batch_size, 64, 64, 3])
    #             feed_dict = {self.image_in: reshaped_image, self.label_in: label}
    #             start_time = time.time()
    #             err, acc, sum, _ = self.sess.run([self.loss, self.accuracy, summary_op, self.opt], feed_dict=feed_dict)
    #             duration = time.time() - start_time
    #             print('[%s]\tEpoch:%d/%d\tTime:%.3f\tLoss:%2.4f\tAcc:%2.4f\t' % (datetime.now().isoformat(),
    #                                                                              data_reader.current_train_batch_index,
    #                                                                              data_reader.epoch, duration, err, acc))
    #             writer_train.add_summary(sum, step)
    #         if step % 2224 == 0:
    #             if not os.path.exists(self.model_dir):
    #                 os.makedirs(self.model_dir)
    #             saver.save(self.sess, self.model_dir, step)
    #
    #         step += 1


if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    config.gpu_options.allow_growth = True
    this_session = tf.Session(config=config)
    sess = tf.Session()
    model = AgeClusterMachine(sess=this_session)
    model.train()
