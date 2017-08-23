import unittest
import tensorflow as tf

class TestModule(unittest.TestCase):
    def test_loss_function(self):
        with tf.Session() as sess:
            embeddings = tf.get_variable(name='fake_embeddings', shape=[6, 2], dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=1))
            delta = tf.Variable(name='fake_delta',initial_value=[[1],
                                                                 [2]],dtype=tf.float32)
            loss1 = get_triplet_loss(embeddings,delta)
            anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, 2]), 3, 1)
            loss2 = get_triplet_loss_facenet(anchor,positive,negative,delta)

            sess.run(tf.global_variables_initializer())
            print sess.run(embeddings)
            print sess.run(loss1)
            print sess.run(anchor)
            print sess.run(positive)
            print sess.run(negative)
            print sess.run(loss2)
            print sess.run(tf.add(embeddings,[1,2]))

def get_triplet_loss(embeddings,deltas):
    anchor = embeddings[0:6:3][:]
    positive = embeddings[1:6:3][:]
    negative = embeddings[2:6:3][:]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),1)
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),tf.reshape(deltas,(2,)))
    loss = tf.reduce_mean(tf.maximum(basic_loss,0.0),0)
    return loss,basic_loss


def get_triplet_loss_facenet(anchor,positive,negative,delta):
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), tf.reshape(delta,(2,)))
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
    return loss,basic_loss

if __name__ == '__main__':
    unittest.main()
