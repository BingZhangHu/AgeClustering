import unittest
import tensorflow as tf
import numpy as np
class MyTestCase(unittest.TestCase):
    def test_something(self):
        # a = tf.Variable(tf.zeros([1, 2]),name='a')
        b = tf.Variable([[1.0,2.0]])
        # assign_op = a.assign(b)
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     print sess.run([a,b])
        #     for _ in range(20):
        #         print sess.run(assign_op)
        index = range(1,-1,-1)
        c = b[index][0]
        b = tf.add(b,1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print sess.run(c)
            print sess.run(b)
            print sess.run(c)


if __name__ == '__main__':
    unittest.main()
