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

import unittest
from util import data_reader
from scipy import io as sio
import numpy as np


class DataReaderTest(unittest.TestCase):
    def testDataReader(self):
        batch_size = 5
        dataReader = data_reader.DataReader('/home/bingzhang/Documents/Dataset/CACD/data', 55606, batch_size, 0.8,
                                            True)

        # print dataReader.train_indices_set
        # print dataReader.test_indices_set
        for i in range(2):
            x, y = dataReader.next_batch(phase_train=True)
        x = np.reshape(x, [batch_size, 250, 250, 3])
        sio.savemat('testDataReader.mat', {'im': x, 'label': y})


if __name__ == '__main__':
    unittest.main()
