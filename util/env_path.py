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

from datetime import datetime
import os


class ENVPATH():
    def __init__(self, workplace):
        """Inits SampleClass with blah."""
        cwd = os.getcwd()
        self.log_base = os.path.join(cwd, './log2')
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        self.log_dir = os.path.join(self.log_base, subdir)
        self.model_dir = os.path.join(self.log_base, 'model.ckpt')
        if workplace == 'lab':
            # self.data_dir = '/home/bingzhang/Documents/Dataset/CACD/CACD2000'
            self.data_dir = '/home/bingzhang/Documents/Dataset/MORPH/MORPH'
            self.data_info = './data/MORPH.mat'
            self.model = '/home/bingzhang/Workspace/PycharmProjects/20170512-110547/model-20170512-110547.ckpt-250000'
            self.val_dir = '/home/bingzhang/Documents/Dataset/MORPH/MORPH'
            self.val_list = './data/morph_val.txt'
        elif workplace == 'server':
            # self.data_dir = '/scratch/BingZhang/dataset/CACD2000_Cropped'
            self.data_dir = '/scratch/BingZhang/dataset/MORPH'
            self.data_info = './data/MORPH.mat'
            self.model = '/scratch/BingZhang/facenet4drfr/model/20170512-110547/model-20170512-110547.ckpt-250000'
            self.val_dir = '/scratch/BingZhang/dataset/MORPH'
            self.val_list = './data/morph_val.txt'
        elif workplace == 'sweet_home':
            self.data_dir = '/Users/bingzhang/Documents/Dataset/CACD2000/'
            self.model = '/Users/bingzhang/Documents/Dataset/model/20170529-141612-52288'
            self.val_dir = '/home/bingzhang/Documents/Dataset/lfw/'
            self.val_list = '/home/bingzhang/Documents/Dataset/ZID/LFW/lfw_trip_val.txt'
        else:
            self.data_dir = '/scratch/BingZhang/dataset/CACD2000/'
            self.model = '/scratch/BingZhang/FaceRecognition.close/models/20170529-141612-52288'
            self.val_dir = '/scratch/BingZhang/lfw/'
            self.val_list = '/scratch/BingZhang/FaceRecognition/val.list'
