# -*- coding:iso-8859-1 -*-

# MIT License
#
# Copyright (c) 2017 Dnargne
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

__author__ = u'Tegona SA'

class Teacher(object):

    _neural_net = None
    _training_dataset = None
    _test_dataset = None

    def __init__(self,
                 neural_net,
                 training_dataset,
                 test_dataset):

        self.neural_net = neural_net
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset

    def sequential_learn(self, learning_rate, epoch):

        for i in range(1, epoch + 1):
            for inputs, label in self.training_dataset:
                outputs, error = self.neural_net.sequential_learn(inputs,
                                                                  label,
                                                                  learning_rate)

    @property
    def neural_net(self):
        return self._neural_net

    @neural_net.setter
    def neural_net(self, value):
        self._neural_net = value

    @property
    def training_dataset(self):
        return self._training_dataset

    @training_dataset.setter
    def training_dataset(self, value):
        self._training_dataset = value

    @property
    def test_dataset(self):
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, value):
        self._test_dataset = value
