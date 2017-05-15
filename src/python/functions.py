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

import numpy

__author__ = u'Tegona SA'

class Identity(object):

    def fun(self, z):
        return z

    def prime(self, x):
        return 1

class Sigmoid(object):

    name = 'sigmoid'

    def fun(self, z):
        return 1.0 / (1.0 + numpy.exp(-z))

    def prime(self, x):
        return x * (1.0 - x)

class SoftMax(object):

    name = 'softmax'

    def fun(self, z):

        s = numpy.sum(numpy.exp(z))

        return numpy.divide(numpy.exp(z), s)

    def prime(self, x):
        """
        (u/v)' = (u'v - uv') / v^2
        """

        u = numpy.exp(x)
        v = numpy.sum(numpy.exp(x))

        return numpy.divide(u * v, v**2)

class Tanh(object):

    name = 'tanh'

    def fun(self, z):
        return numpy.tanh(z)

    def prime(self, z):
        """
        (u/v)' = (u'v - uv') / v^2
        """
        return 1 - numpy.tanh(z)**2
