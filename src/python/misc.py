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

import png
import math
import numpy
import decimal

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
        up =  numpy.exp(x)
        vp = numpy.sum(numpy.exp(x))

        prod = (up * v) - (u * vp)

        return numpy.divide(prod, v**2)

#
## Functions

def set_bit_array(digit):
    outputs = numpy.zeros((10, 1))
    outputs[digit] = 1.0
    return outputs

def set_bit_alphanum(value):

    if value >= 'a':
        value = ord(value) - ord('a')
    elif value >= 'A':
        value = (ord(value) - ord('A')) + 26
    else:
        value = (ord(value) - ord('0')) + 52

    outputs = numpy.zeros((62, 1))
    outputs[value] = 1.0
    return outputs

def matrix2d_to_array(matrix):
    return numpy.reshape(matrix, (784, 1))

def display_row(row):

    def select_char(c):
        if c < 0.25:
            return ' '
        elif c < 0.5:
            return '#'
        return '#'

    print(''.join(['%s' % select_char(v) for v in row]))

def display_image(image, width=None):

    rows = image
    if width is not None:
        for i in range(len(image) // width):
            display_row(image[(i * width):][:width])
    else:
        for row in rows:
            display_row(row)
