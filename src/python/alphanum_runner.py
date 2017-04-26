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

import time
import numpy
import ntpath
import random

import misc
import neural
import dataset

__author__ = u'Tegona SA'

def learn(neural_net,
          learning_rate,
          learn_sample,
          epoch_count,
          counts):

    error_min = 100
    sample_errors = 0
    sample_count, test_count = counts

    if epoch_count == 0:
        return

    t1 = time.time()
    for inputs, label in learn_sample[:sample_count]:
        reference = misc.set_bit_alphanum(label)

        inputs = numpy.array([ float(p) / 255 for p in inputs ]).reshape((32 * 32, 1))
        #print('----------------------- %s' % label)
        #misc.display_image(inputs, 32)
        outputs, error = neural_net.sequential_learn(inputs,
                                                     reference,
                                                     learning_rate,
                                                     ERROR_FILE)
        #print(reference)
        #print(outputs)
        res = numpy.argmax(outputs)
        label = numpy.argmax(reference)
        #print('res, label = %s, %s' % (res, label))
        error_min = numpy.minimum(error, error_min)

        if res != label:
            sample_errors += 1

    test_errors = compute(neural_net, learn_sample[-test_count:])

    duration = time.time() - t1

    print('Epoch %s' % epoch_count)
    print('---- Error min              : %s' % (error_min))
    print('---- Learning errors / total: %s / %s' % (sample_errors, sample_count))
    print('---- Test       ok   / total: %s / %s' % (test_count - test_errors, test_count))
    print('-----Time                   : %s sec' % duration)

    learn(neural_net,
          learning_rate,
          learn_sample,
          epoch_count - 1,
          counts)

def compute(neural_net, sample, image_save=False):
    errors = 0

    for inputs, label in sample:

        reference = misc.set_bit_alphanum(label)

        inputs = numpy.array([ float(p) / 255 for p in inputs ]).reshape((32 * 32, 1))

        outputs = neural_net.compute(inputs)
        res = numpy.argmax(outputs)
        label = numpy.argmax(reference)

        if res != label:
            errors += 1

    return errors

def save(neural_net):

    fd = open('./neural', 'w')
    fd.write(neural_net.json_serialize())
    fd.flush()
    fd.close()

random.seed()

THRESHOLD = 1.0
LEARNING_RATE = 0.3
EPOCH_COUNT = 30
SAMPLE_COUNT = 60000
TEST_COUNT = 10000

ERROR_FILE = open('./error', 'w')
OUTPUT_FILE = open('./output', 'w')

def extract(filename):
    f = ntpath.basename(filename)
    return chr(int(f[:2], 16))

learn_dataset = dataset.CsvFileDataset('./dataset/alphanum')

neural_net = neural.CompleteNeuralNet(layers=[32 * 32, 1024, 62],
                                      neuron_funs=[misc.Sigmoid()])#, misc.SoftMax()])

learn(neural_net,
      LEARNING_RATE,
      learn_dataset,
      EPOCH_COUNT,
      (SAMPLE_COUNT, TEST_COUNT))

ERROR_FILE.flush()
ERROR_FILE.close()

save(neural_net)
