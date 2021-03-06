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

import json
import numpy

import functions

__author__ = u'Tegona SA'

#
## Types

class CompleteNeuralNet(object):

    neurons_matrix = None

    biases = None
    weights = None
    neuron_funs = None
    inputs = None
    outputs = None
    zs = None

    def __init__(self,
                 json=None,
                 layers=[],
                 neuron_funs=None,
                 biases=True):

        if neuron_funs is None:
            neuron_funs = [functions.Identity]

        self.biases = []
        self.weights = []
        self.neuron_funs = neuron_funs

        weight_size = layers[0]
        for i in layers[1:]:

            if biases:
                self.biases.append(numpy.random.randn(i, 1))
            else:
                self.biases.append(numpy.zeros((i, 1)))
            self.weights.append(numpy.random.randn(i, weight_size))

            weight_size = i

    def compute(self, inputs):
        """
        Generic method to compute a solution for:

        * feedforward
        * complete neuron network
        * any hidden neuron layer
        """
        self.outputs = []
        self.inputs = []
        self.zs = []

        for w, b in zip(self.weights, self.biases):
            self.inputs.append(inputs)

            neuron_fun = self.neuron_funs[0]
            if len(self.inputs) == len(self.weights):
                neuron_fun = self.neuron_funs[-1]

            zs = numpy.dot(w, inputs) + b
            inputs = neuron_fun.fun(zs)

            self.zs.append(zs)
            self.outputs.append(inputs)

        return self.outputs[-1]

    def total_output_error(self, outputs, references):

        return (sum((references - outputs) ** 2)) / float(len(outputs))

    def output_error_signals(self, outputs, zs, references):

        return (references - outputs) * self.neuron_funs[-1].prime(zs)

    def hidden_error_signals(self, error_signal, weights, outputs):

        return numpy.dot(error_signal.transpose(), weights).transpose() * self.neuron_funs[0].prime(outputs)

    def set_weights(self, error_signals, learning_rate):

        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + \
                              (learning_rate * numpy.dot(error_signals[i],
                                                         self.inputs[i].transpose()))

    def set_biases(self, error_signals, learning_rate):

        for i in range(len(self.biases)):
            self.biases[i] = self.biases[i] + \
                             (learning_rate * error_signals[i])

    def sequential_learn(self, inputs, references, learning_rate):
        error_signals = []

        outputs = self.compute(inputs)

        error = self.total_output_error(outputs, references)

        error_signals.append(self.output_error_signals(outputs,
                                                       self.zs[-1],
                                                       references))

        hidden_layer_count = len(self.weights) - 1

        for i in reversed(range(hidden_layer_count)):
            error_signals.append(self.hidden_error_signals(error_signals[-1],
                                                           self.weights[i + 1],
                                                           self.zs[i]))

        error_signals.reverse()

        self.set_weights(error_signals, learning_rate)

        self.set_biases(error_signals, learning_rate)

        return outputs, error

    def load_layers(self, layers):

        i = 0
        for layer in layers:
            biases = []
            weights = []

            for neuron in layer['neurons']:
                weights.append(neuron['weights'])
                biases.append(neuron['biases'])

            self.weights[i] = numpy.array(weights)
            self.biases[i] = numpy.array(biases)
            i += 1

    def json_serialize(self):
        """
        :TODO: move the function field in layer def
        """
        json_str = '{"function": "%s", "type": "%s", "layers": [%s]}'

        layers = []
        for layer, biases in zip(self.weights, self.biases):
            neurons = []
            for weights, bias in zip(layer, biases):
                weights = ','.join([str(w) for w in weights])
                bias = '%s' % bias
                neurons.append('{"weights": [%s], "biases": %s}' % (weights, bias))
            layers.append(neurons)

        json_layers = ','.join(['{"neurons": [%s]}' % ','.join(layer) for layer in layers])

        json_str = json_str % (self.neuron_funs[0].name,
                               self.type,
                               json_layers)

        return json_str

    @property
    def type(self):
        return self.__class__.__name__
