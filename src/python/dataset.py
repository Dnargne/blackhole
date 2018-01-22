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

import os
import numpy
import random
import fnmatch
import idx2numpy

try:
    from PIL import Image
except:
    pass

__author__ = u'Tegona SA'

class Dataset(list):

    def __iter__(self):

        indexes = [i for i in range(len(self))]

        random.shuffle(indexes)

        for i in indexes:

            value = self.__getitem__(i)

            yield value

class FileTreeDataset(Dataset):

    _path = None
    _extension = None
    _label_fun = None
    _files = None
    _data = None
    _count = None

    def __init__(self, path, extension, label_fun=lambda f: f):
        self.path = path
        self.extension = extension
        self.label_fun = label_fun
        self.data = {}
        self.count = 0

        super(FileTreeDataset, self).__init__()

    def get_files(self):
        matches = {}

        self.count = 0
        for root, dirnames, filenames in os.walk(self.path):
            for filename in fnmatch.filter(filenames, self.extension):
                filename = os.path.join(root, filename)
                label = self.label_fun(filename)

                if label not in matches.keys():
                    matches[label] = []

                self.count += 1
                matches[label].append(filename)

        return matches

    def prepare_data(self, data):
        return (None, None)

    def __iter__(self):

        for filename in self.files:

            if filename not in self.data.keys():

                data, label = self.prepare_data(filename)

                value = (data, label)

            self.data[filename] = value

            yield self.data[filename]

    def __getitem__(self, index):

        if type(index) == int:
            return self.prepare_data(self.files[index])

        dataset = type(self)(self.path,
                             self.extension,
                             self.label_fun)

        dataset.files = dataset.files[index]

        return dataset

    def __len__(self):
        return self.count

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    @property
    def extension(self):
        return self._extension

    @extension.setter
    def extension(self, value):
        self._extension = value

    @property
    def label_fun(self):
        return self._label_fun

    @label_fun.setter
    def label_fun(self, value):
        self._label_fun = value

    @property
    def files(self):
        file_dict = None
        if self._files is None:
            self._files = []
            file_dict = self.get_files()
            for k, v in file_dict.items():
                print('%s: %s' % (k, len(v)))
            for a in zip(*tuple(file_dict.values())):
                for e in a:
                    self._files.append(e)

        return self._files

    @files.setter
    def files(self, value):
        self._files = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, value):
        self._count = value

class PngFileTreeDataset(FileTreeDataset):

    _weight = None
    _height = None

    def prepare_data(self, filename):

        image = Image.open(filename)

        w = image.size[0] // 4
        h = image.size[1] // 4
        image.thumbnail((w, h))
        image = image.convert('L')

        return (numpy.array([abs(b - 255) for b in image.tobytes()]).reshape((w * h, 1)),
                self.label_fun(filename))

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

class IdxFileDataset(Dataset):

    _dataset = None
    _labels = None
    _data_path = None
    _label_path = None

    def __init__(self, data_path, label_path):
        self.data_path = data_path
        self.label_path = label_path

        self.dataset = idx2numpy.convert_from_file(self.data_path)
        self.labels = idx2numpy.convert_from_file(self.label_path)

        super(IdxFileDataset, self).__init__()

    def prepare_data(self, data):
        h, w = data.shape
        data = numpy.reshape(data, (h * w, 1))

        return data

    def __getitem__(self, index):

        if isinstance(index, int):
            return (self.prepare_data(self.dataset[index]),
                    self.labels[index])

        dataset = IdxFileDataset(self.data_path,
                                 self.label_path)

        dataset.dataset = self.dataset.__getitem__(index)
        dataset.labels = self.labels.__getitem__(index)

        return dataset

    def __getslice__(self, i, j):

        dataset = IdxFileDataset(self.data_path,
                                 self.label_path)

        dataset.dataset = self.dataset.__getslice__(i, j)
        dataset.labels = self.labels.__getslice__(i, j)

        return dataset

    def __len__(self):
        return len(self.dataset)

    @property
    def data_path(self):
        return self._data_path

    @data_path.setter
    def data_path(self, value):
        self._data_path = value

    @property
    def label_path(self):
        return self._label_path

    @label_path.setter
    def label_path(self, value):
        self._label_path = value

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

class CsvFileDataset(Dataset):

    _path = None
    _content = None
    _data = None
    _sep = None

    def __init__(self, path, sep):
        self.path = path
        self.data = {}
        self.sep = sep

    def prepare_data(self, row):

        row = row.replace('\n', '')

        return row.split(self.sep)

    def __getitem__(self, index):

        if isinstance(index, int):

            if index not in self.data.keys():
                self.data[index] = self.prepare_data(self.content[index])

            return self.data[index]

        dataset = CsvFileDataset(self.path, self.sep)

        dataset.content = self.content[index]

        return dataset

    def __getslice__(self, i, j):

        dataset = CsvFileDataset(self.path, self.sep)

        dataset.content = self.content.__getslice__(i, j)

        return dataset

    def __len__(self):
        return len(self.content)

    @property
    def content(self):
        if self._content is None:
            fd = open(self.path, 'r')
            self._content = fd.readlines()
            fd.close()

        return self._content

    @content.setter
    def content(self, value):
        self._content = value

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def sep(self):
        return self._sep

    @sep.setter
    def sep(self, value):
        self._sep = value

class CsvFileArrayDataset(CsvFileDataset):

    def prepare_data(self, row):
        row_array = super(CsvFileArrayDataset, self).prepare_data(row)

        array = eval(row_array[1])
        label = row_array[0]

        return (numpy.array(array).reshape((len(array), 1)), label)
