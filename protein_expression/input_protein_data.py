"""Functions for downloading and reading Protein data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import xlrd
import tensorflow.python.platform
import numpy
from urllib import request
import tensorflow as tf
import numpy as np
SOURCE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00342/'
FILE_NAME = 'Data_Cortex_Nuclear.xls'
def maybe_download(work_directory):
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, FILE_NAME)
  if not os.path.exists(filepath):
    filepath, _ = request.urlretrieve(SOURCE_URL + FILE_NAME, filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', FILE_NAME, statinfo.st_size, 'bytes.')
  return filepath

class DataSet(object):
  def __init__(self, expression_profile, labels, feature_name, label_name, dtype=tf.float32):
    """Construct a DataSet.
    `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid Dataset input dtype %r, expected uint8 or float32' %
                      dtype)
    assert expression_profile.shape[0] == labels.shape[0], (
      'expression_profile.shape: %s labels.shape: %s' % (expression_profile.shape,
                                             labels.shape))
    self._num_examples = expression_profile.shape[0]
    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    assert expression_profile.shape[1] == 77
    expression_profile = expression_profile.reshape(expression_profile.shape[0], expression_profile.shape[1])

    self._expression_profile = expression_profile
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._feature_name = feature_name
    self._label_name = label_name
  @property
  def feature_name(self):
    return self._feature_name
  @property
  def label_name(self):
    return self._label_name
  @property
  def expression_profile(self):
    return self._expression_profile
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._expression_profile = self._expression_profile[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._expression_profile[start:end], self._labels[start:end]

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_labels(filename):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  book = xlrd.open_workbook(filename)
  mapping = {}
  sh = book.sheet_by_index(0)
  header = sh.row(0)
  counter = -1
  indexes = []
  for rx in range(1, sh.nrows):
      label = sh.row(rx)[81].value
      if(label in mapping):
        index = mapping[label]
      else :
        counter += 1
        mapping[label] = counter
        index = counter

      indexes.append(index)
  labels = dense_to_one_hot(np.array(indexes), counter+1)
  return labels, {v: k for k, v in mapping.items()}


def extract_expression_profile(filename):
  """Extract the images into a 77D float numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  book = xlrd.open_workbook(filename)
  sh = book.sheet_by_index(0)
  feature_name = [x.value for x in sh.row(0)[1: 78]]
  expression_profile = np.zeros(shape=[sh.nrows - 1, 77])
  for rx in range(0, sh.nrows-1):
      expression_profile[rx] = np.array([0.0 if(isinstance(x.value, str)) else x.value for x in sh.row(rx)[1: 78]], dtype=np.float32)

  return expression_profile, feature_name


def read_data_sets(train_dir, one_hot=False):
  local_file = maybe_download(train_dir)
  train_expression_profile, feature_name = extract_expression_profile(local_file)
  train_labels, label_name = extract_labels(local_file)
  return DataSet(train_expression_profile, train_labels, feature_name, label_name)