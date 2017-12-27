# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts EM data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import src.helpers as helpers
import src.loss as loss
import hyperparams

import argparse
import os
import sys

import tensorflow as tf
import numpy as np

import hyperparams
import src.data as d
import src.model as models

FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(data, hparams, num_examples, name):
  """Converts a dataset to tfrecords."""

  s_rows = hparams.in_source_width
  t_rows = hparams.in_template_width

  filename = os.path.join(hparams.data_dir, name + '.tfrecords')

  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)

  sess = tf.Session()
  g = models.Graph()

  search = tf.placeholder(tf.float32, shape=[hparams.in_source_width,hparams.in_source_width])
  template = tf.placeholder(tf.float32, shape=[hparams.in_template_width,hparams.in_template_width])

  search_dim = tf.expand_dims(tf.expand_dims(search, dim=0), dim=3)
  template_dim = tf.expand_dims(tf.expand_dims(template, dim=0), dim=3)

  g.source_alpha = [search_dim]
  g.template_alpha = [template_dim]
  g.similar = tf.constant(1)

  g = models.normxcorr(g, hparams)
  g = loss.loss(g, hparams)
  index = 0
  
  while(index < num_examples):
    #if index%20 == 0:
    #    print(str(100*index/float(num_examples))+"%")
    #Get images
    t, s = data.getSample([t_rows, t_rows], [s_rows, s_rows], hparams.resize, data.metadata)

    result = sess.run(g.l, feed_dict={template: t, search: s})

    print(result)
    if(result> -0.14) or result<-0.90:
        print('done', index)
        search_raw = np.asarray(s*255, dtype=np.uint8).tostring()
        temp_raw = np.asarray(t*255, dtype=np.uint8).tostring()

        ex = tf.train.Example(features=tf.train.Features(feature={
            'search_raw': _bytes_feature(search_raw),
            'template_raw': _bytes_feature(temp_raw),}))

        writer.write(ex.SerializeToString())
        index += 1

  writer.close()


def main(unused_argv):
  # Get the data.
  hparams = hyperparams.create_hparams()
  data = d.Data(hparams, prepare = True )

  # Convert to Examples and write the result to TFRecords.
  convert_to(data, hparams, 10000, 'adasd')
  #convert_to(data, hparams, 1000, 'validation_1K')
  #convert_to(data, hparams, 1000, 'test_1K')


if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])
