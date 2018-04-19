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

import argparse
import os
import sys

import tensorflow as tf
import numpy as np

import cavelab as cl

FLAGS = None

#Load hyperparams
hparams = cl.hparams(name="preprocessing")

cloud = cl.Cloud(hparams.cloud_src, mip=hparams.cloud_mip, cache=False, bounded = False, fill_missing=True)
shape = cloud.shape
shape[2] = 800
downsample = hparams.scale

def get_sample(s_size):
    x = np.floor(0.75*shape[0]*np.random.random(1)+shape[0]*0.1).astype(int)
    y = np.floor(0.75*shape[1]*np.random.random(1)+shape[1]*0.1).astype(int)
    z = np.floor(0.75*shape[2]*np.random.random(1)+shape[2]*0.1).astype(int)+1200

    scale_ratio = (1/float(downsample), 1/float(downsample))
    image = cloud.vol[x:x+downsample*s_size, y:y+downsample*s_size, z:z+1]
    template = cloud.vol[x:x+downsample*s_size, y:y+downsample*s_size, z+1:z+2]
    image, template = image[:,:,:,0], template[:,:,:,0]

    #image = cl.image_processing.resize(image[:,:,0], ratio=scale_ratio, order=1)
    #template = cl.image_processing.resize(template[:,:,0], ratio=scale_ratio, order=1)
    return image[:,:,0], template[:,:,0]

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def doncc(image, template):
    ncc = cl.image_processing.cv_normxcorr(image, template)
    pos = np.array(np.unravel_index(ncc.argmax(), ncc.shape))
    tmp = np.array(ncc)
    tmp[pos[0]-5:pos[0]+10, pos[1]-5:pos[1]+10] = 0
    pos2 = np.array(np.unravel_index(tmp.argmax(), ncc.shape))
    r1 = ncc[pos[0], pos[1]]
    r2 = ncc[pos2[0], pos2[1]]
    r = r1-r2
    return r, pos, ncc

def convert_to(hparams, num_examples):
  """Converts a dataset to tfrecords."""

  s_rows = hparams.features['search_raw']['in_width']
  t_rows = hparams.features['template_raw']['width']

  filename = hparams.tfrecord_train_dest #os.path.join(hparams.data_dir, name + '.tfrecords')

  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  index = 0

  while(index < num_examples):
    #if index%20 == 0:
    #    print(str(100*index/float(num_examples))+"%")
    #Get images
    s, t = get_sample(s_rows)

    start = int(t.shape[0]/2-t_rows/2)
    end = start + t_rows

    temp = t[start:end, start:end]
    result, _, ncc = doncc(s, temp)
    print(result)
    if(result < 0.6) and (result>0.01) :
        cl.visual.save(s/255, 'dump/image')
        cl.visual.save(t/255, 'dump/templtate')
        cl.visual.save(temp/255, 'dump/small_template')
        cl.visual.save(ncc, 'dump/ncc')
        print('done', index)
        search_raw = np.asarray(s, dtype=np.uint8).tostring()
        temp_raw = np.asarray(t, dtype=np.uint8).tostring()

        ex = tf.train.Example(features=tf.train.Features(feature={
            'search_raw': _bytes_feature(search_raw),
            'template_raw': _bytes_feature(temp_raw),}))

        writer.write(ex.SerializeToString())
        index += 1

  writer.close()


def main(unused_argv):

  # Convert to Examples and write the result to TFRecords.
  convert_to(hparams, 10000)
  #convert_to(data, hparams, 1000, 'validation_1K')
  #convert_to(data, hparams, 1000, 'test_1K')


if __name__ == '__main__':
  tf.app.run(main=main, argv=[sys.argv[0]])
