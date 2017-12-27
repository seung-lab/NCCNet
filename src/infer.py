from cavelab import Infer, Cloud, h5data
import numpy as np
import time
import sys

# "data/tf/pinky40/train_v3.tfrecords"

CLOUD_SRC = 'gs://neuroglancer/drosophila_sergiy/image'
CLOUD_DST = 'gs://neuroglancer/drosophila_sergiy/ncc_out/'

CLOUD_SRC = 'gs://neuroglancer/piriform_v0/image_single_slices'
CLOUD_DST = 'gs://neuroglancer/piriform_v0/nccnet'

MODEL_DIR = 'logs/NCCNetv1/'
features = { "inputs":"input/image:0", "outputs": "Passes/image_transformed:0"}
input_volume = Cloud(CLOUD_SRC, mip=2, cache=False)
output_volume = Cloud(CLOUD_DST, mip=2, cache=False)
#h5 = h5data('data/slices/')
#output_volume = h5.create_dataset('prealigned_2', shape=input_volume.shape[0:2], dtype='uint8')

infer = Infer(batch_size = 8,
              width = 512,
              n_threads = 8,
              model_directory = MODEL_DIR,
              cloud_volume = True,
              features=features,
              voxel_offset = (0, 0),
              crop_size=60)

#input_volume.vol.flush_cache()
#25088, 15616

locations = [[(0,0,i), (11264,11264,i)] for i in xrange(int(sys.argv[1]), int(sys.argv[2]))]
infer.process_by_superbatch(input_volume, output_volume, locations)
