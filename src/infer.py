from cavelab import Infer, Cloud, h5data
import numpy as np
import time
import sys

# "data/tf/pinky40/train_v3.tfrecords"


CLOUD_SRC = 'gs://neuroglancer/drosophila_v0/image_v14_single_slices'
CLOUD_DST = 'gs://neuroglancer/drosophila_v0/image_v14_single_slices/nccnet/'

MODEL_DIR = 'logs/NCCNet_flyem/'
features = { "inputs":"input/image:0", "outputs": "Pred/image_transformed:0"}
features = { "inputs":"image:0", "outputs": "output/image:0"}

input_volume = Cloud(CLOUD_SRC, mip=2, cache=False)
output_volume = Cloud(CLOUD_DST, mip=2, cache=False)
#h5 = h5data('data/slices/')
#output_volume = h5.create_dataset('prealigned_2', shape=input_volume.shape[0:2], dtype='uint8')

infer = Infer(batch_size = 8,
              width = 384,
              n_threads = 8,
              model_directory = MODEL_DIR,
              cloud_volume = True,
              features=features,
              voxel_offset = (0, 0),
              crop_size=60)

#input_volume.vol.flush_cache()

locations = [[(0,0,i), (65536,43744,i)] for i in xrange(int(sys.argv[1]), int(sys.argv[2]))]
infer.process_by_superbatch(input_volume, output_volume, locations)
