from flask import Flask
from scipy.io import loadmat, savemat
from scipy.ndimage import imread
from flask import request
import json
app = Flask(__name__)
output_rect_path = '/tmp/output_rect.mat'
import cavelab as cl
from evaluation import doncc, process_ncc
import numpy as np

n_threads = 40

from pathos.multiprocessing import ProcessPool, ThreadPool
pool = ThreadPool(n_threads)

@app.route('/process_NCC')
def process_NCC():
   frames_path = request.args.get('frames_path')
   frames = loadmat(frames_path)['frames']
   rect_path = request.args.get('rect_path')
   init_rect = loadmat(rect_path)['rect'][0]

   out_rects = []
   print(init_rect)
   W = int(init_rect[3])
   H = int(init_rect[2])

   image_data = imread(frames[0][0][0])
   if len(image_data.shape) == 2:
       image_data = np.expand_dims(image_data, -1)
   template = cl.image_processing.read_without_borders_3d(np.expand_dims(image_data, -1),
                                     [init_rect[1], init_rect[0], 0],
                                     [W, H, 3]).astype(np.uint8)

   print(template.shape, template.dtype)
   #cl.visual.save(template, "dump/eval/image")
   for f in frames:
       image_path = f[0][0]
       image_data = imread(image_path)

       if len(image_data.shape) == 2:
           image_data = np.expand_dims(image_data, -1)

       ncc, _, pos, _ = doncc(image_data, template)

       template = cl.image_processing.read_without_borders_3d(np.expand_dims(image_data, -1),
                                       [pos[0], pos[1], 0],
                                       [W, H, 3]).astype(np.uint8)

       image_pred = cl.image_processing.read_without_borders_3d(np.expand_dims(image_data, -1),
                                                               [int(pos[0]),int(pos[1]),0],
                                                               [W, H, 3])


       cl.visual.save(template, "dump/eval/image")
       print(pos)
       result_bbox = [pos[1], pos[0], H, W]
       out_rects.append(result_bbox)

   savemat(output_rect_path, {"out_rects": out_rects})
   return output_rect_path


@app.route('/process_NCCNet')
def process_NCCNet():
   frames_path = request.args.get('frames_path')
   frames = loadmat(frames_path)['frames']
   rect_path = request.args.get('rect_path')
   init_rect = loadmat(rect_path)['rect'][0]

   out_rects = []
   print(init_rect)
   W = int(init_rect[3])
   H = int(init_rect[2])

   image_data = imread(frames[0][0][0])
   if len(image_data.shape) == 2:
       image_data = np.expand_dims(image_data, -1)
   #image_data = process_ncc(image_data)

   template = cl.image_processing.read_without_borders_3d(np.expand_dims(image_data, -1),
                                     [init_rect[1], init_rect[0], 0],
                                     [W, H, image_data.shape[2]]).astype(np.uint8)

   print(template.shape, template.dtype)
   images = pool.map(lambda f: imread(f[0][0]), frames)

   for image_data in images:
       if len(image_data.shape) == 2:
           image_data = np.expand_dims(image_data, -1)

       #image_data = process_ncc(image_data)
       ncc, _, pos, _ = doncc(image_data, template)

#       template = cl.image_processing.read_without_borders_3d(np.expand_dims(image_data, -1),
#                                       [pos[0], pos[1], 0],
#                                       [W, H, image_data.shape[2]]).astype(np.uint8)

#      cl.visual.save(template, "dump/eval/image")
       print(pos)
       result_bbox = [pos[1], pos[0], H, W]
       out_rects.append(result_bbox)

   savemat(output_rect_path, {"out_rects": out_rects})
   return output_rect_path




def process_test():
   frames_path = "/tmp/frames.mat" #request.args.get('frames_path')
   frames = loadmat(frames_path)['frames']
   rect_path = "/tmp/rect.mat" #request.args.get('rect_path')
   init_rect = loadmat(rect_path)['rect'][0]

   out_rects = []
   print(init_rect)
   W = int(init_rect[3])
   H = int(init_rect[2])

   image_data = imread(frames[0][0][0])
   if len(image_data.shape) == 2:
       image_data = np.expand_dims(image_data, -1)
   image_data = process_ncc(image_data)

   template = cl.image_processing.read_without_borders_3d(np.expand_dims(image_data, -1),
                                     [init_rect[1], init_rect[0], 0],
                                     [W, H, image_data.shape[2]]).astype(np.uint8)

   print(template.shape, template.dtype)
   #cl.visual.save(template, "dump/eval/image")
   images = pool.map(lambda f: imread(f[0][0]), frames)

   for image_data in images:
#       image_path = f[0][0]
#       image_data = image#imread(image_path)

       if len(image_data.shape) == 2:
           image_data = np.expand_dims(image_data, -1)
       cl.visual.save(image_data[:,:,0], "dump/eval/image")
       image_data = process_ncc(image_data)
       print(image_data.shape, template.shape)

       ncc, _, pos, _ = doncc(image_data, template)

#       template = cl.image_processing.read_without_borders_3d(np.expand_dims(image_data, -1),
#                                        [pos[0], pos[1], 0],
#                                       [W, H, image_data.shape[2]]).astype(np.uint8)

#       image_pred = cl.image_processing.read_without_borders_3d(np.expand_dims(image_data, -1),
#                                                               [int(pos[0]),int(pos[1]),0],
#                                                               [W, H, 3])


       cl.visual.save(ncc[:,:], "dump/eval/template")
       print(pos)
       result_bbox = init_rect #[pos[1], pos[0], H, W]
       out_rects.append(result_bbox)

   savemat(output_rect_path, {"out_rects": out_rects})
   return output_rect_path



if __name__ == '__main__':
    # if not os.path.exists('db.sqlite'):
    #    db.create_all()
    process_test()
    #app.run(host='0.0.0.0', port=5000, debug=False)
