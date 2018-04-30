# use training data
#import tensorflow as tf
import cavelab as cl
from data import Data
import numpy as np
MODEL_DIR = 'log/imagenet_layers_two_v4/'
output_dim = 2
from prepare_data import get_image_data, base, suffix


#FIXME this to go to hparams

s_width = 512
BATCH_SIZE = 4
size = 100
width = 160

#MODEL_DIR = 'logs/NCCNet_flyem/'
features = { "inputs":"image:0", "outputs": "output/image_0:0"}
#features = {"inputs": "image:0", "outputs": "add_96:0"}
#features = { "inputs":"image:0", "outputs": "output/image:0"}



# Init
hparams = cl.hparams(name="evaluation")
#d = Data(hparams, random=False)
#print('data loaded')
#image, template, _ = d.get_batch(switching=False) #Sample test



print('model loaded')

def get_batch():
    #image, template, _ = d.get_batch(switching=False)
    #template = np.pad(template, 128, mode='constant')
    image = np.zeros((BATCH_SIZE,s_width,s_width,3), dtype=np.uint8)
    template = np.zeros((BATCH_SIZE,s_width,s_width,3), dtype=np.uint8)

    i=0
    while i<BATCH_SIZE:
        try:
            image[i], template[i] = get_image_data(base, suffix, s_width, s_width)
            i +=1
        except:
            pass

    #cl.visual.save(ncc, "dump/eval/ncc")
    #template = np.array(elastic_deform(image))
    #loc a= np.array(np.random.randint(source_width-width, size=2*BATCH_SIZE)).reshape((BATCH_SIZE, 2))
    loc = np.repeat((s_width-width)/2, BATCH_SIZE*2).reshape((BATCH_SIZE, 2))
    return image, template, loc


def doncc(image, template):
    #image = image3D[:,:,0]
    #template = template3D[:,:,0]
    nccs = []
    locs = []

    for channel in range(image.shape[-1]):
        #print(image.shape, template.shape)
        #channel = 0
        #cl.visual.save(image[:,:,channel], "dump/eval/image_"+str(channel))
        #cl.visual.save(template[:,:,channel], "dump/eval/template"+str(channel))
        nccs.append(cl.image_processing.cv_normxcorr(image[:,:,channel],
                                                     template[:,:,channel]))
        pos = np.array(np.unravel_index(nccs[-1].argmax(), nccs[-1].shape))
        locs.append(pos)

    ncc = np.sum(nccs, axis=0)/image.shape[-1]
    pos = np.array(np.unravel_index(ncc.argmax(), ncc.shape))
    return ncc, nccs, pos, locs

def get_wrong_matches(imgs, tmps, locs):
    if len(imgs.shape)==3:
        imgs = np.expand_dims(imgs, -1)
        tmps = np.expand_dims(tmps, -1)
    imgs = imgs[:,:s_width, :s_width, :]
    tmps = tmps[:,:s_width, :s_width, :]
    err = 0

    for j in range(BATCH_SIZE):

        tmp = tmps[j,int(locs[j,0]):int(locs[j,0]+width), int(locs[j, 1]):int(locs[j, 1]+width), :]
        #imgs = imgs[j,int(locs[j,0]):int(locs[j,0]+width), int(locs[j, 1]):int(locs[j, 1]+width), :]

        ncc, _, pos, _  = doncc(imgs[j,:,:,:], tmp)
        #cl.visual.save(imgs[:,:,0], "dump/eval/image")
        #cl.visual.save(tmp[:,:,0], "dump/eval/template")
        #cl.visual.save(ncc, "dump/eval/ncc")

        if np.amax(np.abs(pos-locs[j,:]))>20:
            #print('err')
            #print(ncc[pos[0], pos[1]],ncc[locs[j,0], locs[j,1]])
            cl.visual.save(imgs[j,:,:,0], "dump/eval/image")
            cl.visual.save(tmp[:,:,0], "dump/eval/template")
            cl.visual.save(ncc, "dump/eval/ncc")
            person = input('next: ')
            err += 1
    return err

def process_ncc(image):
    #step = 384
    #W = step*4
    #H = step*2
    if image.shape[-1]==1:
        image = np.repeat(image, 3, axis=2)
    image = image/255.0
    image_new = infer.process(image)
    #image_new = cl.image_processing.read_without_borders_3d(np.expand_dims(image, -1),
    #                                [0, 0, 0],
    #                                [W, H, 3])

    return image_new

def process(imgs, tmps):
    #print(imgs.mean())
    #print(tmps.mean())
    #print(imgs[:10,:10])
    #exit()
    imgs = imgs/(255.0)
    tmps = tmps/(255.0)

    imgs_new = model.process({features["inputs"]: imgs}, [features["outputs"]])[0]
    tmps_new = model.process({features["inputs"]: tmps}, [features["outputs"]])[0]
    imgs_new = imgs_new[:,:s_width, :s_width, :]
    tmps_new = tmps_new[:,:s_width, :s_width, :]

    return imgs_new, tmps_new

def evaluate(model_to_use=False):
    err = np.zeros((size, 4))
    count = 0
    for i in range(size):
        # Load data
        imgs, tmps, locs = get_batch()

        #err[i, 0] += get_wrong_matches(imgs, imgs, locs)
        err[i, 1] += get_wrong_matches(imgs, tmps, locs)

        imgs_t, tmps_t = process(imgs, tmps)

        #err[i, 2] += get_wrong_matches(imgs_t, imgs_t, locs)
        err[i, 3] += get_wrong_matches(imgs_t, tmps_t, locs)
        #exit()
        count += BATCH_SIZE

        print(i,  err[:i, :].mean(axis=0)/BATCH_SIZE)

    err = err/BATCH_SIZE
    mean = err.mean(axis=0) # per batch
    std = err[250:, :].std(axis=0) # per batch

    #print('raw_self', mean[0], "+-"+str(std[0]/np.sqrt(1250)))
    print('raw', mean[1],"+-"+str(std[1]/np.sqrt(1250)))
    #print('model_self', mean[2],"+-"+str(std[2]/np.sqrt(1250)))
    print('NCCNet', mean[3],"+-"+str(std[3]/np.sqrt(1250)))

if __name__ == '__main__':
    model = cl.Graph(directory=MODEL_DIR)
    evaluate(True)
else:
    infer = cl.Infer(batch_size = 4,
                  width = 512,
                  n_threads = 8,
                  model_directory = MODEL_DIR,
                  cloud_volume = False,
                  features=features,
                  voxel_offset = (0, 0),
                  crop_size=60,
                  output_dim=output_dim)# FIXME Change according to the model
