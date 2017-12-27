# use training data
import cavelab as cl
from model import build
from data import Data
import numpy as np
import tensorflow as tf
#FIXME this to go to hparams
hparams = cl.hparams(name="evaluation")
elastic_deform = cl.image_processing.elastic_transformations(500, 30)


source_width = 384
BATCH_SIZE = 8
size = 1250
width = 120

#MODEL_DIR = 'logs/NCCNetv1/'
MODEL_DIR = 'logs/NCCNetv37_two/'
#features = { "inputs":"input/image:0", "outputs": "Passes/image_transformed:0"}
features = {"inputs": "image:0", "outputs": "add_96:0"}


model = cl.Graph(directory=MODEL_DIR)
tf.get_default_graph().clear_collection("queue_runners")
tf.get_default_graph().clear_collection("local_variables")
d = Data(hparams)

# Init
hparams = cl.hparams(name="evaluation")
d = Data(hparams, random=False)

elastic_deform = cl.image_processing.elastic_transformations(700, 28)


def get_batch():
    image, _, _ = d.get_batch()
    template = np.array(elastic_deform(image))
    #loc a= np.array(np.random.randint(source_width-width, size=2*BATCH_SIZE)).reshape((BATCH_SIZE, 2))
    loc = np.repeat((source_width-width)/2, BATCH_SIZE*2).reshape((BATCH_SIZE, 2))
    return image, template, loc

def doncc(image, template):

    nccs = []
    locs = []
    for channel in range(image.shape[-1]):
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
    imgs = imgs[:,:source_width, :source_width, :]
    tmps = tmps[:,:source_width, :source_width, :]
    err =0
    for j in range(BATCH_SIZE):
        tmp = tmps[j,locs[j,0]:locs[j,0]+width, locs[j, 1]:locs[j, 1]+width, :]
        ncc, _, pos, _  = doncc(imgs[j,:,:,:], tmp)
        if np.amax(np.abs(pos-locs[j,:]))>10:
            #print(ncc[pos[0], pos[1]],ncc[locs[j,0], locs[j,1]])
            #cl.visual.save(imgs[j,:,:,0], "dump/eval/image")
            #cl.visual.save(tmp, "dump/eval/template")
            #cl.visual.save(ncc, "dump/eval/ncc")
            #person = raw_input('next: ')
            err += 1
    return err

def process(imgs, tmps):
    imgs_new = model.process({features["inputs"]: imgs}, [features["outputs"]])[0]
    tmps_new = model.process({features["inputs"]: tmps}, [features["outputs"]])[0]
    imgs_new = imgs_new[:,:source_width, :source_width, :]
    tmps_new = tmps_new[:,:source_width, :source_width, :]
    return imgs_new, tmps_new


def evaluate(model_to_use=False):
    err = np.zeros((size, 4))
    count = 0
    for i in range(size):
        # Load data
        imgs, tmps, locs = get_batch()

        err[i, 0] += get_wrong_matches(imgs, imgs, locs)
        err[i, 1] += get_wrong_matches(imgs, tmps, locs)

        imgs_t, tmps_t = process(imgs, tmps)
        err[i, 2] += get_wrong_matches(imgs_t, imgs_t, locs)
        err[i, 3] += get_wrong_matches(imgs_t, tmps_t, locs)

        count += BATCH_SIZE

        print(i,  err[:i, :].mean(axis=0)/BATCH_SIZE)

    err = err/BATCH_SIZE
    mean = err.mean(axis=0) # per batch
    std = err[250:, :].std(axis=0) # per batch

    print('raw_self', mean[0], "+-"+str(std[0]/np.sqrt(1250)))
    print('raw_elastic', mean[1],"+-"+str(std[1]/np.sqrt(1250)))
    print('model_self', mean[2],"+-"+str(std[2]/np.sqrt(1250)))
    print('model_elastic', mean[3],"+-"+str(std[3]/np.sqrt(1250)))


evaluate(True)
