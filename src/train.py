import cavelab as cl
from model import build
from data import Data
from time import time
import numpy as np
#from evaluation import get_batch


# April 28
# Work on evaluation
# - Setup validation dataset
# - Explore object tracking
# Work on benchmarking
# - Connect Matlab to Python
# - Run OTB and do tricks?

#Loop
# Get more data 10K -> 80K
# Train Multilayer
# Train Multiscale (given time)

def train(hparams):
    #Data input
    d = Data(hparams)

    model = build(hparams)
    sess = cl.tf.global_session().get_sess()
    cl.tf.global_session().add_log_writers('/projects/NCCNet/log/'+hparams.name+'/',
                                            hparams=hparams,
                                            clean_first=True)
    test_data = []
    crop_coef = 0
    hardness = 0.25
    save = True

    try:
        for i in range(hparams.steps):

            a = time()
            image, template, label = d.get_batch()
            image = image*255.0
            template = template*255.0

            model_run = [model.train_step,
                        model.l,
                        model.cnn_image_big,
                        model.cnn_templ_big,
                        cl.tf.global_session().merged]

            feed_dict = {
                            model.image: image,
                            model.template: template,
                            model.similar: label,
                            model.crop_coef: crop_coef
                        }

            step = sess.run(model_run, feed_dict=feed_dict, run_metadata=cl.tf.global_session().run_metadata)
            c = time()

            #Curriculum learning
            if abs(step[1])>hardness and label[0]>0 and crop_coef<5:
                #crop_coef += 1
                hardness += 0.05

            if i%hparams.log_iterations == 0:
                a1 = time()
                cl.tf.global_session().log_save(cl.tf.global_session().train_writer, step[-1], i)
                a2 = time()

            if i%hparams.eval_iterations == 0:
                b1 = time()
                evaluate(model, test_data, hparams.testing_steps, i)
                #try:

                #except:
                #    pass
                b2 = time()

            if((step[1]==float("Inf") or step[1]==0) and save):
                save = False
                for j in range(8):
                    cl.visual.save(image[j], "dump/img/source/"+str(j))
                    for k in range(4):
                        cl.visual.save(step[2][j,:,:,k], "dump/img/f"+str(k)+"/"+str(j))
                        cl.visual.save(step[3][j,:,:,k], "dump/tmp/f"+str(k)+"/"+str(j))
                    cl.visual.save(template[j], "dump/tmp/template/"+str(j))
                raise Exception('error')
            b = time()
            print('iteration',i, format(b-a, '.2f'), format(step[1], '.2f'), np.mean(label) )
    except KeyboardInterrupt():
        print('exiting')
    finally:
        cl.tf.global_session().model_save()
        print('saved')
        cl.tf.global_session().close_sess()


# write this
def evaluate(model, test_data, testing_steps, i):
    return

if __name__ == "__main__":
    hparams = cl.hparams(name="default")
    train(hparams)
