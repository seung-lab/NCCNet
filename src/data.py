import numpy as np
import cavelab as cl

# Data preprocessing
similar = True
class Data():
    def __init__(self, hparams, random=True):
        self.batch_size = hparams.batch_size
        self.data = cl.tfdata(hparams.train_file,
                                random_order = random,
                                batch_size = self.batch_size,
                                features=hparams.features,
                                flipping=hparams.augmentation["flipping"],
                                random_brightness=hparams.augmentation["random_brightness"],
                                random_elastic_transform=hparams.augmentation["random_elastic_transform"])
        self.similar = True

    def get_batch(self):

        template, image = self.data.get_batch()

        if not self.check_validity(image, template):
            return self.get_batch()

        # Similar or Disimilar
        if not self.similar:
            search_space, template = self.dissimilar(image, template)
        label = np.ones((self.batch_size),dtype=np.float) if self.similar else -1*np.ones((self.batch_size),dtype=np.float)

        if self.similar:
            label = 2*(np.random.rand(self.batch_size)>0)-1 #FIXME add this variable - Probability of misinterpreting the label
        self.similar = not self.similar

        return image, template, label

    def dissimilar(self, images, templates):
        length = templates.shape[0]-1
        temp = np.array(templates[0])
        templates[0:length] = templates[1:length+1]
        templates[length] = temp
        return images, templates

    def check_validity(self, image, template):
        t = np.array(template.shape)
        if np.any(np.sum(image<0.01, axis=(1,2)) >= t[1]*t[2]) or image.shape[0]<self.batch_size:
            return False

        #if np.any(template.var(axis=(1,2))<0.0001):
        #    print("hey")
        #    return False

        return True
