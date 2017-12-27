import cavelab as cl
import loss
import tensorflow as tf

def build(hparams):

    g = cl.tf.Graph()

    # Define inputs
    g.image = tf.placeholder(tf.float32, shape=[hparams.batch_size, hparams.features['search_raw']['width'], hparams.features['search_raw']['width']], name='image')
    g.template = tf.placeholder(tf.float32, shape=[hparams.batch_size, hparams.features['template_raw']['width'], hparams.features['template_raw']['width']], name='template')
    g.similar = tf.placeholder(tf.float32, shape=[hparams.batch_size], name='similarity')
    g.crop_coef = tf.placeholder(tf.int32, shape=[], name='crop_coef')

    # Add to metrics
    cl.tf.metrics.image_summary(g.image, 'input/image')
    cl.tf.metrics.image_summary(g.template, 'input/template')

    # Model
    g.cnn_image, g.cnn_templ = cl.models.SiameseFusionNet(g.image, g.template, hparams.resize_conv, hparams.kernels_shape)

    shape = hparams.kernels_shape[0]
    shape = [shape[0], shape[1], shape[3], hparams.output_layer]

    g.cnn_image, g.cnn_templ = cl.tf.layers.conv_block_dual(g.cnn_image, g.cnn_templ , shape, activation = tf.tanh)
    g.cnn_image, g.cnn_templ = tf.identity(g.cnn_image, name='output/image'), tf.identity(g.cnn_templ, name='output/template')

    crop = 20*g.crop_coef
    t_crop = 10*g.crop_coef

    cl.tf.metrics.scalar(crop, name='crop/image')
    cl.tf.metrics.scalar(t_crop, name='crop/template')

    ishp = g.cnn_image.get_shape()
    tshp = g.cnn_image.get_shape()

    g.cnn_image_big = g.cnn_image[:, crop:hparams.features['search_raw']['width']-crop, crop:hparams.features['search_raw']['width']-crop, :]
    g.cnn_templ_big = g.cnn_templ[:, t_crop:hparams.features['template_raw']['width']-t_crop, t_crop:hparams.features['template_raw']['width']-t_crop, :]

    # Save to Tensorboard
    cl.tf.metrics.image_summary(tf.squeeze(g.cnn_image[:, :, :,0]), 'pred/image')
    cl.tf.metrics.image_summary(tf.squeeze(g.cnn_templ[:, :, :,0]), 'pred/template')

    if hparams.output_layer>1:

        image_feature = tf.transpose(tf.squeeze(g.cnn_image[0, :, :, :]), perm=[2,0,1])
        temp_feature = tf.transpose(tf.squeeze(g.cnn_templ[0, :, :, :]), perm=[2,0,1])

        cl.tf.metrics.image_summary(image_feature, 'features/image')
        cl.tf.metrics.image_summary(temp_feature, 'features/template')

    # Loss
    g.p = cl.tf.layers.batch_normxcorr(g.cnn_image_big, g.cnn_templ_big)
    g.p = tf.reduce_mean(g.p, axis=3, keep_dims=True)
    cl.tf.metrics.image_summary(g.p[:,:,:,0], 'pred/normxcorr_large', resize=False)
    g.l = loss.loss(g.p, g.similar, hparams, name='loss')

    cl.tf.metrics.scalar(tf.reduce_max(tf.gradients(g.l, g.cnn_image_big)), name='gradient_dl_dimage')
    cl.tf.metrics.scalar(tf.reduce_max(tf.gradients(g.l, g.cnn_templ_big)), name='gradient_dl_dtemplate')

    # Output
    g.train_step = tf.train.AdamOptimizer(hparams.learning_rate).minimize(g.l)

    return g

    #ncc = cl.tf.base_layers.batch_normxcorr(g.cnn_image, g.cnn_templ)
    #ncc = tf.reduce_mean(ncc, axis=3, keep_dims=False)
    #cl.tf.metrics.image_summary(ncc, 'pred/normxcorr_large', resize=False)
