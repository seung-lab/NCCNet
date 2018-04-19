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
    g.xs, g.ys = cl.models.SiameseFusionNet(g.image, g.template, hparams.resize_conv, hparams.kernels_shape)
    g.ps = []
    ls = 0
    levels = len(g.xs)
    for i in range(levels):
        shape = hparams.kernels_shape[i]
        shape = [shape[0], shape[1], shape[3], hparams.output_layer]

        g.xs[i], g.ys[i] = cl.tf.layers.conv_block_dual(g.xs[i], g.ys[i], shape, activation = tf.tanh)
        g.xs[i], g.ys[i] = tf.identity(g.xs[i], name='output/image_'+str(i)), tf.identity(g.ys[i], name='output/template_'+str(i))

        crop = (levels-i)*10*g.crop_coef
        t_crop = (levels-i)*5*g.crop_coef

        cl.tf.metrics.scalar(crop, name='crop/image')
        cl.tf.metrics.scalar(t_crop, name='crop/template')

        ishp = g.xs[i].get_shape()
        tshp = g.ys[i].get_shape()

        swidth = hparams.features['search_raw']['width']/(2**i)
        twidth = hparams.features['template_raw']['width']/(2**i)

        g.cnn_image_big = g.xs[i][:, crop:swidth-crop, crop:swidth-crop, :]
        g.cnn_templ_big = g.ys[i][:, t_crop:twidth-t_crop, t_crop:twidth-t_crop, :]

        # Save to Tensorboard
        cl.tf.metrics.image_summary(tf.squeeze(g.xs[i][:, :, :,0]), 'pred/image_mip='+str(i))
        cl.tf.metrics.image_summary(tf.squeeze(g.ys[i][:, :, :,0]), 'pred/template_mip='+str(i))

        if hparams.output_layer>1:

            image_feature = tf.transpose(tf.squeeze(g.xs[i][0, :, :, :]), perm=[2,0,1])
            temp_feature = tf.transpose(tf.squeeze(g.ys[i][0, :, :, :]), perm=[2,0,1])

            cl.tf.metrics.image_summary(image_feature, 'features/image_mip='+str(i))
            cl.tf.metrics.image_summary(temp_feature, 'features/template_mip='+str(i))

        # Loss
        g.p = cl.tf.layers.batch_normxcorr(g.cnn_image_big, g.cnn_templ_big)
        g.p = tf.reduce_mean(g.p, axis=3, keep_dims=True)

        #Resize
        if i>0 and False:
            shape = tf.shape(g.ps[0])#.get_shape()
            g.p = tf.image.resize_images(g.p, size=[264, 264], method=1, align_corners=True)
        g.ps.append(g.p)

        #g.p = tf.add_n(g.ps)/3
        cl.tf.metrics.image_summary(g.p[:,:,:,0], 'pred/normxcorr_large_mip='+str(i), resize=False)
        l = loss.loss(g.p, g.similar, radius=hparams.radius, name='loss_'+str(i))
        ls += l

    # Output
    g.l = ls/3
    tf.summary.scalar('loss/loss', g.l)
    g.train_step = tf.train.AdamOptimizer(hparams.learning_rate).minimize(g.l)

    return g

    #ncc = cl.tf.base_layers.batch_normxcorr(g.cnn_image, g.cnn_templ)
    #ncc = tf.reduce_mean(ncc, axis=3, keep_dims=False)
    #cl.tf.metrics.image_summary(ncc, 'pred/normxcorr_large', resize=False)
