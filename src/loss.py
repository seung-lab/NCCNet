import tensorflow as tf
import cavelab as cl
import numpy as np

def loss(p, similar, hparams, eps = 0.001, name='loss'):

    #Get maximum p and mask the point
    p_max = tf.reduce_max(p, axis=[1,2], keep_dims=True)
    mask_p = tf.cast(p>p_max-tf.constant(eps), tf.float32)

    #Design the shape of the mask
    p_shape = tf.shape(p)
    mask = tf.ones([hparams.radius*2, hparams.radius*2, p_shape[3]], tf.float32)

    mask_p = tf.nn.dilation2d(mask_p, mask, [1,1,1,1], [1,1,1,1], 'SAME')
    mask_p = tf.to_float(mask_p)<=tf.constant(1, dtype='float32')
    mask_p = tf.cast(mask_p, dtype='float32')

    # Take care about second distance
    p_2 = tf.multiply(mask_p,p)
    p_max_2 = tf.reduce_max(p_2, axis=[1,2], keep_dims=True)

    mask_p = tf.multiply(mask_p, p)
    l = -(p_max-p_max_2)

    #Add more sanity checks
    #to_update = tf.logical_and(tf.reduce_all(tf.is_finite(l)), tf.greater(tf.reduce_max(l), -0.1))

    full_loss = tf.identity(l, name='full_loss')

    l = tf.where(tf.reduce_max(similar)>0, l, tf.abs(p_max), name=None)
    l = tf.reduce_mean(l)

    p_max = tf.reduce_mean(p_max)
    p_max_2 = tf.reduce_mean(p_max_2)

    # Summarize into tensorboard
    with tf.name_scope(name):
        tf.summary.scalar('loss', l)
        tf.summary.scalar('distance', p_max-p_max_2)
        tf.summary.scalar('max',  p_max)
        tf.summary.scalar('second_max',  p_max_2)

    return l


def binary_entropy(correlogram, similar, hparams, name="loss", alpha=10.0, threshold=0.05):
    correlogram_power = tf.exp(alpha*correlogram)
    smoothmax = tf.multiply(correlogram, correlogram_power)/tf.reduce_sum(correlogram_power)
    smoothmax = tf.cast(smoothmax<threshold, dtype=tf.int32)
    smoothmax = tf.where(tf.reduce_max(similar)>0.0,
                            smoothmax,
                            tf.zeros(tf.shape(smoothmax), dtype=tf.int32), name=None)
    correlogram, kernel, b = cl.tf.layers.conv_one_by_one(correlogram, 2)
    print(correlogram.get_shape())
    print(smoothmax.get_shape())
    loss, labels = cl.tf.loss.soft_cross_entropy(correlogram, smoothmax)
    tf.summary.scalar('loss',loss)
    return loss
