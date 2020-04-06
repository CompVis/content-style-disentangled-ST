import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import cv2

import tensorflow.contrib.layers as tflayers

from utils import *


def batch_norm(input, is_training=True, name="batch_norm"):
    x = tflayers.batch_norm(inputs=input,
                            scale=True,
                            is_training=is_training,
                            trainable=True,
                            reuse=None)
    return x


def instance_norm(input, name="instance_norm", is_training=True):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        normalized = (input-mean)*tf.rsqrt(variance + epsilon)
        return scale*normalized + offset


def group_norm(input, name="group_norm", is_training=True, G=16):
    epsilon = 1e-5
    with tf.variable_scope(name):
        N, H, W, C = tf.shape(input)[0], tf.shape(input)[1], tf.shape(input)[2], tf.shape(input)[3]
        depth = input.get_shape()[3]
        #N, H, W, C = input.get_shape().as_list()
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        # input = tf.reshape(input, tf.concat(values=[N, H, W, C // G, G], axis=0))
        input = tf.reshape(input, [N, H, W, C // G, G])
        mean, var = tf.nn.moments(input, [1, 2, 3], keep_dims = True)
        input = (input - mean) / tf.sqrt(var + epsilon)
        input = tf.reshape(input, [N, H, W, C])
        return input * scale + offset


def local_group_norm(input, style=None, name="local_group_norm", is_training=True, G=32, window_size=32):
    epsilon = 1e-5
    with tf.variable_scope(name):
        # print("\n\nIn local_group_norm", name)
        # print("input:", input)
        # print("style:", style)
        N, H, W, C = tf.shape(input)[0], tf.shape(input)[1], tf.shape(input)[2], tf.shape(input)[3]

        depth = input.get_shape()[3]
        if style is None:
            scale = tf.get_variable("scale",
                                    [depth],
                                    initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
            offset = tf.get_variable("offset",
                                     [depth],
                                     initializer=tf.constant_initializer(0.0))
        else:
            scale = \
                tf.layers.dense(inputs=style,
                                units=depth,
                                activation=lrelu,
                                name='scale_1')
            offset = tf.layers.dense(inputs=style,
                                     units=depth,
                                     activation=lrelu,
                                     name='offset_1')
            scale = tf.expand_dims(scale, axis=1)
            scale = tf.expand_dims(scale, axis=2)
            offset = tf.expand_dims(offset, axis=1)
            offset = tf.expand_dims(offset, axis=2)

        # print("scale computed:", scale)
        # print("offset computed:", offset)
        input_reshaped = tf.reshape(input, [N, H, W, C // G, G])
        means = tf.reduce_mean(input_reshaped, axis=-1, )
        means = tf.nn.separable_conv2d(input=means,
                                       depthwise_filter=tf.ones(shape=[window_size, window_size, C // G, 1],
                                                                dtype=tf.float32) / tf.cast(window_size * window_size,
                                                                                            dtype=tf.float32),
                                       pointwise_filter=tf.expand_dims(
                                           input=tf.expand_dims(input=tf.eye(num_rows=C // G,
                                                                             num_columns=C // G,
                                                                             dtype=tf.float32),
                                                                axis=0),

                                           axis=0),
                                       strides=[1, window_size // 2, window_size // 2, 1],
                                       # strides=[1, 4, 4, 1],
                                       padding='VALID')
        means = tf.image.resize_images(images=means,
                                       size=tf.shape(input)[1:3],
                                       method=tf.image.ResizeMethod.BILINEAR)
        means = tf.tile(tf.expand_dims(means, axis=-1), [1, 1, 1, 1, G])
        stds = input_reshaped - means
        stds = tf.square(stds)
        stds = tf.reduce_mean(stds, axis=-1)

        stds = tf.nn.separable_conv2d(input=stds,
                                      depthwise_filter=tf.ones(shape=[window_size, window_size, C // G, 1],
                                                               dtype=tf.float32) / tf.cast(window_size * window_size,
                                                                                           dtype=tf.float32),
                                      pointwise_filter=tf.expand_dims(
                                          input=tf.expand_dims(input=tf.eye(num_rows=C // G,
                                                                            num_columns=C // G,
                                                                            dtype=tf.float32),
                                                               axis=0),

                                          axis=0),
                                      strides=[1, window_size // 2, window_size // 2, 1],
                                      # strides=[1, 4, 4, 1],
                                      padding='VALID')
        stds = tf.image.resize_images(images=stds,
                                      size=tf.shape(input)[1:3],
                                      method=tf.image.ResizeMethod.BILINEAR)
        stds = tf.tile(tf.expand_dims(stds, axis=-1), [1, 1, 1, 1, G])
        input = (input_reshaped - means) / tf.sqrt(stds + epsilon)
        input = tf.reshape(input, [N, H, W, C])
        # print("output:", input)
        # print("local_group_norm", name, " is finished\n\n")
        return tf.multiply(input, scale) + offset


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d", activation_fn=None):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=activation_fn,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    # Like this articles suggests: https://distill.pub/2016/deconv-checkerboard/, we upsample
    # tensol like an image at first and then apply convolutions
    with tf.variable_scope(name):
        input_ = tf.image.resize_images(images=input_,
                                        size=tf.shape(input_)[1:3] * s,
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # That is optional
        return conv2d(input_=input_, output_dim=output_dim, ks=ks, s=1, padding='SAME')


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def tf_mean_cov(x):
    """
    Computes mean and covariance of rank 2 tensor x
    Args:
        x: input tensor of rank2
    Returns:
        mean and variance
    """

    mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x) / tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return mean_x, cov_xx


def get_clsf_acc(in_, labels_):
    """
    Computes accuracy
    Args:
        in_:
        labels_:

    Returns:

    """
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(in_, -1), tf.argmax(labels_, -1)),
                                  tf.float32))

